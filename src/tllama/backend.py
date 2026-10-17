import time
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
import shutil
import itertools

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

from typing import Dict, Optional, Any, List, Mapping

from tllama.config import BackendConfig, ConfigError, load_backend_config_from_env
from tllama.helpers.common import normalize_keep_alive
from tllama.helpers.gguf_metadata import read_gguf_metadata, build_model_metadata_payload
from tllama.helpers.model_toml import (
    TomlModelError,
    VirtualModelSpec,
    parse_model_toml,
    render_model_toml,
    resolve_repo_relative_path,
    write_model_toml,
)
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides
from tllama.native_config import resolve_worker_config
from tllama.native_params import freeze_values
from tllama.residency import NativeResidency, ModelDefinition, ResidencyClosed, _settle_owned
from tllama.worker import NativeWorker
from tllama.helpers.metadata_cache import (
    build_model_file_fingerprint,
    load_metadata_cache,
    save_metadata_cache,
    load_digest_cache,
    save_digest_cache,
)

__all__ = ('model_manager', 'load_backend_config_from_env')

logger = logging.getLogger(__name__)

# Read size used when hashing model files. Large enough to keep sequential
# reads efficient on multi-gigabyte GGUF files.
_CONTENT_HASH_CHUNK_SIZE = 8 * 1024 * 1024

# Outcome of asking HuggingFace about a file. MISSING and UNKNOWN must stay
# distinct: the first means the repository answered and has no such path, so
# the pull can be refused before anything touches the disk. The second means
# the question could not be asked, which is never a reason to refuse.
HF_LOOKUP_FOUND = "found"
HF_LOOKUP_MISSING = "missing"
HF_LOOKUP_UNKNOWN = "unknown"


@dataclass(frozen=True)
class HfFileLookup:
    status: str
    sha256: str | None = None
    size: int | None = None


class PullProgress:
    """Byte counters for a download in progress, updated from a worker thread.

    `hf_hub_download` runs off the event loop via `asyncio.to_thread`, so the
    streaming response reads this from the async side while
    `_HfTqdmProgressAdapter` writes to it from the download thread. A lock
    guards the pair of ints; nothing here ever blocks.
    """

    def __init__(self, total: int = 0):
        self._lock = threading.Lock()
        self._total = total
        self._completed = 0

    def set_total(self, total: int | None) -> None:
        if not total:
            return
        with self._lock:
            self._total = int(total)

    def add(self, n: int) -> None:
        if not n:
            return
        with self._lock:
            self._completed += int(n)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._completed, self._total


def _hf_tqdm_class_for(progress: "PullProgress"):
    """Build a tqdm-compatible class that reports into `progress` instead of drawing a bar.

    `hf_hub_download` only ever calls `cls(total=..., initial=..., ...)`, then
    `.update(n)` per chunk, and uses the instance as a context manager. It
    does not require a `tqdm` subclass -- see huggingface_hub's
    `_create_progress_bar`, which calls a non-tqdm class with plain kwargs.
    """

    class _HfTqdmProgressAdapter:
        def __init__(self, *args, total: int | None = None, initial: int = 0, **kwargs):
            progress.set_total(total)
            progress.add(initial)

        def update(self, n: int = 1) -> None:
            progress.add(n)

        def close(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    return _HfTqdmProgressAdapter


@dataclass(frozen=True)
class CachedMetadataEntry:
    fingerprint: str
    cached_at_monotonic: float
    value: Dict[str, Any]


class ModelManager:
    def __init__(self, config: BackendConfig | None = None, *, native_worker_factory=NativeWorker,
                 native_launch_options: Mapping[str, Any] | None = None):
        self.config = config or load_backend_config_from_env()

        self._metadata_lock = asyncio.Lock()

        self.models_dir = Path(self.config.models_dir)
        self.metadata_cache_dir = self.models_dir / ".tllama" / "metadata-cache"

        self._janitor_task: asyncio.Task | None = None

        self._metadata_cache: Dict[str, CachedMetadataEntry] = {}

        # Paths already reported as sitting outside the reference scheme, so a
        # stray file does not produce a warning on every model listing.
        self._reported_unusable_files: set[str] = set()

        # Validate global cache types at startup without loading a model.
        self._validate_global_kv_cache_types()

        # Validated using the native contract here rather than in the raw
        # environment collector, and before load time so an
        # unknown parameter or an unusable value stops the server instead
        # of surfacing on the first request that needs a model.
        coerce_runtime_overrides(
            self.config.runtime_overrides
        )

        # Validate sampling configuration before any request admits a worker.
        coerce_sampling_overrides(
            self.config.sampling_overrides
        )

        self.hf_models_dir = self.models_dir / "HuggingFace"
        self.local_models_dir = self.models_dir / "Local"
        self.tllama_models_dir = self.models_dir / "TLlama"

        self._stopping = False
        self._shutdown_task: asyncio.Task | None = None
        self._native_factory = native_worker_factory
        options = dict(native_launch_options or {})
        unknown = set(options) - {"environment", "executable", "load_timeout", "request_timeout"}
        if unknown:
            raise ConfigError(f"Unknown native supervisor options: {', '.join(sorted(unknown))}")
        self._native_launch_options = freeze_values(options)
        self._native_generation_numbers = itertools.count(1)
        self._native = self._new_native_residency()

    def _new_native_residency(self):
        return NativeResidency(self._resolve_native_definition,
                               max_loaded_models=self.config.max_loaded_models,
                               worker_factory=self._native_factory,
                               generation_numbers=self._native_generation_numbers)

    def _check_running(self):
        if self._stopping:
            raise ResidencyClosed("ModelManager is shutting down")

    async def _resolve_native_definition(self, model_name, num_ctx):
        # Definition/path/header work is independent of the residency lock.
        return await asyncio.to_thread(self._resolve_native_definition_sync, model_name, num_ctx)

    def _resolve_native_definition_sync(self, model_name, num_ctx):
        for _ in range(3):
            before = self._toml_fingerprint_for_reference(model_name)
            spec = self._resolve_virtual_model_spec(model_name)
            if spec is None or spec.llm_model is None:
                raise FileNotFoundError(f"Model '{model_name}' is not registered/imported in {self.models_dir}")
            model = resolve_repo_relative_path(spec.llm_model, self.models_dir)
            if not model.is_file():
                raise FileNotFoundError(f"Model '{model_name}' does not point at an existing GGUF")
            mmproj = self.resolve_mmproj_path(spec, model_name)
            file_before = {"model": build_model_file_fingerprint(model),
                           "projector": build_model_file_fingerprint(mmproj) if mmproj else None}
            info = self._build_model_file_info_from_path(model)
            if info is None:
                raise FileNotFoundError(f"Model '{model_name}' does not point at an existing GGUF")
            # Listing does not hash a potentially changed GGUF behind a live
            # generation. Retain only a content digest already cached for it.
            info['content_sha256'] = (load_digest_cache(self.metadata_cache_dir, model) or {}).get('content_sha256', '')
            metadata_scanned = False
            try:
                metadata = load_metadata_cache(self.metadata_cache_dir, model)
                if metadata is None:
                    metadata = self._get_model_metadata_sync(str(model))
                    metadata_scanned = metadata is not None
                    metadata = metadata or {}
            except Exception as exc:
                # The native loader remains authoritative for model validity.
                # Missing optional header recommendations must not invent defaults.
                logger.warning("Native metadata scan failed for %s: %s", model_name, exc)
                metadata = {}
            after = self._toml_fingerprint_for_reference(model_name)
            file_after = {"model": build_model_file_fingerprint(model),
                          "projector": build_model_file_fingerprint(mmproj) if mmproj else None}
            if before != after or file_before != file_after:
                continue
            resolved = resolve_worker_config(self.config, model.resolve(), spec, num_ctx=num_ctx,
                                             mmproj=mmproj.resolve() if mmproj else None,
                                             **self._native_launch_options)
            if metadata_scanned:
                # Admission must warm the same header cache as /api/show.
                # Otherwise every request rescans a cold GGUF, even while its
                # worker remains loaded. Preserve the scanned fingerprint if
                # the weights change again during the optional cache write.
                save_metadata_cache(self.metadata_cache_dir, model_name, model, metadata,
                                    fingerprint=file_after['model'])
            fingerprint = {"toml": after, **file_after}
            return ModelDefinition(model_name, resolved, fingerprint, info, metadata)
        raise RuntimeError(f"Model definition '{model_name}' kept changing while it was resolved")

    @asynccontextmanager
    async def acquire_native_model(self, model_name: str, num_ctx: int | None = None,
                                   keep_alive: str | int | float | None = None):
        """Hold one resolved worker generation through slot wait and inference."""
        self._check_running()
        self._ensure_janitor_running()
        seconds = self.resolve_keep_alive(keep_alive)
        async with self._native.lease(model_name, num_ctx=num_ctx, keep_alive_seconds=seconds) as lease:
            yield lease

    async def preload_native_model(self, model_name: str, num_ctx: int | None = None,
                                   keep_alive: str | int | float | None = None):
        self._check_running()
        self._ensure_janitor_running()
        return await self._native.preload(model_name, num_ctx=num_ctx,
                                          keep_alive_seconds=self.resolve_keep_alive(keep_alive))

    async def unload_native_model(self, model_name: str):
        self._check_running()
        return await self._native.unload(model_name)

    async def wait_native_closed(self, generation: int):
        await self._native.wait_closed(generation)

    def native_snapshot(self):
        """Internal lifecycle diagnostics; not the public /api/ps memory shape."""
        return self._native.snapshot()

    def native_reporting_snapshot(self):
        """Public reporting reads native ownership without touching inference."""
        return self._native.reporting_snapshot()


    def ensure_storage(self) -> None:
        """Create the model store layout.

        Kept out of the constructor deliberately. A ModelManager is built at
        import time, so doing this there meant that merely importing the
        package created directories under the configured models path, by
        default somewhere in /var/lib. That fired for anything that imports
        the module without intending to run a server: a test, a documentation
        build, a one-off script. It also needed write access to that path
        before the process had done anything.

        Called from start(), so a running server behaves exactly as before.
        """
        for directory in (
            self.models_dir,
            self.metadata_cache_dir,
            self.hf_models_dir,
            self.local_models_dir,
            self.tllama_models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


    def resolve_keep_alive(self, keep_alive: str | int | float | None) -> int | None:
        """Resolve a request-level keep_alive to seconds.

        None means the caller did not ask for anything, in which case the
        configured TLLAMA_KEEP_ALIVE applies. Any other value overrides it,
        which is how Ollama treats OLLAMA_KEEP_ALIVE.

        Public because the HTTP layer needs the very same answer to decide
        whether a request means "unload now". Computing it in two places is
        what let the two drift apart.
        """
        if keep_alive is None:
            keep_alive = self.config.keep_alive

        return normalize_keep_alive(keep_alive)

    def _build_model_file_info_from_path(self, file_path: Path) -> Optional[Dict[str, Any]]:
        if not file_path.exists():
            return None

        stats = file_path.stat()

        hash_sha256 = sha256()
        hash_sha256.update(file_path.name.encode("utf-8"))
        hash_sha256.update(str(stats.st_size).encode("utf-8"))
        hash_sha256.update(str(stats.st_mtime).encode("utf-8"))

        return {
            "id": file_path.stem,
            "filename": file_path.name,
            "path": str(file_path),
            "size": stats.st_size,
            "mtime": int(stats.st_mtime),
            "sha256": hash_sha256.hexdigest(),
        }

    def _toml_path_for_reference(self, model_ref: str) -> Optional[Path]:
        """Where a virtual model's .toml would live for this reference, if any.

        Mirrors resolve_model_storage_path's segment-based routing, but for
        .toml files at their fixed depth (spec doc §5) instead of a raw
        .gguf's own (unbounded-below for HuggingFace) depth. There is no
        Local-nested fallback for a 2-segment reference the way
        resolve_model_storage_path has for .gguf: the fixed-depth invariant
        makes that ambiguity impossible for .toml in the first place.

        Returns None for a reference with more than 3 segments -- a .toml
        never sits deeper than that, so there is nowhere to look.
        """
        parts = self._split_model_reference(model_ref)

        if len(parts) == 1:
            base_dir = self.local_models_dir
        elif len(parts) == 2:
            base_dir = self.tllama_models_dir
        elif len(parts) == 3:
            base_dir = self.hf_models_dir
        else:
            return None

        return base_dir.joinpath(*parts[:-1], f"{parts[-1]}.toml")

    def _resolve_virtual_model_spec(self, model_ref: str) -> Optional[VirtualModelSpec]:
        """Parse the .toml for model_ref, if one exists at its fixed depth.

        Unlike the listing scan (_list_local_models_sync), which silently
        skips a broken .toml so one bad file cannot take down the whole
        listing, a name actively requested here is a single, specific ask:
        TomlModelError propagates rather than being swallowed, so a typo in
        the file surfaces clearly instead of looking identical to the model
        simply not existing.
        """
        toml_path = self._toml_path_for_reference(model_ref)
        if toml_path is None or not toml_path.is_file():
            return None

        text = toml_path.read_text(encoding="utf-8")
        return parse_model_toml(text, source=str(toml_path))

    def _toml_fingerprint_for_reference(self, model_ref: str) -> Optional[Dict[str, Any]]:
        """Identify the revision of the .toml a model would be built from.

        Deliberately the same fingerprint the metadata cache already uses
        for .gguf files -- resolved path, size, mtime_ns -- rather than a
        second idiom for the same job. Timestamp alone would do; size comes
        free from the one stat() and costs nothing to compare.

        A content hash was considered and rejected: `touch model.toml` as a
        way to force a reload is a wanted property, and hashing would
        silently ignore it.

        None when there is no .toml, which for a resident model means the
        definition has been deleted.
        """
        toml_path = self._toml_path_for_reference(model_ref)
        if toml_path is None:
            return None

        try:
            return build_model_file_fingerprint(toml_path)
        except OSError:
            return None

    def read_model_definition(self, model_ref: str) -> Optional[str]:
        """The raw text of a model's .toml, exactly as it sits on disk.

        Returned unparsed and unformatted on purpose. It is what
        `ollama show --modelfile` reports, and the workflow that makes
        that useful is fetching it, editing it and putting it back: a
        round trip through a parser would drop the comments a person
        wrote, which is the very thing the format was chosen to keep.

        None when there is no file or it cannot be read. A malformed one
        is still returned -- showing someone the broken file is how they
        find out what is wrong with it.
        """
        toml_path = self._toml_path_for_reference(model_ref)
        if toml_path is None or not toml_path.is_file():
            return None

        try:
            return toml_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", toml_path, exc)
            return None

    def _build_model_file_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            virtual_spec = self._resolve_virtual_model_spec(model_name)
        except TomlModelError:
            # A broken .toml is a real problem worth surfacing, not the
            # same as the reference simply not existing -- let it propagate.
            raise
        except ValueError:
            # An invalid reference (empty, "..", etc.) is not a model,
            # same as before .toml existed.
            return None

        if virtual_spec is not None:
            if virtual_spec.llm_model is None:
                # 'from =' has not been imported into the repo yet (spec
                # §5): there is no physical file inside the repo to report
                # on until that happens.
                return None

            target = resolve_repo_relative_path(virtual_spec.llm_model, self.models_dir)

            # Raised, not skipped: a name asked for by hand is a single
            # specific request, and reporting what is wrong with it beats
            # reporting that it does not exist.
            self.resolve_mmproj_path(virtual_spec, model_name)

            return self._build_model_file_info_from_path(target)

        # No .toml names this reference at all. Per the strict policy (spec
        # §3) that applies to loading exactly as it does to listing: a bare
        # file nothing points at is not a model, so there is nothing left
        # to fall back to here.
        return None


    def _filter_metadata_raw_for_cache(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only small scalar metadata values in cache.
        This avoids holding large/raw structures in memory.
        """
        filtered: Dict[str, Any] = {}

        for key, value in (meta or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value

        return filtered

    def _get_cached_metadata_entry(self, model_name: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        entry = self._metadata_cache.get(model_name)
        if entry is None:
            return None

        if entry.fingerprint != fingerprint:
            return None

        age_seconds = time.monotonic() - entry.cached_at_monotonic
        if age_seconds > self.config.metadata_cache_ttl_seconds:
            return None

        return entry.value

    def _set_cached_metadata_entry(self, model_name: str, fingerprint: str, value: Dict[str, Any]) -> None:
        self._metadata_cache[model_name] = CachedMetadataEntry(
            fingerprint=fingerprint,
            cached_at_monotonic=time.monotonic(),
            value=value,
        )

    def _invalidate_metadata_cache_entry(self, model_name: str) -> None:
        self._metadata_cache.pop(model_name, None)


    def _ensure_janitor_running(self) -> None:
        if self._janitor_task is None or self._janitor_task.done():
            self._janitor_task = asyncio.create_task(
                self._janitor_loop(),
                name="tllama-model-janitor",
            )


    def _compute_content_sha256(self, file_path: str | Path) -> str:
        """
        Hash the full contents of a model file.

        This is the portable identity of the model: the same bytes yield the
        same digest regardless of where the file lives, how it got there or
        when it was written. It is deliberately not derived from path, name
        or mtime.
        """
        digest = sha256()

        with open(file_path, "rb", buffering=0) as handle:
            while True:
                chunk = handle.read(_CONTENT_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)

        return digest.hexdigest()

    def _build_model_digest_sync(self, model_path: str | Path) -> Dict[str, Any]:
        return {
            "content_sha256": self._compute_content_sha256(model_path),
            "source": "local",
        }

    async def get_model_digest(self, model_path: str | Path) -> Optional[Dict[str, Any]]:
        """
        Return the content digest of a model file, computing it when missing.

        Takes a filesystem path rather than a model reference on purpose: the
        digest is a property of the bytes on disk and must not depend on name
        resolution or on the GGUF header being parseable.

        The cost of the full read is paid once. The result is stored next to
        the metadata cache in its own document, which invalidates on
        size + mtime_ns, so a changed file is rehashed and an unchanged one
        is never hashed twice.

        Returns None only when the file itself cannot be read.
        """
        file_path = Path(model_path)

        cached = await asyncio.to_thread(
            load_digest_cache,
            self.metadata_cache_dir,
            file_path,
        )
        if cached is not None:
            return cached

        try:
            digest = await asyncio.to_thread(self._build_model_digest_sync, file_path)
        except Exception as exc:
            logger.warning("Digest computation failed for %s: %s", file_path, exc)
            return None

        try:
            model_name = self._build_model_ref_from_path(file_path)
        except ValueError:
            model_name = str(file_path)

        await asyncio.to_thread(
            save_digest_cache,
            self.metadata_cache_dir,
            model_name,
            file_path,
            digest,
        )

        return digest

    def _get_model_metadata_sync(self, model_path: str) -> Optional[Dict[str, Any]]:
        meta = read_gguf_metadata(model_path)
        return build_model_metadata_payload(meta)

    async def get_model_metadata(
        self,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata without loading the full model into inference memory.

        The GGUF-derived metadata is cached (TTL in-memory, persistent on
        disk) exactly as before. A virtual model's [template]/[system]/
        [sampling] overrides, if any, are applied fresh on top of that on
        every call rather than baked into the cached value -- editing a
        .toml takes effect immediately without needing the underlying
        .gguf to also change, which is what would otherwise be needed to
        bust a cache keyed by the .gguf's own fingerprint.

        default_system_prompt is deliberately not called "system_prompt":
        it is a fallback a caller applies only when neither the request nor
        the messages it was given already carry one (client always wins),
        never a value to send outright.
        """
        metadata = await self._get_raw_model_metadata(model_name)
        if metadata is None:
            return None

        try:
            virtual_spec = self._resolve_virtual_model_spec(model_name)
        except TomlModelError:
            raise
        except ValueError:
            virtual_spec = None

        if virtual_spec is not None:
            overrides = {}
            if virtual_spec.template is not None:
                overrides["template"] = virtual_spec.template
                # A nonempty native override replaces both GGUF variants.
                # An empty override keeps native's model-template fallback.
                if virtual_spec.template:
                    overrides["tool_template"] = virtual_spec.template
            if virtual_spec.system_prompt is not None:
                overrides["default_system_prompt"] = virtual_spec.system_prompt
            if virtual_spec.sampling:
                overrides["sampling_defaults"] = virtual_spec.sampling
            if virtual_spec.stop:
                overrides["stop_defaults"] = virtual_spec.stop

            # Whether this model can see, answerable without loading it.
            # model_has_projector() asks the loaded model, which is the
            # right question when an image is already in hand; a client
            # asking what a model can do has not loaded anything and
            # should not have to.
            #
            # resolve_mmproj_path checks the file is really a projector,
            # so this says the capability is usable rather than merely
            # named. A [mmproj] with an unimported 'from' returns None and
            # is correctly not claimed yet.
            projector = self.resolve_mmproj_path(virtual_spec, model_name)
            overrides["has_projector"] = projector is not None
            projector_info = self._read_metadata_for_path(projector) if projector is not None else {}
            has_encoder = (projector_info or {}).get('metadata_raw', {}).get('clip.has_vision_encoder')
            overrides['vision_configured'] = projector is not None and has_encoder is not False

            if overrides:
                metadata = {**metadata, **overrides}

        return metadata

    async def _get_raw_model_metadata(
        self,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        The GGUF header's own metadata, unaffected by any .toml override.
        Runs off the event loop and uses a lightweight TTL cache.
        """
        model_info = self._build_model_file_info(model_name)
        if not model_info:
            return None

        fingerprint = model_info["sha256"]

        async with self._metadata_lock:
            cached_value = self._get_cached_metadata_entry(model_name, fingerprint)
            if cached_value is not None:
                return cached_value

        persistent_cached_value = await asyncio.to_thread(
            load_metadata_cache,
            self.metadata_cache_dir,
            model_info["path"],
        )

        if persistent_cached_value is not None:
            async with self._metadata_lock:
                self._set_cached_metadata_entry(model_name, fingerprint, persistent_cached_value)
            return persistent_cached_value

        try:
            metadata = await asyncio.to_thread(
                self._get_model_metadata_sync, model_info["path"]
            )
        except Exception as exc:
            logger.warning("Metadata scan failed for model %s: %s", model_name, exc)
            return None

        if metadata is None:
            return None

        await asyncio.to_thread(
            save_metadata_cache,
            self.metadata_cache_dir,
            model_name,
            model_info["path"],
            metadata,
        )

        async with self._metadata_lock:
            self._set_cached_metadata_entry(model_name, fingerprint, metadata)

        return metadata

    async def ensure_metadata_cache_for_path(
        self,
        model_path: str | Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Ensure that persistent metadata cache exists for a known model file.

        This is intended for post-download indexing. Cache creation is best-effort:
        failures are logged and returned as None, but they must not make the model
        pull fail.
        """
        file_path = Path(model_path)

        try:
            model_name = self._build_model_ref_from_path(file_path)
            model_info = self._build_model_file_info_from_path(file_path)
        except Exception as exc:
            logger.debug("Cannot resolve metadata cache target for %s: %s", file_path, exc)
            return None

        if model_info is None:
            return None

        fingerprint = model_info["sha256"]

        async with self._metadata_lock:
            cached_value = self._get_cached_metadata_entry(model_name, fingerprint)
            if cached_value is not None:
                return cached_value

        persistent_cached_value = await asyncio.to_thread(
            load_metadata_cache,
            self.metadata_cache_dir,
            model_info["path"],
        )

        if persistent_cached_value is not None:
            async with self._metadata_lock:
                self._set_cached_metadata_entry(model_name, fingerprint, persistent_cached_value)
            return persistent_cached_value

        try:
            metadata = await asyncio.to_thread(
                self._get_model_metadata_sync,
                model_info["path"],
            )
        except Exception as exc:
            logger.warning("Metadata cache creation failed for %s: %s", model_name, exc)
            return None

        if metadata is None:
            return None

        await asyncio.to_thread(
            save_metadata_cache,
            self.metadata_cache_dir,
            model_name,
            model_info["path"],
            metadata,
        )

        async with self._metadata_lock:
            self._set_cached_metadata_entry(model_name, fingerprint, metadata)

        return metadata

    # Where a .toml sits inside each repository (spec doc §5). The .gguf
    # it names may live at any depth; these numbers constrain only the
    # .toml itself, which is what gives a model its name.
    _TOML_DEPTH_BY_REPOSITORY = (
        ("local_models_dir", 1),
        ("tllama_models_dir", 2),
        ("hf_models_dir", 3),
    )

    def _toml_location_for_model_file(self, file_path: Path) -> Optional[Path]:
        """The directory a .toml naming this .gguf has to sit in.

        A HuggingFace repository puts its files wherever its uploader felt
        like -- bartowski files quantisations into subdirectories,
        ggml-org leaves shards in the root -- and that is an organisational
        habit with no protocol meaning. The .toml's own depth is fixed
        regardless, so the name a model gets does not depend on how the
        uploader arranged their repository. That separation is the point of
        the indirection.

        None when the file sits too shallow to be named at all: a .gguf
        directly inside HuggingFace/ has no namespace and repository to
        take a name from.
        """
        for attribute, depth in self._TOML_DEPTH_BY_REPOSITORY:
            repo_dir = getattr(self, attribute)
            if not file_path.is_relative_to(repo_dir):
                continue

            parts = file_path.relative_to(repo_dir).parts
            if len(parts) < depth:
                return None

            return repo_dir.joinpath(*parts[: depth - 1])

        return None

    def _allocate_toml_path(self, directory: Path, stem: str, target_rel: str) -> "tuple[Optional[Path], bool]":
        """Pick the .toml path for a model, avoiding an existing name.

        Returns (path, already_defined). already_defined is True when a
        .toml for this exact file was found: pulling the same model twice
        must not leave a second definition behind, and the path is still
        returned so a caller that means to replace it knows which one.

        Two files in one HuggingFace repository can share a basename when
        the uploader sorted quantisations into subdirectories, and the
        .toml's fixed depth flattens both onto the same name. Rare --
        quantisation is normally in the filename -- but the generator has
        to survive it rather than silently overwrite one with the other, so
        the second one becomes <name>_01.

        That numbering answers a name held by a readable definition of some
        other model. A candidate that will not parse gets an error instead:
        it may well be this model's own definition, broken, and stepping
        around it would answer a mistake in a file with a second file, the
        very duplicate the numbering exists to prevent. The person is told
        which file and why, and fixes it.
        """
        for candidate in self._iter_candidate_toml_paths(directory, stem):
            if not candidate.exists():
                return candidate, False

            try:
                existing = parse_model_toml(
                    candidate.read_text(encoding="utf-8"), source=str(candidate)
                )
            except (TomlModelError, OSError) as exc:
                raise TomlModelError(
                    f"Cannot place a .toml for {target_rel}: {candidate} "
                    f"cannot be read and may be this model's own definition "
                    f"({exc})."
                ) from exc

            if existing.llm_model == target_rel:
                return candidate, True

        raise TomlModelError(
            f"Cannot place a .toml for {target_rel}: {stem} and its numbered "
            f"variants are all taken in {directory}."
        )

    def _iter_candidate_toml_paths(self, directory: Path, stem: str):
        yield directory / f"{stem}.toml"
        for suffix in range(1, 100):
            yield directory / f"{stem}_{suffix:02d}.toml"

    def _ensure_virtual_model_toml_sync(
        self,
        model_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Give a physical model file a .toml, so that it is a model.

        A .gguf nothing points at is not listed and cannot be loaded (spec
        doc §3), which means a pull that only fetches bytes produces
        nothing usable. This is what closes that gap.

        Returns None, without writing anything, when the file is not a
        model in its own right:

        - a projector, which belongs to a model rather than being one
        - a continuation shard, which is part of one file split across
          several and is picked up automatically by llama.cpp from the
          first shard alone
        - a file too shallow in its repository to be named
        - a file that already has a .toml

        Never raises for an ordinary failure to write: a pull that fetched
        several gigabytes successfully must not be reported as failed
        because a small text file could not be created afterwards.
        """
        file_path = Path(model_path).resolve()
        metadata = metadata or {}

        if metadata.get("is_projector"):
            logger.debug("No .toml for %s: it is a projector, not a model", file_path)
            return None

        if metadata.get("is_continuation_shard"):
            logger.debug(
                "No .toml for %s: shard %s of %s, named by its first shard",
                file_path, metadata.get("shard_index"), metadata.get("shard_count"),
            )
            return None

        directory = self._toml_location_for_model_file(file_path)
        if directory is None:
            logger.warning(
                "No .toml for %s: it sits too shallow in its repository to be named",
                file_path,
            )
            return None

        try:
            target_rel = str(file_path.relative_to(self.models_dir.resolve()))
        except ValueError:
            logger.warning("No .toml for %s: outside the model repository", file_path)
            return None

        stem = self._strip_gguf_suffix(file_path.name)

        try:
            toml_path, already_defined = self._allocate_toml_path(directory, stem, target_rel)
            if already_defined:
                return None

            text = render_model_toml(target_rel, metadata=metadata)
            write_model_toml(toml_path, text)
        except Exception as exc:
            logger.warning("Could not write a .toml for %s: %s", file_path, exc)
            return None

        logger.info("Created %s for %s", toml_path, file_path)
        return toml_path

    def _iter_repository_gguf_files(self):
        """Every .gguf on disk, at any depth, in all three repositories.

        Deliberately not _iter_repository_model_files(), which enforces the
        depths the old .gguf-derived naming scheme needed. A .gguf may now
        sit anywhere, because the .toml that names it does not have to sit
        with it -- so for the purpose of finding files that lack a
        definition, every one of them counts.
        """
        for attribute, _ in self._TOML_DEPTH_BY_REPOSITORY:
            repo_dir = getattr(self, attribute)
            if not repo_dir.exists():
                continue

            for file_path in sorted(repo_dir.rglob("*.gguf")):
                if file_path.is_file():
                    yield file_path

    def rebuild_repository(
        self,
        *,
        force: bool = False,
        overwrite_backups: bool = False,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Give every .gguf on disk a .toml, so that every one is a model.

        Explicitly run, never on startup: a repository quietly rewriting
        itself when a service restarts is not something anyone asked for,
        and the decision about what is a model belongs to whoever owns the
        files.

        Not only a one-off migration. Copying a .gguf into Local/ by hand
        leaves it invisible for the same reason a pull without a .toml did,
        so this is the answer to that too, and it is safe to run at any
        time: without force it only adds files that were not there.

        force replaces a definition that already exists. Whatever a person
        wrote into it -- an uncommented n_ctx, a [system].prompt, their own
        comments -- is kept as a .toml.bak alongside it, unconditionally:
        anyone reaching for force is thinking about what they want to gain,
        not about what they are about to lose, and would only think to ask
        for the copy once it was already too late to make one.

        A copy that already exists is not replaced by another one. Running
        force twice used to overwrite the .toml.bak with a copy of the
        generated file, so the second run destroyed the only thing
        protecting against the first -- and the longer a mistake went
        unnoticed, the less was left to recover. Those files are skipped
        instead, one by one rather than as an abort, so everything without
        a backup is still done. overwrite_backups says the copy is
        expendable; on its own it is not a licence to replace anything,
        and does nothing without force.

        Returns one entry per file describing what was done or would be,
        so a caller can report it without repeating the reasoning.
        """
        inventory: List[Any] = []
        results: List[Dict[str, Any]] = []

        for file_path in self._iter_repository_gguf_files():
            try:
                metadata = self._get_model_metadata_sync(str(file_path)) or {}
            except Exception as exc:
                # A header that will not parse is not a reason to stop:
                # the rest of the repository is still worth doing.
                results.append({
                    "file": file_path,
                    "action": "skip",
                    "toml": None,
                    "reason": f"unreadable GGUF header ({exc})",
                })
                continue

            inventory.append((file_path, metadata))

        # Everything is read before anything is written, so a model can be
        # told about a projector sitting beside it. Deriving that from a
        # filename would be cheaper and would also be a guess; this is
        # what the repository actually contains.
        projectors = self._projectors_by_directory(inventory)

        for file_path, metadata in inventory:
            results.append(
                self._rebuild_one(
                    file_path,
                    metadata,
                    projectors=projectors,
                    force=force,
                    overwrite_backups=overwrite_backups,
                    dry_run=dry_run,
                )
            )

        return results

    def _projectors_by_directory(self, inventory) -> Dict[Path, List[Path]]:
        """Where the projectors are, grouped by the directory holding them.

        Deliberately not repository-wide. A projector is converted
        alongside the model it belongs to, so offering one from an
        unrelated repository would be noise a person has to recognise and
        dismiss.
        """
        found: Dict[Path, List[Path]] = {}

        for file_path, metadata in inventory:
            if metadata.get("is_projector"):
                found.setdefault(file_path.parent, []).append(file_path)

        return found

    def _suggested_projector(
        self, file_path: Path, projectors: Dict[Path, List[Path]]
    ) -> Optional[str]:
        """The one projector beside this model, if there is exactly one.

        Two of them is not a suggestion but a question, and a generated
        configuration file is not the place to ask it.
        """
        candidates = projectors.get(file_path.parent) or []
        if len(candidates) != 1:
            return None

        try:
            return str(candidates[0].resolve().relative_to(self.models_dir.resolve()))
        except ValueError:
            return None

    def _rebuild_one(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
        *,
        projectors: Optional[Dict[Path, List[Path]]] = None,
        force: bool,
        overwrite_backups: bool = False,
        dry_run: bool,
    ) -> Dict[str, Any]:
        def outcome(action, toml=None, reason=""):
            return {"file": file_path, "action": action, "toml": toml, "reason": reason}

        if metadata.get("is_projector"):
            return outcome("skip", reason="projector, belongs to a model")

        if metadata.get("is_continuation_shard"):
            return outcome(
                "skip",
                reason=f"shard {metadata.get('shard_index')} of {metadata.get('shard_count')}",
            )

        directory = self._toml_location_for_model_file(file_path)
        if directory is None:
            return outcome("skip", reason="too shallow in its repository to be named")

        try:
            target_rel = str(file_path.resolve().relative_to(self.models_dir.resolve()))
            toml_path, already_defined = self._allocate_toml_path(
                directory, self._strip_gguf_suffix(file_path.name), target_rel
            )
        except (TomlModelError, ValueError) as exc:
            return outcome("skip", reason=str(exc))

        if already_defined and not force:
            return outcome("skip", toml_path, "already defined")

        backup_path = toml_path.with_name(toml_path.name + ".bak")

        # Ahead of the dry run rather than beside the copy itself: a dry
        # run that promises an overwrite the real run refuses is worse
        # than no dry run at all, because it is consulted precisely by
        # someone deciding whether to go ahead.
        if already_defined and backup_path.exists() and not overwrite_backups:
            return outcome(
                "skip",
                toml_path,
                f"{backup_path.name} already exists; it holds what was "
                "replaced last time and would be lost. Move it aside, or "
                "pass --overwrite-backups to say it is expendable",
            )

        action = "overwrite" if already_defined else "create"

        if dry_run:
            return outcome(action, toml_path, "")

        try:
            if already_defined:
                shutil.copy2(toml_path, backup_path)

            text = render_model_toml(
                target_rel,
                metadata=metadata,
                suggested_mmproj=self._suggested_projector(file_path, projectors or {}),
            )
            write_model_toml(toml_path, text, overwrite=already_defined)
        except Exception as exc:
            return outcome("skip", toml_path, f"could not write ({exc})")

        return outcome(action, toml_path, "")

    async def ensure_virtual_model_toml(
        self,
        model_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        return await asyncio.to_thread(
            self._ensure_virtual_model_toml_sync, model_path, metadata
        )

    def _list_local_models_sync(self) -> List[Dict[str, Any]]:
        """
        Scan all known repositories for virtual model .toml files.

        A .gguf/mmproj with no .toml naming it is not a model as far as
        this listing is concerned (spec doc §3: .toml is the only source of
        truth for what shows up here). One broken .toml must not break the
        whole listing.
        """
        model_list: List[Dict[str, Any]] = []

        for toml_path in self._iter_repository_toml_files():
            try:
                text = toml_path.read_text(encoding="utf-8")
                spec = parse_model_toml(text, source=str(toml_path))

                target_path = self._resolve_virtual_model_target_path(spec, toml_path)
                if target_path is None:
                    continue

                model_info = self._build_model_file_info_from_path(target_path)
                if model_info is None:
                    continue

                model_info["id"] = self._build_virtual_model_ref_from_toml_path(toml_path)

                if toml_path.is_relative_to(self.hf_models_dir):
                    model_info["repository"] = "HuggingFace"
                elif toml_path.is_relative_to(self.local_models_dir):
                    model_info["repository"] = "Local"
                elif toml_path.is_relative_to(self.tllama_models_dir):
                    model_info["repository"] = "TLlama"
                else:
                    continue

                model_list.append({
                    "id": model_info["id"],
                    "filename": model_info["filename"],
                    "path": model_info["path"],
                    "size": model_info["size"],
                    "mtime": model_info["mtime"],
                    "sha256": model_info["sha256"],
                    "repository": model_info["repository"],
                })
            except TomlModelError as e:
                logger.warning("Ignoring %s: %s", toml_path, e)
            except Exception as e:
                logger.warning("Failed to inspect virtual model file %s: %s", toml_path, e)

        return model_list

    async def list_local_models(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._list_local_models_sync)

    async def list_local_models_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Return local models enriched with GGUF metadata and content digest.

        The single place that needs both together: previously /api/tags
        fetched metadata and digest itself in an inline loop, duplicating
        this. A failure scanning one model's metadata or hashing one
        model's content is isolated to that model, not the whole listing.

        That isolation has to be enforced here as well as in the scan.
        _list_local_models_sync parses each .toml and drops the ones it
        cannot use, but get_model_metadata() re-reads the same file from
        disk afterwards, so a file edited in between is parsed twice with
        different results: the scan let the model through and the second
        read raised, taking the whole listing with it. A .toml is an
        ordinary file a person edits in place, and /api/tags is exactly
        what a client polls while that is happening.

        The model is dropped, matching what the scan does with a file it
        cannot parse.
        """
        models = await self.list_local_models()
        enriched: List[Dict[str, Any]] = []

        for model in models:
            item = dict(model)

            try:
                metadata = await self.get_model_metadata(model["id"])
            except TomlModelError as e:
                logger.warning("Ignoring %s: %s", model["id"], e)
                continue

            if metadata:
                item.update(metadata)

            digest_info = await self.get_model_digest(model["path"]) or {}
            item["digest"] = digest_info.get("content_sha256", "")

            enriched.append(item)

        return enriched


    def _validate_global_kv_cache_types(self) -> None:
        from tllama.native_params import kv_type_name

        for variable, value in (
            ("TLLAMA_KV_CACHE_TYPE", self.config.kv_cache_type),
            ("TLLAMA_K_CACHE_TYPE", self.config.k_cache_type),
            ("TLLAMA_V_CACHE_TYPE", self.config.v_cache_type),
        ):
            if value is not None:
                try:
                    kv_type_name(value)
                except ValueError as exc:
                    raise ConfigError(f"Unsupported {variable} value: {value}. {exc}") from exc

    async def start(self):
        if self._shutdown_task is not None:
            await _settle_owned(self._shutdown_task)
            self._shutdown_task = None
        if self._native.closed:
            self._native = self._new_native_residency()
        self._stopping = False
        self.ensure_storage()
        self._ensure_janitor_running()

    async def shutdown(self):
        self._stopping = True
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown_impl())
        await _settle_owned(self._shutdown_task)

    async def _shutdown_impl(self):
        janitor_task, self._janitor_task = self._janitor_task, None
        try:
            if janitor_task is not None:
                janitor_task.cancel()
                try:
                    await janitor_task
                except asyncio.CancelledError:
                    pass
        finally:
            await self._native.shutdown()

    async def _janitor_loop(self):
        while True:
            await asyncio.sleep(self.config.janitor_interval_seconds)
            await self._native.expire()

    def _split_model_reference(self, model_ref: str) -> List[str]:
        cleaned = (model_ref or "").strip().strip("/")
        parts = [part.strip() for part in cleaned.split("/") if part.strip()]

        if not parts:
            raise ValueError("Empty model reference")

        if any(part in {".", ".."} for part in parts):
            raise ValueError("Invalid model reference")

        return parts

    def resolve_hf_pull_target(self, model_ref: str) -> Dict[str, Any]:
        parts = self._split_model_reference(model_ref)

        if len(parts) < 3:
            raise ValueError(
                "Expected HuggingFace pull reference in format 'namespace/repo/filename' "
                "or 'namespace/repo/path/to/file[.gguf]'"
            )

        namespace = parts[0]
        repo = parts[1]
        filename_parts = parts[2:]

        filename_parts[-1] = self._normalize_pull_filename(filename_parts[-1])
        filename = "/".join(filename_parts)

        target_path = self.hf_models_dir.joinpath(namespace, repo, *filename_parts)

        return {
            "model_ref": "/".join(parts),
            "namespace": namespace,
            "repo": repo,
            "repo_id": f"{namespace}/{repo}",
            "filename": filename,
            "target_dir": target_path.parent,
            "target_path": target_path,
        }

    def _pull_hf_file_sync(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        revision: str | None = None,
        progress: "PullProgress | None" = None,
    ) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it to enable HuggingFace pulls."
            ) from e

        namespace, repo = repo_id.split("/", 1)
        target_root = self.hf_models_dir / namespace / repo
        target_root.mkdir(parents=True, exist_ok=True)

        tqdm_class = _hf_tqdm_class_for(progress) if progress is not None else None

        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                local_dir=target_root,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm_class,
            )
        except ValueError as e:
            # huggingface_hub decides to use the Xet backend by checking
            # whether the hf_xet *package metadata* is present, then only
            # tries the actual `import hf_xet` once it has already committed
            # to that path. A package whose compiled extension does not
            # match the running interpreter (seen in practice: a .so built
            # for the free-threaded 3.13t ABI installed under plain 3.13)
            # is "present" but not importable, and surfaces here as a
            # ValueError with no indication that a plain HTTP download
            # would have worked fine.
            if "Xet storage" in str(e):
                raise RuntimeError(
                    f"{e} This repository uses Xet storage and the installed "
                    "hf_xet package cannot actually be imported (check "
                    "`python3 -c \"import hf_xet\"` in this server's "
                    "environment). Set HF_HUB_DISABLE_XET=1 to fall back to "
                    "a regular HTTP download until hf_xet is fixed."
                ) from e
            raise

    async def pull_hf_file(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        revision: str | None = None,
        progress: "PullProgress | None" = None,
    ) -> str:
        return await asyncio.to_thread(
            self._pull_hf_file_sync,
            repo_id,
            filename,
            token,
            revision,
            progress,
        )

    def _fetch_hf_file_info_sync(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        revision: str | None = None,
    ) -> "HfFileLookup":
        """
        Look up the published sha256 and size of a file on HuggingFace.

        The LFS oid is the sha256 of the file contents and is still published
        for Xet-backed repositories, where the Xet hash is an additional field
        rather than a replacement. This lets a pull record the digest without
        reading the downloaded file back.

        Only an empty answer from a repository that did reply is reported as
        MISSING. Anything else, including every error, is UNKNOWN, so a
        network or credential problem can never be mistaken for a bad model
        reference.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return HfFileLookup(HF_LOOKUP_UNKNOWN)

        try:
            entries = HfApi().get_paths_info(
                repo_id,
                [filename],
                revision=revision,
                token=token,
            )
        except Exception as exc:
            logger.debug("HuggingFace file info lookup failed for %s/%s: %s", repo_id, filename, exc)
            return HfFileLookup(HF_LOOKUP_UNKNOWN)

        if not entries:
            return HfFileLookup(HF_LOOKUP_MISSING)

        for entry in entries:
            lfs = getattr(entry, "lfs", None)
            if lfs is None:
                continue

            try:
                return HfFileLookup(HF_LOOKUP_FOUND, str(lfs.sha256), int(lfs.size))
            except (AttributeError, TypeError, ValueError):
                continue

        # The path exists but carries no usable LFS metadata, for instance a
        # directory or a small non-LFS file. Not a reason to refuse the pull.
        return HfFileLookup(HF_LOOKUP_UNKNOWN)

    async def fetch_hf_file_info(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        revision: str | None = None,
    ) -> "HfFileLookup":
        return await asyncio.to_thread(
            self._fetch_hf_file_info_sync,
            repo_id,
            filename,
            token,
            revision,
        )

    async def store_hf_digest(
        self,
        model_path: str | Path,
        hf_file_info: Optional["HfFileLookup"],
    ) -> Optional[Dict[str, Any]]:
        """
        Record a digest taken from HuggingFace for a freshly pulled file.

        HuggingFace publishes the sha256, but hf_hub_download only verifies a
        download against the expected size, so the value is a claim about the
        bytes rather than a measurement of them. The size is checked before
        the claim is accepted; on mismatch nothing is stored and the digest is
        computed locally on next use.

        The stored source is "hf" precisely so that a later verification can
        tell a claimed digest from a measured one.
        """
        if hf_file_info is None or hf_file_info.status != HF_LOOKUP_FOUND:
            return None

        sha256_hex = hf_file_info.sha256
        expected_size = hf_file_info.size

        if not sha256_hex or expected_size is None:
            return None

        file_path = Path(model_path)

        try:
            actual_size = file_path.stat().st_size
        except OSError as exc:
            logger.warning("Cannot stat %s for digest recording: %s", file_path, exc)
            return None

        if actual_size != expected_size:
            logger.warning(
                "Size mismatch for %s (published %s, on disk %s); "
                "digest will be computed locally",
                file_path, expected_size, actual_size,
            )
            return None

        digest = {"content_sha256": sha256_hex, "source": "hf"}

        try:
            model_name = self._build_model_ref_from_path(file_path)
        except ValueError:
            model_name = str(file_path)

        await asyncio.to_thread(
            save_digest_cache,
            self.metadata_cache_dir,
            model_name,
            file_path,
            digest,
        )

        return digest

    def _normalize_pull_filename(self, filename: str) -> str:
        cleaned = (filename or "").strip()
        if not cleaned:
            raise ValueError("Missing filename in model reference")

        if cleaned.lower().endswith(".gguf"):
            return cleaned

        return f"{cleaned}.gguf"

    def resolve_model_storage_path(self, model_ref: str) -> Path:
        """
        Resolve a model reference to its on-disk path inside known repositories.

        The repository is chosen by how many segments the reference has:

        - one:            Local/<name>.gguf
        - two:            TLlama/<namespace>/<name>.gguf
        - three or more:  HuggingFace/<namespace>/<repo>/<path>.gguf

        A HuggingFace reference may nest further, because a repository there
        can keep a quantisation in a subdirectory.
        """
        parts = self._split_model_reference(model_ref)

        def _normalized_file_path(base_dir: Path, rel_parts: List[str]) -> Path:
            normalized_parts = list(rel_parts)
            normalized_parts[-1] = self._normalize_pull_filename(normalized_parts[-1])
            return base_dir.joinpath(*normalized_parts)

        def _first_existing_or_default(candidates: List[Path]) -> Path:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        # Prefixless Local reference.
        # Example: "MyModel-Instruct-Q4_K_L"
        if len(parts) == 1:
            return _normalized_file_path(self.local_models_dir, parts)

        # Prefixless TLlama reference.
        # Example: "collection/model-file"
        #
        # If a matching Local nested file exists and the TLlama file does not, keep it
        # usable as a fallback. Explicit "Local/..." remains the unambiguous form.
        if len(parts) == 2:
            return _first_existing_or_default([
                _normalized_file_path(self.tllama_models_dir, parts),
                _normalized_file_path(self.local_models_dir, parts),
            ])

        if len(parts) >= 3:
            normalized_parts = list(parts)
            normalized_parts[-1] = self._normalize_pull_filename(normalized_parts[-1])
            return self.hf_models_dir.joinpath(*normalized_parts)

        raise ValueError(
            "Unsupported model reference. Expected one of: "
            "'namespace/repo/file', 'namespace/repo/path/to/file[.gguf]', "
            "'Local/name', 'Local/path/to/file', "
            "'TLlama/name', or 'TLlama/path/to/file'."
        )

    def _remove_empty_parents(self, start_dir: Path, stop_dir: Path) -> None:
        current = start_dir

        while True:
            try:
                if current == stop_dir or stop_dir not in current.parents:
                    break

                current.rmdir()
            except OSError:
                # Directory is not empty or cannot be removed; stop quietly.
                break

            current = current.parent

    def delete_model_definition(self, model_ref: str) -> Dict[str, Any]:
        """Delete a model by deleting its definition, and only that.

        The .toml is the manifest and the .gguf is the blob, and real
        Ollama's `rm` works the same way for the same reason (verified,
        not assumed): it removes the manifest and leaves the blob for as
        long as anything else refers to it. Here more than one virtual
        model can name one physical file deliberately, since sharing
        weights instead of copying them is a stated purpose of the
        indirection, so deleting the weights along with one name would
        break the others silently.

        The weights are never removed automatically; reclaiming disk space
        is separate work with its own decisions to make about orphans.

        The two on-disk caches stay too. They are keyed by the .gguf,
        which has not changed, and the digest cache in particular cost a
        full read of a multi-gigabyte file to build -- discarding it
        because a name went away would mean paying that again for a file
        that never moved. Only the in-memory entry keyed by the model's
        name is dropped, because that name no longer refers to anything.
        """
        toml_path = self._toml_path_for_reference(model_ref)

        if toml_path is None:
            raise ValueError(f"Not a model name this repository can hold: {model_ref}")

        if not toml_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_ref}")

        kept_file: Optional[str] = None
        try:
            spec = parse_model_toml(
                toml_path.read_text(encoding="utf-8"), source=str(toml_path)
            )
            kept_file = spec.llm_model
        except (TomlModelError, OSError):
            # Deleting a definition that will not parse is exactly when
            # someone most wants to be able to delete it.
            pass

        toml_path.unlink(missing_ok=True)
        self._invalidate_metadata_cache_entry(model_ref)

        repo_root = self._get_repo_root_for_path(toml_path)
        self._remove_empty_parents(toml_path.parent, repo_root)

        return {
            "model_ref": model_ref,
            "deleted_path": str(toml_path),
            "kept_model_file": kept_file,
        }

    async def delete_model(self, model_ref: str) -> Dict[str, Any]:
        async def retire_and_delete():
            await self._native.unload(model_ref)
            try:
                return await asyncio.to_thread(self.delete_model_definition, model_ref)
            finally:
                # Cover acquisitions overlapping definition deletion as well.
                await self._native.unload(model_ref)
        # Do not let caller cancellation leave a late file deletion with a
        # freshly selectable native generation after its first retirement.
        return await _settle_owned(asyncio.create_task(retire_and_delete()))

    def _strip_gguf_suffix(self, value: str) -> str:
        return value[:-5] if value.lower().endswith(".gguf") else value

    def _build_relative_ref_without_suffix(self, base_dir: Path, file_path: Path) -> str:
        rel = file_path.relative_to(base_dir)
        parts = list(rel.parts)
        parts[-1] = self._strip_gguf_suffix(parts[-1])
        return "/".join(parts)

    def _build_model_ref_from_path(self, file_path: Path) -> str:
        if file_path.is_relative_to(self.hf_models_dir):
            return self._build_relative_ref_without_suffix(self.hf_models_dir, file_path)

        if file_path.is_relative_to(self.local_models_dir):
            return self._build_relative_ref_without_suffix(self.local_models_dir, file_path)

        if file_path.is_relative_to(self.tllama_models_dir):
            return self._build_relative_ref_without_suffix(self.tllama_models_dir, file_path)

        raise ValueError(f"File path is outside known model repositories: {file_path}")

    def _iter_repository_toml_files(self):
        """Yield .toml files that name a virtual model.

        Fixed depth per category (spec doc TLlama_virtual_models_spec.md
        §5): exactly 1 for Local, exactly 2 for TLlama, exactly 3 for
        HuggingFace. Unlike a raw .gguf's depth (see
        _iter_repository_model_files), a .toml's own depth has nothing to
        do with how deep the .gguf/mmproj it points at physically sits --
        decoupling the two is the whole point of the indirection.
        """
        repositories = (
            (self.local_models_dir, 1),
            (self.tllama_models_dir, 2),
            (self.hf_models_dir, 3),
        )

        for repo_dir, depth in repositories:
            if not repo_dir.exists():
                continue

            pattern = "/".join(["*"] * (depth - 1) + ["*.toml"])
            for file_path in sorted(repo_dir.glob(pattern)):
                if file_path.is_file():
                    yield file_path

    def _build_virtual_model_ref_from_toml_path(self, toml_path: Path) -> str:
        if toml_path.is_relative_to(self.hf_models_dir):
            base_dir = self.hf_models_dir
        elif toml_path.is_relative_to(self.local_models_dir):
            base_dir = self.local_models_dir
        elif toml_path.is_relative_to(self.tllama_models_dir):
            base_dir = self.tllama_models_dir
        else:
            raise ValueError(f"File path is outside known model repositories: {toml_path}")

        rel = toml_path.relative_to(base_dir)
        parts = list(rel.parts)
        name = parts[-1]
        if name.lower().endswith(".toml"):
            name = name[:-5]
        parts[-1] = name
        return "/".join(parts)

    def _read_metadata_for_path(self, file_path: Path) -> Dict[str, Any]:
        """Metadata for a file, from the on-disk cache when it is there.

        Used on paths that run per listing, so re-reading a GGUF header
        every time is worth avoiding; the cache is keyed by the file and
        invalidates on its own when the file changes.
        """
        cached = load_metadata_cache(self.metadata_cache_dir, file_path)
        if cached is not None:
            return cached

        metadata = self._get_model_metadata_sync(str(file_path)) or {}

        if metadata:
            save_metadata_cache(
                self.metadata_cache_dir,
                self._strip_gguf_suffix(file_path.name),
                file_path,
                metadata,
            )

        return metadata

    def resolve_mmproj_path(self, spec: VirtualModelSpec, source: str) -> Optional[Path]:
        """The projector a virtual model names, checked to actually be one.

        Checked rather than trusted because the failure it prevents is
        silent and late: a path that points at an ordinary model, or at
        nothing, produces a model that lists and loads perfectly and then
        cannot see. Saying so while someone still has the file they just
        edited open is worth the header read.

        None when there is no [mmproj] section, or when it holds a 'from'
        that has not been imported yet -- the same transient state [llm]
        allows.
        """
        if spec.mmproj_model is None:
            return None

        path = resolve_repo_relative_path(spec.mmproj_model, self.models_dir)

        if not path.is_file():
            raise TomlModelError(
                f"{source}: [mmproj] model = {spec.mmproj_model!r} does not "
                "point at an existing file."
            )

        metadata = self._read_metadata_for_path(path)

        if metadata and not metadata.get("is_projector"):
            raise TomlModelError(
                f"{source}: [mmproj] model = {spec.mmproj_model!r} is not a "
                f"projector (its architecture is {metadata.get('arch', 'unknown')!r}; "
                "a projector reports 'clip')."
            )

        return path

    def _resolve_virtual_model_target_path(
        self, spec: VirtualModelSpec, toml_path: Path
    ) -> Optional[Path]:
        """Resolve a virtual model's [llm] section to a physical file.

        Returns None (never raises) when the .toml isn't ready to be listed
        yet -- 'from =' with no matching import performed, a 'model =' that
        escapes the repo, or one that simply doesn't exist -- since a single
        bad .toml must not break the rest of the listing (matches
        _list_local_models_sync's existing per-file isolation).
        """
        if spec.llm_model is None:
            # 'from =' is only ever meant to be transient (spec §5): it
            # gets rewritten to 'model =' once an import runs. Until that
            # happens there is no physical file inside the repo yet.
            logger.warning(
                "Ignoring %s: 'from' has not been imported into the repo yet "
                "(no 'model =' to load).",
                toml_path,
            )
            return None

        try:
            target = resolve_repo_relative_path(spec.llm_model, self.models_dir)
        except TomlModelError as exc:
            logger.warning("Ignoring %s: %s", toml_path, exc)
            return None

        if not target.is_file():
            logger.warning(
                "Ignoring %s: model = %r does not point at an existing file",
                toml_path, spec.llm_model,
            )
            return None

        try:
            self.resolve_mmproj_path(spec, str(toml_path))
        except TomlModelError as exc:
            # Same treatment as a broken [llm]: a definition that names a
            # projector it cannot use is not usable, and hiding that until
            # someone sends an image would be worse than saying so now.
            logger.warning("Ignoring %s: %s", toml_path, exc)
            return None

        return target

    def _iter_repository_model_files(self):
        """Yield the model files that the reference scheme can name.

        A reference has one segment for Local, two for TLlama and three or
        more for HuggingFace, so a file is only usable at the matching depth
        inside its repository. The scan used to accept any depth, which meant
        a file placed a level too deep was listed by /api/tags and then failed
        to load, because the name it was given resolved somewhere else.
        """
        repositories = (
            (self.local_models_dir, 1, 1, "directly in the repository"),
            (self.tllama_models_dir, 2, 2, "one directory deep"),
            (self.hf_models_dir, 3, None, "at least two directories deep"),
        )

        for repo_dir, min_depth, max_depth, expectation in repositories:
            if not repo_dir.exists():
                continue

            for file_path in sorted(repo_dir.rglob("*.gguf")):
                if not file_path.is_file():
                    continue

                depth = len(file_path.relative_to(repo_dir).parts)

                if depth < min_depth or (max_depth is not None and depth > max_depth):
                    key = str(file_path)
                    if key not in self._reported_unusable_files:
                        self._reported_unusable_files.add(key)
                        logger.warning(
                            "Ignoring %s: a model in %s has to sit %s",
                            file_path, repo_dir.name, expectation,
                        )
                    continue

                yield file_path

    def _get_repo_root_for_path(self, file_path: Path) -> Path:
        if file_path.is_relative_to(self.hf_models_dir):
            return self.hf_models_dir
        if file_path.is_relative_to(self.local_models_dir):
            return self.local_models_dir
        if file_path.is_relative_to(self.tllama_models_dir):
            return self.tllama_models_dir
        raise ValueError(f"File path is outside known model repositories: {file_path}")

    def _get_hf_repo_dir_for_file(self, file_path: Path) -> Path | None:
        if not file_path.is_relative_to(self.hf_models_dir):
            return None

        rel = file_path.relative_to(self.hf_models_dir)
        if len(rel.parts) < 2:
            return None

        return self.hf_models_dir / rel.parts[0] / rel.parts[1]

    def _cleanup_hf_repo_auxiliary(self, file_path: Path) -> None:
        """
        Best-effort cleanup for HuggingFace repo-local helper data.

        If no GGUF files remain in the HuggingFace repo subtree, remove the
        repo-local .cache directory so normal empty-parent cleanup can finish.
        """
        repo_dir = self._get_hf_repo_dir_for_file(file_path)
        if repo_dir is None or not repo_dir.exists():
            return

        has_remaining_models = any(repo_dir.rglob("*.gguf"))
        if has_remaining_models:
            return

        cache_dir = repo_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


model_manager = ModelManager(load_backend_config_from_env())

