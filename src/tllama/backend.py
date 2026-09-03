import os
import time
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
import gc
import shutil

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, MTMDChatHandler
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone, timedelta

from tllama.config import BackendConfig, ConfigError, load_backend_config_from_env
from tllama.helpers.common import normalize_keep_alive
from tllama.helpers.llama_stats import load_llama_with_captured_stats
from tllama.helpers.gguf_metadata import read_gguf_metadata, build_model_metadata_payload
from tllama.helpers.model_toml import (
    TomlModelError,
    VirtualModelSpec,
    parse_model_toml,
    render_model_toml,
    resolve_kv_cache_types,
    resolve_repo_relative_path,
    write_model_toml,
)
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides
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


class _TemplateOverridingMTMDChatHandler(MTMDChatHandler):
    """A projector handler that renders a [template] instead of the GGUF's.

    MTMDChatHandler does its own templating: it reads the model's
    tokenizer.chat_template, renders it itself, and only then hands the
    result to mtmd for splitting on the media marker. That leaves nothing
    for _apply_template_override_to_chat_handler to replace afterwards, so
    without this a [template] next to an [mmproj] would be silently
    ignored -- against the rule that a .toml outranks what the GGUF says.

    _get_chat_template is the single seam where the handler decides what to
    render, so overriding it is the whole change. Everything downstream --
    the media marker substitution, the tokenization, the eval loop -- is
    untouched and keeps working on the rendered text either way.
    """

    def __init__(self, *args, template: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._template_override = template

    def _get_chat_template(self, llama_model) -> str:
        return self._template_override


def model_has_projector(llm) -> bool:
    """Whether a loaded model can actually take an image.

    Asked of the loaded model rather than of its definition on purpose:
    what decides is whether a projector handler is attached and holding an
    mtmd context, which is the thing an image is eventually handed to. A
    .toml naming an [mmproj] says what was asked for; this says what
    happened.
    """
    return isinstance(getattr(llm, "chat_handler", None), MTMDChatHandler)


def _initialise_projector(llm) -> None:
    """Bring a projector into memory now rather than at the first image.

    MTMDChatHandler builds its mtmd context lazily, on the first call that
    carries one. That defers the projector's weights and its compute
    buffers -- hundreds of MiB between them -- past the point where anyone
    is looking, with two consequences. The allocation lands mid-request
    instead of at load, and, because it happens outside the window
    load_llama_with_captured_stats reads, /api/ps reports a size for the
    model that leaves the projector out entirely. Somebody checking
    whether a vision model fits in VRAM is then reading a number that
    cannot answer the question.

    Ollama preloads and preallocates for exactly this reason: after a load
    finishes, what it says about a model is the whole truth about it.

    Not an error path. A projector that does not fit in VRAM spills into
    system RAM and keeps working, only slower -- and being able to see
    that is the point. Only a projector that fits nowhere at all fails,
    and that failure belongs to the load, where it is raised.
    """
    handler = getattr(llm, "chat_handler", None)
    initialise = getattr(handler, "_init_mtmd_context", None)
    if initialise is None:
        return

    initialise(llm)


def _apply_template_override_to_chat_handler(llm, virtual_spec: Optional[VirtualModelSpec]) -> None:
    """Make a virtual model's [template] reach /api/chat and /v1/chat/completions too.

    metadata_info["template"] (set from this same virtual_spec.template by
    get_model_metadata) only ever reaches /api/generate's own
    render_generate_prompt(). /api/chat and /v1/chat/completions call
    create_chat_completion_ex()/create_chat_completion(), which resolve a
    handler and let it render the prompt internally -- by default from a
    Jinja2ChatFormatter llama-cpp-python already built and cached during
    Llama.__init__, straight from the GGUF's own baked-in
    tokenizer.chat_template, with no way for anything set afterward to reach
    it. llm.chat_handler, once set, takes absolute priority over that cached
    default (see lib/llama_wrap.py's _resolve_chat_completion_handler), so
    replacing it here -- built the same way Llama.__init__ builds its own
    default handlers -- is the one place that actually reaches every
    endpoint.

    Left untouched when llm.chat_handler is already set, which now means a
    projector handler built by _build_vision_chat_handler and passed to
    Llama(). Replacing it here would throw the projector away and leave the
    model unable to see. That handler carries the [template] itself (see
    _TemplateOverridingMTMDChatHandler), so nothing is lost by stopping.
    """
    if virtual_spec is None or virtual_spec.template is None:
        return
    if llm.chat_handler is not None:
        return

    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    bos_token = llm.detokenize([bos_id]).decode("utf-8", errors="ignore") if bos_id != -1 else ""
    eos_token = llm.detokenize([eos_id]).decode("utf-8", errors="ignore") if eos_id != -1 else ""

    llm.chat_handler = Jinja2ChatFormatter(
        template=virtual_spec.template,
        eos_token=eos_token,
        bos_token=bos_token,
        stop_token_ids=[eos_id] if eos_id != -1 else None,
    ).to_chat_handler()


class ModelManager:
    def __init__(self, config: BackendConfig | None = None):
        self.config = config or load_backend_config_from_env()

        self.models: Dict[str, Llama] = {}

        # Three locks, one per piece of state, because a single one made a
        # model load block everything: a listing, a metadata read, even a
        # request for a different model that was already resident.
        #
        # Two rules keep this safe. No path holds more than one of them at a
        # time, and none of them is held across a long await. The model load
        # in get_model is the remaining exception and is dealt with next.
        self._models_lock = asyncio.Lock()      # models, active_models, janitor
        self._metadata_lock = asyncio.Lock()    # the in-memory metadata cache
        self._slots_lock = asyncio.Lock()       # the generation slot registry

        self.models_dir = Path(self.config.models_dir)
        self.metadata_cache_dir = self.models_dir / ".tllama" / "metadata-cache"

        self.active_models: Dict[str, Dict[str, Any]] = {}

        self._janitor_task: asyncio.Task | None = None

        self._metadata_cache: Dict[str, CachedMetadataEntry] = {}

        # Paths already reported as sitting outside the reference scheme, so a
        # stray file does not produce a warning on every model listing.
        self._reported_unusable_files: set[str] = set()

        # One semaphore per model name, guarding generation on that model.
        self._inference_slots: Dict[str, asyncio.Semaphore] = {}

        # Loads under way, by model name. A model is here between the moment
        # the decision to load it is taken and the moment it becomes
        # available, so that concurrent callers can wait rather than start a
        # second load, and so that capacity accounts for it.
        self._loading: Dict[str, asyncio.Future] = {}

        # Resolved once, here, so a bad TLLAMA_KV_CACHE_TYPE stops the
        # server at startup. It used to be resolved on every load, which
        # meant an unknown name surfaced as a 400 on the first request to
        # need a model -- possibly hours later, and reported as a failure
        # to load that model rather than as the misconfiguration it was.
        self._type_k, self._type_v = self._resolve_global_kv_cache_types()

        # Converted here rather than in config.py, which cannot see
        # Llama()'s signature, and here rather than at load time so an
        # unknown parameter or an unusable value stops the server instead
        # of surfacing on the first request that needs a model.
        self._runtime_overrides = coerce_runtime_overrides(
            self.config.runtime_overrides
        )

        # Same reasoning for the sampling layer, read off the completion
        # signatures instead of Llama()'s. Public because the routers
        # hand it to build_sampling_kwargs() on every request: a global
        # value is server configuration, not model metadata, so it does
        # not belong in the metadata dict the other layers travel in.
        self.sampling_overrides = coerce_sampling_overrides(
            self.config.sampling_overrides
        )

        self.hf_models_dir = self.models_dir / "HuggingFace"
        self.local_models_dir = self.models_dir / "Local"
        self.tllama_models_dir = self.models_dir / "TLlama"

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

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _future_iso(self, seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def _parse_iso_datetime(self, value: str | None) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    def _is_model_entry_expired(self, model_info: Dict[str, Any]) -> bool:
        expires_at = self._parse_iso_datetime(model_info.get("expires_at"))
        if expires_at is None:
            return False
        return expires_at <= datetime.now(timezone.utc)

    def _unload_model_internal(self, model_name: str) -> bool:
        llm = self.models.pop(model_name, None)

        removed = False
        if model_name in self.active_models:
            del self.active_models[model_name]
            removed = True

        # Deliberately not calling llm.close() here. The weights are freed
        # when the last reference to the Llama goes, and a request still
        # generating on this model holds one: an unload that is asked for
        # mid-generation has to wait for it. Freeing eagerly destroys the
        # llama_context under the running request and segfaults the server,
        # which is what happened when this did call close().
        #
        # ollama stop is a /api/generate with keep_alive 0, and the unload it
        # triggers runs outside the generation slot, so this is not a corner
        # case.
        gc.collect()

        return (llm is not None) or removed

    def unload_all_models(self):
        for model_name in list(self.models.keys()):
            self._unload_model_internal(model_name)
        self.active_models.clear()
        gc.collect()

    def _normalize_num_ctx(self, value, default: int = 0) -> int:
        if value is None:
            return default
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

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

    def _to_float_mib(self, value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _mib_to_bytes(self, value_mib: float) -> int:
        return int(round(value_mib * 1024 * 1024))

    def _build_memory_accounting(self, load_stats: Dict[str, Any]) -> Dict[str, Any]:
        gpu_model_mib = self._to_float_mib(load_stats.get("gpu_model_mib"))
        gpu_kv_mib = self._to_float_mib(load_stats.get("gpu_kv_mib"))
        gpu_compute_mib = self._to_float_mib(load_stats.get("gpu_compute_mib"))
        gpu_output_mib = self._to_float_mib(load_stats.get("gpu_output_mib"))
        gpu_rs_mib = self._to_float_mib(load_stats.get("gpu_rs_mib"))
        gpu_projector_mib = self._to_float_mib(load_stats.get("gpu_projector_mib"))

        gpu_host_model_mib = self._to_float_mib(load_stats.get("gpu_host_model_mib"))
        gpu_host_kv_mib = self._to_float_mib(load_stats.get("gpu_host_kv_mib"))
        gpu_host_compute_mib = self._to_float_mib(load_stats.get("gpu_host_compute_mib"))
        gpu_host_output_mib = self._to_float_mib(load_stats.get("gpu_host_output_mib"))
        gpu_host_rs_mib = self._to_float_mib(load_stats.get("gpu_host_rs_mib"))

        cpu_model_mib = self._to_float_mib(load_stats.get("cpu_model_mib"))
        cpu_kv_mib = self._to_float_mib(load_stats.get("cpu_kv_mib"))
        cpu_compute_mib = self._to_float_mib(load_stats.get("cpu_compute_mib"))
        cpu_output_mib = self._to_float_mib(load_stats.get("cpu_output_mib"))
        cpu_rs_mib = self._to_float_mib(load_stats.get("cpu_rs_mib"))
        cpu_projector_mib = self._to_float_mib(load_stats.get("cpu_projector_mib"))

        # True residency buckets for Ollama-like processor split.
        # A projector's weights are resident the same way the model's are:
        # loaded once, held for the lifetime of the model, and sitting on
        # whichever device clip_ctx named.
        gpu_loaded_mib = gpu_model_mib + gpu_kv_mib + gpu_projector_mib
        cpu_loaded_mib = cpu_model_mib + cpu_kv_mib + cpu_projector_mib
        loaded_total_mib = gpu_loaded_mib + cpu_loaded_mib

        # Ollama-like ps size:
        # - count real GPU-loaded model+KV+projector
        # - count true CPU-loaded model+KV+projector
        # - include small GPU helper buffers
        # - intentionally DO NOT include gpu_host_compute_mib, because that is
        #   host-side staging / pinned-memory fallback and it pollutes ps output
        ps_size_mib = (
            gpu_loaded_mib +
            cpu_loaded_mib +
            gpu_compute_mib +
            gpu_rs_mib +
            gpu_host_model_mib +
            gpu_host_output_mib
        )

        # For Ollama-like PROCESSOR split, only true CPU-loaded model/KV memory
        # should count as CPU. GPU host/helper buffers are still GPU-associated.
        ps_size_vram_mib = max(ps_size_mib - cpu_loaded_mib, 0.0)

        # Full debug/runtime footprint
        gpu_total_runtime_mib = (
            gpu_model_mib +
            gpu_kv_mib +
            gpu_compute_mib +
            gpu_output_mib +
            gpu_rs_mib +
            gpu_projector_mib +
            gpu_host_model_mib +
            gpu_host_kv_mib +
            gpu_host_compute_mib +
            gpu_host_output_mib +
            gpu_host_rs_mib
        )

        cpu_total_runtime_mib = (
            cpu_model_mib +
            cpu_kv_mib +
            cpu_compute_mib +
            cpu_output_mib +
            cpu_rs_mib +
            cpu_projector_mib
        )

        total_runtime_mib = gpu_total_runtime_mib + cpu_total_runtime_mib

        return {
            "gpu_loaded_mib": gpu_loaded_mib,
            "cpu_loaded_mib": cpu_loaded_mib,
            "loaded_total_mib": loaded_total_mib,

            "gpu_loaded_bytes": self._mib_to_bytes(gpu_loaded_mib),
            "cpu_loaded_bytes": self._mib_to_bytes(cpu_loaded_mib),
            "loaded_total_bytes": self._mib_to_bytes(loaded_total_mib),

            "ps_size_vram_mib": ps_size_vram_mib,
            "ps_size_mib": ps_size_mib,
            "ps_size_vram_bytes": self._mib_to_bytes(ps_size_vram_mib),
            "ps_size_bytes": self._mib_to_bytes(ps_size_mib),

            "gpu_total_runtime_mib": gpu_total_runtime_mib,
            "cpu_total_runtime_mib": cpu_total_runtime_mib,
            "total_runtime_mib": total_runtime_mib,

            "gpu_total_runtime_bytes": self._mib_to_bytes(gpu_total_runtime_mib),
            "cpu_total_runtime_bytes": self._mib_to_bytes(cpu_total_runtime_mib),
            "total_runtime_bytes": self._mib_to_bytes(total_runtime_mib),
        }

    def _with_runtime_totals(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of loaded model info with:
        - Ollama-like ps fields (size, size_vram, size_ram)
        - debug/runtime totals
        """
        item = {**model_info, **self._build_memory_accounting(model_info)}

        # Ollama-compatible public fields
        item["size_vram"] = item["ps_size_vram_bytes"]
        item["size_ram"] = item["cpu_loaded_bytes"]
        item["size"] = item["ps_size_bytes"]

        return item

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

    def _load_model_sync(
        self,
        model_path: str,
        requested_n_ctx: int,
        virtual_spec: Optional[VirtualModelSpec] = None,
    ):
        llama_kwargs = self._build_llama_load_kwargs(model_path, requested_n_ctx, virtual_spec)
        llm, stats, log_text = load_llama_with_captured_stats(
            Llama,
            after_load=_initialise_projector,
            **llama_kwargs,
        )

        _apply_template_override_to_chat_handler(llm, virtual_spec)

        return llm, stats, log_text

    def _ensure_capacity_for_load(self, requested_model_name: str) -> None:
        if requested_model_name in self.models:
            return

        # Loads already under way occupy capacity even though the models are
        # not in self.models yet. _blocking_load makes a caller wait when they
        # leave no room, so anything reaching here has room by that count.
        in_flight = sum(1 for name in self._loading if name != requested_model_name)

        if self.config.max_loaded_models <= 1:
            for loaded_model_name in list(self.models.keys()):
                if loaded_model_name != requested_model_name:
                    self.unload_model(loaded_model_name)
            return

        if len(self.models) + in_flight >= self.config.max_loaded_models:
            raise RuntimeError(
                f"Loaded model limit reached ({self.config.max_loaded_models}). "
                "Unload a model first or increase TLLAMA_MAX_LOADED_MODELS."
            )

    def _inference_slots_for(self, model_name: str) -> int:
        """How many generations a model can run at once.

        One, because llama-cpp-python gives no way to have more. Its Llama
        object owns a single context and a single KV cache, and neither the
        constructor nor create_completion accepts a sequence id, so two
        concurrent generations on one object corrupt each other's state. The
        library's own server serialises every request globally for this
        reason; per model is the same guarantee, only finer.

        llama.cpp itself can do better. A context carries n_seq_max and a
        batch carries a seq_id, which is how llama-server serves --parallel N
        with the weights loaded once and only the KV cache multiplied. Reaching
        that means replacing or extending the binding layer, and when it
        happens this method is what changes.
        """
        return 1

    @asynccontextmanager
    async def acquire_inference_slot(self, model_name: str):
        """Hold a generation slot for a model for the duration of a request.

        Callers must hold this for the whole generation, streaming included,
        and must not assume anything about ordering between models: slots are
        per model, so different models run concurrently.
        """
        async with self._slots_lock:
            slots = self._inference_slots.get(model_name)
            if slots is None:
                slots = asyncio.Semaphore(self._inference_slots_for(model_name))
                self._inference_slots[model_name] = slots

        async with slots:
            yield

    def _ensure_janitor_running(self) -> None:
        if self._janitor_task is None or self._janitor_task.done():
            self._janitor_task = asyncio.create_task(
                self._janitor_loop(),
                name="tllama-model-janitor",
            )

    def _refresh_active_model(self, model_name: str, keep_alive_seconds: int | None) -> None:
        now_iso = self._now_iso()

        entry = self.active_models[model_name]
        entry["last_used_at"] = now_iso
        entry["expires_at"] = None if keep_alive_seconds is None else self._future_iso(keep_alive_seconds)
        entry["keep_alive"] = keep_alive_seconds

    def _register_loaded_model(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        llm: Llama,
        load_stats: Dict[str, Any],
        keep_alive_seconds: int | None,
        toml_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> None:
        actual_n_ctx = llm.n_ctx()
        now_iso = self._now_iso()

        if keep_alive_seconds is None:
            expires_at = None
        else:
            expires_at = self._future_iso(keep_alive_seconds)

        self.models[model_name] = llm
        self.active_models[model_name] = {
                    "id": model_name,
                    "model": model_name,
                    "filename": model_info["filename"],
                    "path": model_info["path"],
                    "size": model_info["size"],
                    "mtime": model_info["mtime"],
                    "sha256": model_info["sha256"],
                    "loaded_at": now_iso,
                    "last_used_at": now_iso,
                    "expires_at": expires_at,
                    "keep_alive": keep_alive_seconds,
                    "n_ctx": actual_n_ctx,

                    # The revision of the definition this model was built
                    # from, so a later request can tell whether the file
                    # still says what it said. Taken before the load rather
                    # than after: a .toml edited while the load was running
                    # did not shape the model that came out of it, and
                    # recording the newer revision here would hide that.
                    #
                    # Internal to the manager. /api/ps builds its response
                    # from named keys, so nothing here reaches a client.
                    "toml_fingerprint": toml_fingerprint,

                    # Stats from load log
                    "processor": load_stats.get("processor", "100% CPU"),
                    "offloaded_layers": load_stats.get("offloaded_layers", 0),
                    "total_layers": load_stats.get("total_layers", 0),
                    "gpu_model_mib": load_stats.get("gpu_model_mib", 0.0),
                    "gpu_kv_mib": load_stats.get("gpu_kv_mib", 0.0),
                    "gpu_compute_mib": load_stats.get("gpu_compute_mib", 0.0),
                    "gpu_output_mib": load_stats.get("gpu_output_mib", 0.0),
                    "gpu_rs_mib": load_stats.get("gpu_rs_mib", 0.0),
                    "cpu_model_mib": load_stats.get("cpu_model_mib", 0.0),
                    "cpu_kv_mib": load_stats.get("cpu_kv_mib", 0.0),
                    "cpu_compute_mib": load_stats.get("cpu_compute_mib", 0.0),
                    "cpu_output_mib": load_stats.get("cpu_output_mib", 0.0),
                    "cpu_rs_mib": load_stats.get("cpu_rs_mib", 0.0),
                    "gpu_host_model_mib": load_stats.get("gpu_host_model_mib", 0.0),
                    "gpu_host_kv_mib": load_stats.get("gpu_host_kv_mib", 0.0),
                    "gpu_host_compute_mib": load_stats.get("gpu_host_compute_mib", 0.0),
                    "gpu_host_output_mib": load_stats.get("gpu_host_output_mib", 0.0),
                    "gpu_host_rs_mib": load_stats.get("gpu_host_rs_mib", 0.0),

                    **self._build_memory_accounting(load_stats),
                }

    def _blocking_load(self, model_name: str) -> Optional[asyncio.Future]:
        """A load already under way that leaves no room for another one.

        Capacity counts loaded models plus loads in progress. Counting only
        the loaded ones would let two concurrent requests both pass the check
        and exceed the limit, or exhaust memory outright, because a model
        being loaded is not in self.models yet.
        """
        in_flight = [
            future for name, future in self._loading.items()
            if name != model_name
        ]

        if not in_flight:
            return None

        if self.config.max_loaded_models > len(self.models) + len(in_flight):
            return None

        return in_flight[0]

    async def get_model(
        self,
        model_name: str,
        num_ctx: int | None = None,
        keep_alive: str | int | float | None = None,
    ) -> Llama:
        """Return a loaded model, loading it first if necessary.

        The load itself runs outside the lock. Holding it across an await of
        tens of seconds meant that loading one model stopped everything else,
        including work on models already resident.

        Concurrent callers therefore have to be told apart. Asking for the
        model already being loaded means waiting for that load rather than
        starting a second one. Asking for a different model when there is no
        room for it means waiting for the load in progress to finish, which
        is the serialising behaviour a single-model limit implies.
        """
        while True:
            async with self._models_lock:
                self._ensure_janitor_running()

                try:
                    model_info = self._build_model_file_info(model_name)
                except TomlModelError as exc:
                    # TomlModelError intentionally not swallowed: a broken
                    # .toml or an out-of-repo path in one should stop this
                    # request, not be treated as "no virtual model"
                    # silently. It does mean the resident model, if any,
                    # can no longer be reached by anything.
                    self._drop_unreachable_model(model_name, str(exc))
                    raise

                if not model_info:
                    self._drop_unreachable_model(
                        model_name,
                        f"nothing in {self.models_dir} defines it any more",
                    )
                    raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")

                virtual_spec = self._resolve_virtual_model_spec(model_name)

                effective_num_ctx = num_ctx
                if effective_num_ctx is None and virtual_spec is not None:
                    effective_num_ctx = virtual_spec.runtime.get("n_ctx")
                if effective_num_ctx is None:
                    effective_num_ctx = self.config.context_length

                requested_n_ctx = self._normalize_num_ctx(effective_num_ctx, default=0)
                keep_alive_seconds = self.resolve_keep_alive(keep_alive)

                toml_fingerprint = self._toml_fingerprint_for_reference(model_name)

                loaded_entry = self.active_models.get(model_name, {})
                current_n_ctx = loaded_entry.get("n_ctx")
                loaded_toml_fingerprint = loaded_entry.get("toml_fingerprint")

                # An edited definition has to reach the model it defines.
                #
                # Only some of a .toml takes effect without a reload:
                # [sampling] and [system] are read through
                # get_model_metadata() at generation time, so they already
                # follow the file. [runtime], [llm] model, [mmproj] and a
                # vision [template] are all consumed while Llama() is being
                # constructed, and nothing afterwards can reach them.
                #
                # Reloading on any change rather than on those sections is
                # the deliberate choice: telling them apart would make the
                # rule depend on which key someone edited, which is not
                # something a person editing a file should have to know.
                # It also makes `touch model.toml` a way to force a reload.
                #
                # Without this the reload only ever happened when a client
                # sent num_ctx explicitly and it differed. `ollama run`
                # sends no num_ctx at all, so an edited .toml was answered
                # by the model built from the previous one, indefinitely.
                if model_name in self.models and (
                    (num_ctx is not None and requested_n_ctx != current_n_ctx)
                    or toml_fingerprint != loaded_toml_fingerprint
                ):
                    self.unload_model(model_name)

                if model_name in self.models:
                    self._refresh_active_model(model_name, keep_alive_seconds)
                    return self.models[model_name]

                pending = self._loading.get(model_name)
                loading_here = False
                waiting_for_this_model = pending is not None

                if pending is None:
                    blocking = self._blocking_load(model_name)
                    if blocking is None:
                        self._ensure_capacity_for_load(model_name)
                        pending = asyncio.get_running_loop().create_future()
                        self._loading[model_name] = pending
                        loading_here = True
                    else:
                        pending = blocking

            if not loading_here:
                # Shielded: whoever is waiting may be cancelled without
                # taking the load, and everyone else waiting on it, down.
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    if not pending.cancelled():
                        raise
                    # Abandoned rather than failed, so there is nothing to
                    # inherit and the situation is worth another look.
                except Exception:
                    if waiting_for_this_model:
                        # The same model, and it will fail the same way for
                        # this caller too. Repeating a doomed load once per
                        # waiter helps nobody.
                        raise

                    # A different model was occupying the capacity. Its
                    # failure says nothing about this request, and it has
                    # freed the room.

                continue

            logger.debug("Loading model %s with n_ctx=%s", model_name, requested_n_ctx)

            try:
                llm, load_stats, load_log = await asyncio.to_thread(
                    self._load_model_sync,
                    model_info["path"],
                    requested_n_ctx,
                    virtual_spec,
                )
            except BaseException as exc:
                async with self._models_lock:
                    self._loading.pop(model_name, None)

                if not pending.done():
                    if isinstance(exc, asyncio.CancelledError):
                        pending.cancel()
                    else:
                        pending.set_exception(exc)
                        # Nobody may be waiting, and an unretrieved exception
                        # on a future is reported at garbage collection.
                        pending.exception()

                raise

            async with self._models_lock:
                self._loading.pop(model_name, None)
                self._register_loaded_model(
                    model_name,
                    model_info,
                    llm,
                    load_stats,
                    keep_alive_seconds,
                    toml_fingerprint,
                )

            if not pending.done():
                pending.set_result(None)

            return llm

    def _drop_unreachable_model(self, model_name: str, reason: str) -> None:
        """Free a resident model that nothing can ask for any more.

        A model is only reachable through get_model(), which re-reads the
        .toml before it will hand the resident one back. Once that file
        will not parse, or names something that is not there, or is gone,
        every request for the model fails on the way in -- while the
        weights stay in memory, and with keep_alive unset stay there until
        the process ends or a capacity limit evicts them. Unreachable and
        resident at the same time is a leak with a name.

        Deleting a definition through /api/delete already unloads the model
        first, for the same reason. This is the same event arriving by a
        different route: someone editing or removing the file directly.

        Safe mid-generation. unload_model() drops the manager's reference
        and nothing else, so a request already generating keeps the one it
        holds and the weights go when it is done.
        """
        if model_name not in self.models and model_name not in self.active_models:
            return

        logger.warning("Unloading %s: %s", model_name, reason)
        self.unload_model(model_name)

    def unload_model(self, model_name: str):
        self._unload_model_internal(model_name)

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    def get_loaded_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        model_info = self.active_models.get(model_name)
        if model_info is None:
            return None
        return self._with_runtime_totals(model_info)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [self._with_runtime_totals(model_info) for model_info in self.active_models.values()]

    def list_loading_models(self) -> List[str]:
        """Names of models currently being loaded but not yet resident.

        A load in progress occupies capacity and is meaningful to show
        alongside already-loaded models, mirroring what Ollama does. Only
        the name is guaranteed here: everything else about the model (VRAM
        split, context size actually applied) is not known until the load
        finishes and `_register_loaded_model` runs.
        """
        return list(self._loading.keys())

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
        timeout_seconds: float | None = None,
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
        metadata = await self._get_raw_model_metadata(model_name, timeout_seconds)
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
            overrides["has_projector"] = (
                self.resolve_mmproj_path(virtual_spec, model_name) is not None
            )

            if overrides:
                metadata = {**metadata, **overrides}

        return metadata

    async def _get_raw_model_metadata(
        self,
        model_name: str,
        timeout_seconds: float | None = None,
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
            scan_task = asyncio.to_thread(self._get_model_metadata_sync, model_info["path"])

            if timeout_seconds is None:
                metadata = await scan_task
            else:
                metadata = await asyncio.wait_for(scan_task, timeout=timeout_seconds)

        except asyncio.TimeoutError:
            logger.warning("Metadata scan timed out for model %s", model_name)
            return None
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

        action = "overwrite" if already_defined else "create"

        if dry_run:
            return outcome(action, toml_path, "")

        try:
            if already_defined:
                shutil.copy2(toml_path, toml_path.with_name(toml_path.name + ".bak"))

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

    async def get_model_metadata_best_effort(
        self,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """get_model_metadata(), but a broken .toml yields None.

        For callers reporting on a model whose existence is established by
        something other than its .toml. /api/ps reports what is resident in
        memory right now: a model loaded before its .toml was edited is
        still loaded, still holding the memory it holds, and still able to
        answer requests, so leaving it out would describe the machine's
        actual state less accurately than listing it with whatever is still
        known about it.

        Every other caller wants the exception. For /api/show or an
        inference request a broken .toml is the answer to the question
        being asked, not an inconvenience along the way -- which is why
        this is a separate method rather than a change to the default.
        """
        try:
            return await self.get_model_metadata(model_name)
        except TomlModelError as e:
            logger.warning("Metadata unavailable for %s: %s", model_name, e)
            return None

    def build_model_file_info_best_effort(self, model_name: str) -> Optional[Dict[str, Any]]:
        """_build_model_file_info(), but a broken .toml yields None.

        Same reasoning as get_model_metadata_best_effort, for the model
        whose load /api/ps reports as still in flight.
        """
        try:
            return self._build_model_file_info(model_name)
        except TomlModelError as e:
            logger.warning("File info unavailable for %s: %s", model_name, e)
            return None

    def _resolve_global_kv_cache_types(self) -> "tuple[Optional[int], Optional[int]]":
        """Resolve the global cache type names into ggml type ints.

        The three variables are handed to the resolver [runtime] uses, in
        the shape it expects, so TLLAMA_KV_CACHE_TYPE means what type_kv
        means and the per-side ones mean what type_k/type_v mean. Two
        implementations of the same precedence would be free to drift; one
        cannot.

        A name is looked up dynamically against the GGML_TYPE_* constants,
        so this accepts whatever the installed build knows rather than a
        list written here.

        Errors are translated on the way out: resolve_kv_cache_types
        reports about a .toml field, which is the wrong thing to tell
        someone who set an environment variable. ConfigError is what the
        entry point turns into a message and a non-zero exit; both it and
        TomlModelError subclass ValueError, so anything catching the
        broader type is unaffected.
        """
        by_variable = {
            "type_kv": ("TLLAMA_KV_CACHE_TYPE", self.config.kv_cache_type),
            "type_k": ("TLLAMA_K_CACHE_TYPE", self.config.k_cache_type),
            "type_v": ("TLLAMA_V_CACHE_TYPE", self.config.v_cache_type),
        }

        # Only the ones actually set. A key present with None would read as
        # "explicitly nothing" to the resolver and defeat the type_kv
        # fallback, which is the opposite of what an unset variable means.
        runtime = {
            field: value
            for field, (_, value) in by_variable.items()
            if value
        }

        if not runtime:
            return None, None

        # Each variable is checked on its own first, purely so a rejection
        # can name the one that was actually wrong. Reading the field name
        # back out of the combined failure does not work: with only
        # TLLAMA_KV_CACHE_TYPE set, the resolver reports type_k, because
        # that is the field the value reached through the fallback.
        for field, value in runtime.items():
            variable = by_variable[field][0]
            try:
                resolve_kv_cache_types({"type_kv": value})
            except TomlModelError as exc:
                raise ConfigError(
                    f"Unsupported {variable} value: {value}. "
                    "Expected a ggml type name such as f16, q8_0 or q4_0 "
                    f"({exc})."
                ) from exc

        # The precedence itself stays where the .toml gets it from.
        return resolve_kv_cache_types(runtime)

    def _build_llama_load_kwargs(
        self,
        model_path: str,
        requested_n_ctx: int,
        virtual_spec: Optional[VirtualModelSpec] = None,
    ) -> Dict[str, Any]:
        # A value belongs here only when TLlama has a reason to differ from
        # llama-cpp-python's own default; anything else is left out so that
        # default applies, rather than being restated as a shadow copy that
        # can drift from it. use_mmap is the case in point: it was pinned to
        # False as a workaround for a memory leak in a since-superseded GPU
        # driver / llama.cpp combination, and holding it there now overrides
        # a library default of True for a reason that no longer exists --
        # and would also override the library's own handling, which turns
        # mmap off by itself when a LoRA is in play.
        #
        # n_gpu_layers and verbose stay, for two different reasons:
        #
        # - n_gpu_layers: the library defaults to 0, i.e. offloading
        #   nothing. Offloading everything by default is TLlama policy
        #   (Ollama behaves the same way), so it is a genuine difference,
        #   not a restatement.
        # - verbose: the library also defaults to True, so this one does
        #   agree -- but it is a dependency rather than a preference.
        #   /api/ps gets every number it reports from
        #   parse_llama_verbose_load_log() reading the load log, so
        #   verbose=False empties the memory accounting entirely. It is
        #   explicit to keep that from being tidied away as redundant.
        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": requested_n_ctx,
            "n_gpu_layers": -1,
            "verbose": True,
        }

        # Server-wide Llama() arguments. Below a model's [runtime], which
        # is applied after this and overwrites what it names.
        kwargs.update(self._runtime_overrides)

        if self._type_k is not None:
            kwargs["type_k"] = self._type_k
        if self._type_v is not None:
            kwargs["type_v"] = self._type_v

        if virtual_spec is not None:
            # 1:1 passthrough of [runtime] into Llama() kwargs (spec doc
            # §8), minus the handful of keys handled specially above/below:
            # model_path and n_ctx are controlled by [llm] and the
            # request/global-config priority chain, not by [runtime]
            # directly, and type_k/type_v/type_kv need the name-to-ggml-enum
            # resolution in resolve_kv_cache_types rather than a raw pass.
            for key, value in virtual_spec.runtime.items():
                if key in ("model_path", "n_ctx", "type_k", "type_v", "type_kv"):
                    continue
                kwargs[key] = value

            type_k, type_v = resolve_kv_cache_types(virtual_spec.runtime)
            if type_k is not None:
                kwargs["type_k"] = type_k
            if type_v is not None:
                kwargs["type_v"] = type_v

        chat_handler = self._build_vision_chat_handler(virtual_spec, model_path)
        if chat_handler is not None:
            kwargs["chat_handler"] = chat_handler

        return kwargs

    def _build_vision_chat_handler(
        self, virtual_spec: Optional[VirtualModelSpec], source: str
    ):
        """The projector handler a virtual model's [mmproj] calls for.

        None when there is no projector to load, which is every model that
        has no [mmproj] section -- Llama() then resolves its own text
        handler exactly as before.

        MTMDChatHandler is the default because it takes the chat template
        from the model itself rather than carrying a hardcoded one, so it
        works with any model whose template can render content parts. The
        older Llava15ChatHandler family each bakes in its own prompt format
        and would impose it on the model; picking one of those is a
        deliberate choice for a specific model, not a default.

        Constructed here rather than after loading because mtmd needs the
        model to initialise against, and Llama() only wires a handler up to
        the model when it is passed in at construction time.
        """
        if virtual_spec is None:
            return None

        projector_path = self.resolve_mmproj_path(virtual_spec, source)
        if projector_path is None:
            return None

        if virtual_spec.template is not None:
            return _TemplateOverridingMTMDChatHandler(
                clip_model_path=str(projector_path),
                template=virtual_spec.template,
            )

        return MTMDChatHandler(clip_model_path=str(projector_path))

    async def start(self):
        self.ensure_storage()

        async with self._models_lock:
            if self._janitor_task is None or self._janitor_task.done():
                self._janitor_task = asyncio.create_task(
                    self._janitor_loop(),
                    name="tllama-model-janitor",
                )

    async def shutdown(self):
        janitor_task = None

        async with self._models_lock:
            if self._janitor_task is not None:
                janitor_task = self._janitor_task
                self._janitor_task = None

        if janitor_task is not None:
            janitor_task.cancel()
            try:
                await janitor_task
            except asyncio.CancelledError:
                pass

        async with self._models_lock:
            self.unload_all_models()

    async def _janitor_loop(self):
        try:
            while True:
                await asyncio.sleep(self.config.janitor_interval_seconds)

                async with self._models_lock:
                    expired_model_names = [
                        model_name
                        for model_name, model_info in self.active_models.items()
                        if self._is_model_entry_expired(model_info)
                    ]

                    for model_name in expired_model_names:
                        logger.debug("Auto-unloading expired model %s", model_name)
                        self._unload_model_internal(model_name)
        except asyncio.CancelledError:
            raise

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
        return await asyncio.to_thread(self.delete_model_definition, model_ref)

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

