import os
import time
import asyncio
import gc
import shutil

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

from llama_cpp import Llama, llama_cpp as llama_cpp_lib
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone, timedelta

from tllama.config import BackendConfig, load_backend_config_from_env
from tllama.helpers.common import normalize_keep_alive
from tllama.helpers.llama_stats import load_llama_with_captured_stats
from tllama.helpers.gguf_metadata import read_gguf_metadata, build_model_metadata_payload
from tllama.helpers.metadata_cache import (
    load_metadata_cache,
    save_metadata_cache,
    delete_metadata_cache,
    load_digest_cache,
    save_digest_cache,
    delete_digest_cache
)

__all__ = ('model_manager', 'load_backend_config_from_env')

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


@dataclass(frozen=True)
class CachedMetadataEntry:
    fingerprint: str
    cached_at_monotonic: float
    value: Dict[str, Any]


class ModelManager:
    def __init__(self, config: BackendConfig | None = None):
        self.config = config or load_backend_config_from_env()

        self.models: Dict[str, Llama] = {}
        self._lock = asyncio.Lock()

        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_cache_dir = self.models_dir / ".tllama" / "metadata-cache"
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)

        self.active_models: Dict[str, Dict[str, Any]] = {}

        self._janitor_task: asyncio.Task | None = None

        self._metadata_cache: Dict[str, CachedMetadataEntry] = {}

        self.hf_models_dir = self.models_dir / "HuggingFace"
        self.local_models_dir = self.models_dir / "Local"
        self.tllama_models_dir = self.models_dir / "TLlama"

        self.hf_models_dir.mkdir(parents=True, exist_ok=True)
        self.local_models_dir.mkdir(parents=True, exist_ok=True)
        self.tllama_models_dir.mkdir(parents=True, exist_ok=True)

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

    def _build_model_file_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            model_path = self.resolve_model_storage_path(model_name)
        except ValueError:
            return None

        return self._build_model_file_info_from_path(model_path)

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

        # True residency buckets for Ollama-like processor split
        gpu_loaded_mib = gpu_model_mib + gpu_kv_mib
        cpu_loaded_mib = cpu_model_mib + cpu_kv_mib
        loaded_total_mib = gpu_loaded_mib + cpu_loaded_mib

        # Ollama-like ps size:
        # - count real GPU-loaded model+KV
        # - count true CPU-loaded model+KV
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
            cpu_rs_mib
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
        - Ollama-like ps fields (size, size_vram)
        - debug/runtime totals
        """
        item = dict(model_info)

        gpu_model_mib = self._to_float_mib(item.get("gpu_model_mib"))
        gpu_kv_mib = self._to_float_mib(item.get("gpu_kv_mib"))
        gpu_compute_mib = self._to_float_mib(item.get("gpu_compute_mib"))
        gpu_output_mib = self._to_float_mib(item.get("gpu_output_mib"))
        gpu_rs_mib = self._to_float_mib(item.get("gpu_rs_mib"))

        gpu_host_model_mib = self._to_float_mib(item.get("gpu_host_model_mib"))
        gpu_host_kv_mib = self._to_float_mib(item.get("gpu_host_kv_mib"))
        gpu_host_compute_mib = self._to_float_mib(item.get("gpu_host_compute_mib"))
        gpu_host_output_mib = self._to_float_mib(item.get("gpu_host_output_mib"))
        gpu_host_rs_mib = self._to_float_mib(item.get("gpu_host_rs_mib"))

        cpu_model_mib = self._to_float_mib(item.get("cpu_model_mib"))
        cpu_kv_mib = self._to_float_mib(item.get("cpu_kv_mib"))
        cpu_compute_mib = self._to_float_mib(item.get("cpu_compute_mib"))
        cpu_output_mib = self._to_float_mib(item.get("cpu_output_mib"))
        cpu_rs_mib = self._to_float_mib(item.get("cpu_rs_mib"))

        # True residency buckets
        gpu_loaded_mib = gpu_model_mib + gpu_kv_mib
        cpu_loaded_mib = cpu_model_mib + cpu_kv_mib
        loaded_total_mib = gpu_loaded_mib + cpu_loaded_mib

        # Ollama-like ps-facing size fields
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

        # Full runtime/debug totals
        gpu_total_runtime_mib = (
            gpu_model_mib +
            gpu_kv_mib +
            gpu_compute_mib +
            gpu_output_mib +
            gpu_rs_mib +
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
            cpu_rs_mib
        )

        total_runtime_mib = gpu_total_runtime_mib + cpu_total_runtime_mib

        item["gpu_loaded_mib"] = gpu_loaded_mib
        item["cpu_loaded_mib"] = cpu_loaded_mib
        item["loaded_total_mib"] = loaded_total_mib

        item["gpu_loaded_bytes"] = self._mib_to_bytes(gpu_loaded_mib)
        item["cpu_loaded_bytes"] = self._mib_to_bytes(cpu_loaded_mib)
        item["loaded_total_bytes"] = self._mib_to_bytes(loaded_total_mib)

        item["ps_size_vram_mib"] = ps_size_vram_mib
        item["ps_size_mib"] = ps_size_mib
        item["ps_size_vram_bytes"] = self._mib_to_bytes(ps_size_vram_mib)
        item["ps_size_bytes"] = self._mib_to_bytes(ps_size_mib)

        item["gpu_total_runtime_mib"] = gpu_total_runtime_mib
        item["cpu_total_runtime_mib"] = cpu_total_runtime_mib
        item["total_runtime_mib"] = total_runtime_mib

        item["gpu_total_runtime_bytes"] = self._mib_to_bytes(gpu_total_runtime_mib)
        item["cpu_total_runtime_bytes"] = self._mib_to_bytes(cpu_total_runtime_mib)
        item["total_runtime_bytes"] = self._mib_to_bytes(total_runtime_mib)

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

    def _load_model_sync(self, model_path: str, requested_n_ctx: int):
        llama_kwargs = self._build_llama_load_kwargs(model_path, requested_n_ctx)
        return load_llama_with_captured_stats(
            Llama,
            **llama_kwargs,
        )

    def _ensure_capacity_for_load(self, requested_model_name: str) -> None:
        if requested_model_name in self.models:
            return

        if self.config.max_loaded_models <= 1:
            for loaded_model_name in list(self.models.keys()):
                if loaded_model_name != requested_model_name:
                    self.unload_model(loaded_model_name)
            return

        if len(self.models) >= self.config.max_loaded_models:
            raise RuntimeError(
                f"Loaded model limit reached ({self.config.max_loaded_models}). "
                "Unload a model first or increase TLLAMA_MAX_LOADED_MODELS."
            )

    async def get_model(
        self,
        model_name: str,
        num_ctx: int | None = None,
        keep_alive: str | int | float | None = None,
    ) -> Llama:
        async with self._lock:
            if self._janitor_task is None or self._janitor_task.done():
                self._janitor_task = asyncio.create_task(
                    self._janitor_loop(),
                    name="tllama-model-janitor",
                )

            model_info = self._build_model_file_info(model_name)
            if not model_info:
                raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")

            model_path = model_info["path"]

            effective_num_ctx = num_ctx
            if effective_num_ctx is None:
                effective_num_ctx = self.config.context_length

            requested_n_ctx = self._normalize_num_ctx(effective_num_ctx, default=0)
            keep_alive_seconds = self.resolve_keep_alive(keep_alive)

            current_n_ctx = self.active_models.get(model_name, {}).get("n_ctx")

            # Reload only when caller explicitly requested a different context size
            if model_name in self.models and num_ctx is not None and requested_n_ctx != current_n_ctx:
                self.unload_model(model_name)

            if model_name not in self.models:
                self._ensure_capacity_for_load(model_name)

                print(f"DEBUG: Loading model {model_name} with n_ctx={requested_n_ctx}...")

                llm, load_stats, load_log = await asyncio.to_thread(
                    self._load_model_sync,
                    model_path,
                    requested_n_ctx,
                )

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
                    "n_gpu_layers": -1,
                    "use_mmap": False,
                    "flash_attention": self.config.flash_attention,
                    "kv_cache_type": self.config.kv_cache_type,

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
            else:
                now_iso = self._now_iso()

                if keep_alive_seconds is None:
                    expires_at = None
                else:
                    expires_at = self._future_iso(keep_alive_seconds)

                self.active_models[model_name]["last_used_at"] = now_iso
                self.active_models[model_name]["expires_at"] = expires_at
                self.active_models[model_name]["keep_alive"] = keep_alive_seconds

            return self.models[model_name]

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
            print(f"DEBUG: Digest computation failed for {file_path}: {exc}")
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
        Runs off the event loop and uses a lightweight TTL cache.
        """
        model_info = self._build_model_file_info(model_name)
        if not model_info:
            return None

        fingerprint = model_info["sha256"]

        async with self._lock:
            cached_value = self._get_cached_metadata_entry(model_name, fingerprint)
            if cached_value is not None:
                return cached_value

        persistent_cached_value = await asyncio.to_thread(
            load_metadata_cache,
            self.metadata_cache_dir,
            model_info["path"],
        )

        if persistent_cached_value is not None:
            async with self._lock:
                self._set_cached_metadata_entry(model_name, fingerprint, persistent_cached_value)
            return persistent_cached_value

        try:
            scan_task = asyncio.to_thread(self._get_model_metadata_sync, model_info["path"])

            if timeout_seconds is None:
                metadata = await scan_task
            else:
                metadata = await asyncio.wait_for(scan_task, timeout=timeout_seconds)

        except asyncio.TimeoutError:
            print(f"DEBUG: Metadata scan timed out for model {model_name}")
            return None
        except Exception as exc:
            print(f"DEBUG: Metadata scan failed for model {model_name}: {exc}")
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

        async with self._lock:
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
            print(f"DEBUG: Cannot resolve metadata cache target for {file_path}: {exc}")
            return None

        if model_info is None:
            return None

        fingerprint = model_info["sha256"]

        async with self._lock:
            cached_value = self._get_cached_metadata_entry(model_name, fingerprint)
            if cached_value is not None:
                return cached_value

        persistent_cached_value = await asyncio.to_thread(
            load_metadata_cache,
            self.metadata_cache_dir,
            model_info["path"],
        )

        if persistent_cached_value is not None:
            async with self._lock:
                self._set_cached_metadata_entry(model_name, fingerprint, persistent_cached_value)
            return persistent_cached_value

        try:
            metadata = await asyncio.to_thread(
                self._get_model_metadata_sync,
                model_info["path"],
            )
        except Exception as exc:
            print(f"DEBUG: Metadata cache creation failed for {model_name}: {exc}")
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

        async with self._lock:
            self._set_cached_metadata_entry(model_name, fingerprint, metadata)

        return metadata

    def _list_local_models_sync(self) -> List[Dict[str, Any]]:
        """
        Scan all known repositories for GGUF files.
        One broken file must not break the whole listing.
        """
        model_list: List[Dict[str, Any]] = []

        for file_path in self._iter_repository_model_files():
            try:
                model_info = self._build_model_file_info_from_path(file_path)
                if model_info is None:
                    continue

                model_info["id"] = self._build_model_ref_from_path(file_path)

                if file_path.is_relative_to(self.hf_models_dir):
                    model_info["repository"] = "HuggingFace"
                elif file_path.is_relative_to(self.local_models_dir):
                    model_info["repository"] = "Local"
                elif file_path.is_relative_to(self.tllama_models_dir):
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
            except Exception as e:
                print(f"DEBUG: Failed to inspect model file {file_path}: {e}")

        return model_list

    async def list_local_models(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._list_local_models_sync)

    async def list_local_models_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Return local models enriched with metadata.
        Metadata failures are isolated per model.
        """
        models = await self.list_local_models()
        enriched: List[Dict[str, Any]] = []

        for model in models:
            item = dict(model)
            metadata = await self.get_model_metadata(model["id"])
            if metadata:
                item.update(metadata)
            enriched.append(item)

        return enriched

    def _resolve_kv_cache_type(self, value: str | None) -> int | None:
        if not value:
            return None

        normalized = value.strip().lower()

        name_map = {
            "f16": "GGML_TYPE_F16",
            "q8_0": "GGML_TYPE_Q8_0",
            "q4_0": "GGML_TYPE_Q4_0",
        }

        constant_name = name_map.get(normalized)
        if constant_name is None:
            raise ValueError(
                f"Unsupported TLLAMA_KV_CACHE_TYPE value: {value}. "
                "Supported values: f16, q8_0, q4_0."
            )

        resolved = getattr(llama_cpp_lib, constant_name, None)
        if resolved is None:
            raise ValueError(
                f"KV cache type constant {constant_name} is not available in this llama-cpp-python build."
            )

        return int(resolved)

    def _build_llama_load_kwargs(self, model_path: str, requested_n_ctx: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": requested_n_ctx,
            "n_gpu_layers": -1,
            "use_mmap": False,
            "verbose": True,
        }

        if self.config.flash_attention:
            kwargs["flash_attn"] = True

        kv_cache_type = self._resolve_kv_cache_type(self.config.kv_cache_type)
        if kv_cache_type is not None:
            kwargs["type_k"] = kv_cache_type
            kwargs["type_v"] = kv_cache_type

        return kwargs

    async def start(self):
        async with self._lock:
            if self._janitor_task is None or self._janitor_task.done():
                self._janitor_task = asyncio.create_task(
                    self._janitor_loop(),
                    name="tllama-model-janitor",
                )

    async def shutdown(self):
        janitor_task = None

        async with self._lock:
            if self._janitor_task is not None:
                janitor_task = self._janitor_task
                self._janitor_task = None

        if janitor_task is not None:
            janitor_task.cancel()
            try:
                await janitor_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self.unload_all_models()

    async def _janitor_loop(self):
        try:
            while True:
                await asyncio.sleep(self.config.janitor_interval_seconds)

                async with self._lock:
                    expired_model_names = [
                        model_name
                        for model_name, model_info in self.active_models.items()
                        if self._is_model_entry_expired(model_info)
                    ]

                    for model_name in expired_model_names:
                        print(f"DEBUG: Auto-unloading expired model {model_name}")
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

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=target_root,
            local_dir_use_symlinks=False,
        )

    async def pull_hf_file(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        revision: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._pull_hf_file_sync,
            repo_id,
            filename,
            token,
            revision,
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
            print(f"DEBUG: HuggingFace file info lookup failed for {repo_id}/{filename}: {exc}")
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
            print(f"DEBUG: Cannot stat {file_path} for digest recording: {exc}")
            return None

        if actual_size != expected_size:
            print(
                f"DEBUG: Size mismatch for {file_path} "
                f"(published {expected_size}, on disk {actual_size}); "
                "digest will be computed locally"
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

        Supported forms:
        - HuggingFace: namespace/repo/path/to/file[.gguf]
        - Local:       Local/name            -> Local/name/model.gguf
        - Local:       Local/path/to/file    -> Local/path/to/file.gguf
        - TLlama:      TLlama/name           -> TLlama/name/model.gguf
        - TLlama:      TLlama/path/to/file   -> TLlama/path/to/file.gguf
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
            local_candidates = [
                _normalized_file_path(self.local_models_dir, parts),
                self.local_models_dir.joinpath(*parts, "model.gguf"),
            ]

            tllama_legacy_candidates = [
                self.tllama_models_dir.joinpath(*parts, "model.gguf"),
            ]

            return _first_existing_or_default(local_candidates + tllama_legacy_candidates)

        # Prefixless TLlama reference.
        # Example: "collection/model-file"
        #
        # If a matching Local nested file exists and the TLlama file does not, keep it
        # usable as a fallback. Explicit "Local/..." remains the unambiguous form.
        if len(parts) == 2:
            tllama_candidates = [
                _normalized_file_path(self.tllama_models_dir, parts),
                self.tllama_models_dir.joinpath(*parts, "model.gguf"),
            ]

            local_fallback_candidates = [
                _normalized_file_path(self.local_models_dir, parts),
                self.local_models_dir.joinpath(*parts, "model.gguf"),
            ]

            return _first_existing_or_default(tllama_candidates + local_fallback_candidates)

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

    def delete_model_file(self, model_ref: str) -> Dict[str, Any]:
        target_path = self.resolve_model_storage_path(model_ref)

        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {target_path}")

        if not target_path.is_file():
            raise ValueError(f"Target is not a file: {target_path}")

        try:
            target_path.unlink()
        except FileNotFoundError:
            pass

        delete_metadata_cache(self.metadata_cache_dir, target_path)
        delete_digest_cache(self.metadata_cache_dir, target_path)
        self._invalidate_metadata_cache_entry(model_ref)

        self._cleanup_hf_repo_auxiliary(target_path)

        # Best-effort cleanup of empty parent directories, but do not fail
        # if other files remain in the directory tree.
        repo_root = self._get_repo_root_for_path(target_path)
        self._remove_empty_parents(target_path.parent, repo_root)

        return {
            "model_ref": model_ref,
            "deleted_path": str(target_path),
        }

    async def delete_model(self, model_ref: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.delete_model_file, model_ref)

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
            rel = file_path.relative_to(self.local_models_dir)
            if rel.name.lower() == "model.gguf" and len(rel.parts) >= 2:
                return "/".join(rel.parts[:-1])
            return self._build_relative_ref_without_suffix(self.local_models_dir, file_path)

        if file_path.is_relative_to(self.tllama_models_dir):
            rel = file_path.relative_to(self.tllama_models_dir)
            if rel.name.lower() == "model.gguf" and len(rel.parts) >= 2:
                return "/".join(rel.parts[:-1])
            return self._build_relative_ref_without_suffix(self.tllama_models_dir, file_path)

        raise ValueError(f"File path is outside known model repositories: {file_path}")

    def _iter_repository_model_files(self):
        for repo_dir in (self.hf_models_dir, self.local_models_dir, self.tllama_models_dir):
            if not repo_dir.exists():
                continue

            for file_path in sorted(repo_dir.rglob("*.gguf")):
                if file_path.is_file():
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

