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
from tllama.helpers.llama_stats import load_llama_with_captured_stats


__all__ = ('model_manager', 'load_backend_config_from_env')


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
        self.models_dir.mkdir(exist_ok=True)

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

    def _normalize_keep_alive(self, keep_alive: str | int | float | None) -> int | None:
        """Normalize Ollama-style keep_alive to seconds.

        Returns:
            int:
                Number of seconds for finite keep-alive values.
            0:
                Immediate unload semantics.
            None:
                Infinite keep-alive.
        """
        if keep_alive is None:
            return 300

        if isinstance(keep_alive, (int, float)):
            if keep_alive < 0:
                return None
            return int(keep_alive)

        value = str(keep_alive).strip().lower()

        if value == "":
            return 300

        try:
            numeric = float(value)
            if numeric < 0:
                return None
            return int(numeric)
        except ValueError:
            pass

        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
        }

        suffix = value[-1]
        if suffix in multipliers:
            try:
                numeric = float(value[:-1])
            except ValueError:
                raise ValueError(f"Invalid keep_alive value: {keep_alive}")

            if numeric < 0:
                return None

            return int(numeric * multipliers[suffix])

        raise ValueError(f"Invalid keep_alive value: {keep_alive}")

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
        keep_alive: str | int | float | None = "5m",
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

            effective_keep_alive = keep_alive
            if effective_keep_alive is None:
                effective_keep_alive = self.config.keep_alive

            requested_n_ctx = self._normalize_num_ctx(effective_num_ctx, default=0)
            keep_alive_seconds = self._normalize_keep_alive(effective_keep_alive)

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

    def _get_model_metadata_sync(self, model_path: str) -> Optional[Dict[str, Any]]:
        temp_llm = None
        try:
            temp_llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
            meta = dict(temp_llm.metadata or {})

            arch = meta.get("general.architecture", "llama")
            params = meta.get("general.parameter_count", 0)
            bits = meta.get("general.quantization_version", "unknown")
            template = meta.get("tokenizer.chat_template", "")

            return {
                "arch": arch,
                "params": params,
                "bits": bits,
                "template": template,
                "metadata_raw": self._filter_metadata_raw_for_cache(meta),
            }
        finally:
            if temp_llm is not None:
                try:
                    del temp_llm
                except Exception:
                    pass
            gc.collect()

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

        effective_timeout = (
            self.config.model_scan_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )

        try:
            metadata = await asyncio.wait_for(
                asyncio.to_thread(self._get_model_metadata_sync, model_info["path"]),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            print(f"DEBUG: Metadata scan timed out for model {model_name}")
            return None
        except Exception as e:
            print(f"DEBUG: Metadata scan failed for model {model_name}: {e}")
            return None

        if metadata is None:
            return None

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
        first = parts[0]

        def _normalized_file_path(base_dir: Path, rel_parts: List[str]) -> Path:
            normalized_parts = list(rel_parts)
            normalized_parts[-1] = self._normalize_pull_filename(normalized_parts[-1])
            return base_dir.joinpath(*normalized_parts)

        if first == "Local":
            rel_parts = parts[1:]
            if not rel_parts:
                raise ValueError("Missing Local model path")

            candidates = [
                _normalized_file_path(self.local_models_dir, rel_parts),
                self.local_models_dir.joinpath(*rel_parts, "model.gguf"),
            ]

            for candidate in candidates:
                if candidate.exists():
                    return candidate

            return candidates[0]

        if first == "TLlama":
            rel_parts = parts[1:]
            if not rel_parts:
                raise ValueError("Missing TLlama model path")

            candidates = [
                _normalized_file_path(self.tllama_models_dir, rel_parts),
                self.tllama_models_dir.joinpath(*rel_parts, "model.gguf"),
            ]

            for candidate in candidates:
                if candidate.exists():
                    return candidate

            return candidates[0]

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
                return f"Local/{'/'.join(rel.parts[:-1])}"
            return f"Local/{self._build_relative_ref_without_suffix(self.local_models_dir, file_path)}"

        if file_path.is_relative_to(self.tllama_models_dir):
            rel = file_path.relative_to(self.tllama_models_dir)
            if rel.name.lower() == "model.gguf" and len(rel.parts) >= 2:
                return f"TLlama/{'/'.join(rel.parts[:-1])}"
            return f"TLlama/{self._build_relative_ref_without_suffix(self.tllama_models_dir, file_path)}"

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
