import os
import time
import asyncio
import gc

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

from llama_cpp import Llama
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone, timedelta

from tllama.helpers.llama_stats import load_llama_with_captured_stats


__all__ = ('model_manager', 'load_backend_config_from_env')

DEFAULT_MODELS_DIR = "/var/lib/tllama/models"

@dataclass(frozen=True)
class BackendConfig:
    models_dir: str = DEFAULT_MODELS_DIR
    default_num_ctx: int = 0
    default_keep_alive: str | int | float | None = "5m"
    max_loaded_models: int = 1


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_backend_config_from_env() -> BackendConfig:
    return BackendConfig(
        models_dir=os.getenv("TLLAMA_MODELS_DIR", DEFAULT_MODELS_DIR),
        default_num_ctx=_env_int("TLLAMA_DEFAULT_NUM_CTX", 0),
        max_loaded_models=_env_int("TLLAMA_MAX_LOADED_MODELS", 1),
        default_keep_alive=os.getenv("TLLAMA_DEFAULT_KEEP_ALIVE", "5m"),
    )


class ModelManager:
    def __init__(self, config: BackendConfig | None = None):
        self.config = config or load_backend_config_from_env()

        self.models: Dict[str, Llama] = {}
        self._lock = asyncio.Lock()

        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.active_models: Dict[str, Dict[str, Any]] = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _future_iso(self, seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

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

    def _build_model_file_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        model_path = self.models_dir / f"{model_name}.gguf"

        if not model_path.exists():
            return None

        stats = model_path.stat()

        hash_sha256 = sha256()
        hash_sha256.update(model_path.name.encode("utf-8"))
        hash_sha256.update(str(stats.st_size).encode("utf-8"))
        hash_sha256.update(str(stats.st_mtime).encode("utf-8"))

        return {
            "id": model_name,
            "filename": model_path.name,
            "path": str(model_path),
            "size": stats.st_size,
            "mtime": int(stats.st_mtime),
            "sha256": hash_sha256.hexdigest(),
        }

    def _load_model_sync(self, model_path: str, requested_n_ctx: int):
        return load_llama_with_captured_stats(
            Llama,
            model_path=model_path,
            n_ctx=requested_n_ctx,
            n_gpu_layers=-1,
            use_mmap=False,
            verbose=True,
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
            model_info = self._build_model_file_info(model_name)
            if not model_info:
                raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")

            model_path = model_info["path"]

            effective_num_ctx = num_ctx
            if effective_num_ctx is None:
                effective_num_ctx = self.config.default_num_ctx

            effective_keep_alive = keep_alive
            if effective_keep_alive is None:
                effective_keep_alive = self.config.default_keep_alive

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

                    # stats from load log
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
                    "cpu_rs_mib": load_stats.get("cpu_rs_mib", 0.0)
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
        llm = self.models.pop(model_name, None)
        if llm is not None:
            del llm

        if model_name in self.active_models:
            del self.active_models[model_name]

        gc.collect()

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    def get_loaded_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.active_models.get(model_name)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return list(self.active_models.values())

    def get_model_metadata(self, model_name: str):
        """
        Get model metadata without loading into memory.
        """
        model_path = str(self.models_dir / f"{model_name}.gguf")

        if not os.path.exists(model_path):
            return None

        try:
            temp_llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
            meta = temp_llm.metadata

            arch = meta.get("general.architecture", "llama")
            params = meta.get("general.parameter_count", 0)
            bits = meta.get("general.quantization_version", "unknown")
            template = meta.get("tokenizer.chat_template", "")

            del temp_llm

            return {
                "arch": arch,
                "params": params,
                "bits": bits,
                "template": template,
                "metadata_raw": meta
            }
        except Exception:
            return None

    def list_local_models(self):
        """Scan models GGUF dir."""
        model_list = []
        for file in self.models_dir.glob("*.gguf"):
            stats = file.stat()

            hash_sha256 = sha256()
            hash_sha256.update(file.name.encode("utf-8"))
            hash_sha256.update(str(stats.st_size).encode("utf-8"))
            hash_sha256.update(str(stats.st_mtime).encode("utf-8"))

            model_list.append({
                "id": file.stem,
                "filename": file.name,
                "size": stats.st_size,
                "mtime": int(stats.st_mtime),
                "sha256": hash_sha256.hexdigest()
            })
        return model_list


model_manager = ModelManager(load_backend_config_from_env())
