import os
import time
import asyncio

from pathlib import Path
from hashlib import sha256

from llama_cpp import Llama
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone, timedelta

from tllama.helpers.llama_stats import load_llama_with_captured_stats


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _future_iso(minutes: int = 5):
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _normalize_num_ctx(value, default: int = 2048) -> int:
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class ModelManager:
    def __init__(self, models_dir: str = "./models"):
        self.models: Dict[str, Llama] = {}
        self._lock = asyncio.Lock()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.active_models: Dict[str, Dict[str, Any]] = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

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

    async def get_model(self, model_name: str, num_ctx: int | None = None) -> Llama:
        async with self._lock:
            model_info = self._build_model_file_info(model_name)
            if not model_info:
                raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")

            model_path = model_info["path"]
            requested_n_ctx = _normalize_num_ctx(num_ctx, default=0)

            # už je loadnutý?
            if model_name in self.models:
                current_n_ctx = self.active_models.get(model_name, {}).get("n_ctx")

                # pokud klient výslovně chce jiný num_ctx, reload
                if current_n_ctx != requested_n_ctx:
                    del self.models[model_name]
                    if model_name in self.active_models:
                        del self.active_models[model_name]

            # po případném unloadu znovu zkontroluj
            if model_name not in self.models:
                print(f"DEBUG: Dynamic loading of model {model_name} with n_ctx={requested_n_ctx}...")

                llm, load_stats, load_log = load_llama_with_captured_stats(
                    Llama,
                    model_path=model_path,
                    n_ctx=requested_n_ctx,
                    n_gpu_layers=-1,
                    use_mmap=False,
                    verbose=True,
                )

                actual_n_ctx = llm.n_ctx()

                self.models[model_name] = llm
                self.active_models[model_name] = {
                    "id": model_name,
                    "model": model_name,
                    "filename": model_info["filename"],
                    "path": model_info["path"],
                    "size": model_info["size"],
                    "mtime": model_info["mtime"],
                    "sha256": model_info["sha256"],
                    "loaded_at": _now_iso(),
                    "last_used_at": _now_iso(),
                    "expires_at": _future_iso(5),
                    "n_ctx": actual_n_ctx,
                    "n_gpu_layers": -1,
                    "use_mmap": False,

                    # stats from log
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
                self.active_models[model_name]["last_used_at"] = _now_iso()
                self.active_models[model_name]["expires_at"] = _future_iso(5)

            return self.models[model_name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.active_models:
            del self.active_models[model_name]

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


model_manager = ModelManager()
