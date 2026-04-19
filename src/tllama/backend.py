import os
import time
import asyncio

from pathlib import Path
from hashlib import sha256

from llama_cpp import Llama
from typing import Dict


class ModelManager:
    def __init__(self, models_dir: str = "./models"):
        self.models: Dict[str, Llama] = {}
        self._lock = asyncio.Lock()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.active_models = {}

    async def get_model(self, model_name: str) -> Llama:
        async with self._lock:
            model_path = f"./models/{model_name}.gguf"

            if model_name not in self.models:
                print(f"DEBUG: Dynamic loading of model {model_name}...")
                # Load model to (V)RAM
                self.models[model_name] = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    use_mmap=False
                )
            return self.models[model_name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]

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
        except Exception as e:
            return None

    def list_local_models(self):
        """Scan models GGUF dir."""
        model_list = []
        for file in self.models_dir.glob("*.gguf"):
            stats = file.stat()

            hash_sha256 = sha256()
            hash_sha256.update(file.name.encode('utf-8'))
            hash_sha256.update(str(stats.st_size).encode('utf-8'))
            hash_sha256.update(str(stats.st_mtime).encode('utf-8'))

            model_list.append({
                "id": file.stem,        # Name w/o extension
                "filename": file.name,
                "size": stats.st_size,
                "mtime": int(stats.st_mtime),
                "sha256": hash_sha256.hexdigest()
            })
        return model_list


model_manager = ModelManager()
