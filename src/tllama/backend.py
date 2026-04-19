import asyncio

from llama_cpp import Llama
from typing import Dict


class ModelManager:
    def __init__(self):
        self.models: Dict[str, Llama] = {}
        self._lock = asyncio.Lock()

    async def get_model(self, model_name: str) -> Llama:
        async with self._lock:
            model_path = f"./models/{model_name}.gguf"

            if model_name not in self.models:
                print(f"DEBUG: Dynamic loading of model {model_name}...")
                # Load model to (V)RAM
                self.models[model_name] = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1
                )
            return self.models[model_name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]


model_manager = ModelManager()
