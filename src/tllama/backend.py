from llama_cpp import Llama
from contextlib import asynccontextmanager
from fastapi import FastAPI


class ModelManager:
    def __init__(self):
        self.llm: Llama = None

    def load_model(self, path: str):
        self.llm = Llama(
            model_path=path,
            n_ctx=4096,         # Context window
            n_gpu_layers=-1,    # Load all layers to GPU if available
            use_mmap=False      # Disable mmap or it goes to CPU on Vulkan API
        )


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialisation on start
    model_manager.load_model("./models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")

    yield

    # Cleanup on exit
    del model_manager.llm
