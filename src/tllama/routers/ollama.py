import json
import time
from datetime import datetime, timezone
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)


def get_iso_time():
    """Ollama wants specific time format."""
    return datetime.now(timezone.utc).isoformat()


async def ollama_chat_stream(model_id: str):
    """Generator NDJSON stream for /api/chat."""
    chunks = ["This ", "answer ", "is ", "stream ", "via ", "Ollama API."]

    for chunk in chunks:
        data = {
            "model": model_id,
            "created_at": get_iso_time(),
            "message": {"role": "assistant", "content": chunk},
            "done": False
        }
        yield f"{json.dumps(data)}\n"
        await asyncio.sleep(0.05)

    # End chunk signaling finish (done: true)
    final_data = {
        "model": model_id,
        "created_at": get_iso_time(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 1337000000,
        "load_duration": 1337000,
        "prompt_eval_count": 10,
        "eval_count": len(chunks)
    }
    yield f"{json.dumps(final_data)}\n"


@router.get("/version")
async def get_version():
    return {"version": "0.0.0"}  # Return version, that client expect


@router.post("/show")
async def show_model_info(request: dict):
    return {
        "modelfile": "FROM local-model:latest\nTEMPLATE \"{{ .System }}\\nUser: {{ .Prompt }}\\nAssistant: \"",
        "parameters": "stop                           \"<|end_of_text|>\"\nstop                           \"<|eot_id|>\"",
        "template": "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: ",
        "details": {
            "format": "gguf",
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": "Q4_K_M"
        },
        "messages": []  # Important: CLI can expect array of history with messages
    }


@router.get("/tags")
async def list_tags():
    """List models."""
    return {
        "models": [
            {
                "name": "local-model:latest",
                "model": "local-model:latest",
                "modified_at": get_iso_time(),
                "size": 4000000000,
                "digest": "sha256:1234567890abcdef",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest):
    """Chat endpoint."""
    if request.stream:
        return StreamingResponse(
            ollama_chat_stream(request.model),
            media_type="application/x-ndjson"
        )

    # Non-streaming answer
    return {
        "model": request.model,
        "created_at": get_iso_time(),
        "message": {"role": "assistant", "content": "Statická odpověď z Ollama rozhraní."},
        "done": True
    }


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """Older endpoint completion (used by some UIs for quick prompts)."""
    async def generate_stream():
        data = {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": "Answer from /api/generate",
            "done": True
        }
        yield f"{json.dumps(data)}\n"

    if request.stream:
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson"
        )

    return {
        "model": request.model,
        "created_at": get_iso_time(),
        "response": "Answer from /api/generate",
        "done": True
    }


@router.post("/show")
async def show_model_info(request: dict):
    return {
        "modelfile": "FROM local-model:latest",
        "parameters": "",
        "template": "{{ .Prompt }}",
        "details": {
            "format": "gguf",
            "family": "llama",
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    }
