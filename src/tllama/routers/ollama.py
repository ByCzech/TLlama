import json
import time
from datetime import datetime, timezone
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.backend import model_manager

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)


def get_iso_time():
    """Ollama wants specific time format."""
    return datetime.now(timezone.utc).isoformat()


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
    response_iter = model_manager.llm.create_chat_completion(
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        stream=True
    )

    async def generate_ollama():
        for chunk in response_iter:
            # Check existence content in delta (llama-cpp format)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")

            data = {
                "model": request.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": content},
                "done": False
            }
            yield f"{json.dumps(data)}\n"

        # Final chunk for Ollama
        yield f"{json.dumps({'model': request.model, 'done': True})}\n"

    return StreamingResponse(generate_ollama(), media_type="application/x-ndjson")


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
