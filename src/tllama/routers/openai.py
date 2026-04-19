import json
import time
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.openai import ChatCompletionRequest

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI API"]
)


async def stream_generator(model_id: str):
    """Async generator for Server-Sent Events."""
    chunks = ["Answer", " generated", " by", " chunks", "."]
    for chunk in chunks:
        data = {
            "id": "chat-id",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "local-model", "object": "model", "owned_by": "user"}]
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_generator(request.model),
            media_type="text/event-stream"
        )

    return {
        "id": "chat-id",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": "Static answer."},
            "finish_reason": "stop",
            "index": 0
        }]
    }
