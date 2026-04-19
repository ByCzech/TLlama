import json
import time
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.openai import ChatCompletionRequest
from tllama.backend import model_manager

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI API"]
)


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "local-model", "object": "model", "owned_by": "user"}]
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Call llama-cpp backend
    response_iter = model_manager.llm.create_chat_completion(
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        stream=True,
        temperature=request.temperature
    )

    async def generate():
        for chunk in response_iter:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
