from fastapi import APIRouter, Request
from tllama.schemas.openai import ChatCompletionRequest
from tllama.backend import model_manager
from tllama.native_routes import serve_native
from tllama.schemas.embeddings import OpenAIEmbeddingRequest
from tllama.native_embeddings import embedding_response

router = APIRouter(prefix="/v1", tags=["OpenAI API"])


@router.get("/models")
async def list_models_openai():
    local_models = await model_manager.list_local_models()

    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": m["mtime"],
                "owned_by": "local-ai"
            } for m in local_models
        ]
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await serve_native(model_manager, request, api="openai")


@router.post('/embeddings')
async def embeddings(request: OpenAIEmbeddingRequest, http_request: Request):
    return await embedding_response(http_request, model_manager, request, api='openai')
