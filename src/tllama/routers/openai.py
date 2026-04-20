import json
import time
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.openai import ChatCompletionRequest
from tllama.backend import model_manager

from tllama.helpers.common import (
    get_iso_time,
    normalize_stop,
    normalize_max_tokens_from_options,
    resolve_think_flag,
    estimate_completion_prompt_eval_count,
)
from tllama.helpers.prompt_render import (
    render_chat_prompt_with_explicit_think,
    render_generate_prompt,
)
from tllama.helpers.openai_compat import openai_reasoning_effort_to_explicit_think, build_openai_chat_messages

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI API"]
)


@router.get("/models")
async def list_models_openai():
    local_models = model_manager.list_local_models()

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
    llm = await model_manager.get_model(request.model)
    metadata_info = model_manager.get_model_metadata(request.model) or {}

    messages = build_openai_chat_messages(request)
    explicit_think = openai_reasoning_effort_to_explicit_think(request)

    max_tokens = normalize_max_tokens_from_options({
        "num_predict": getattr(request, "max_tokens", None)
    })

    temperature = getattr(request, "temperature", None)
    top_p = getattr(request, "top_p", None)
    stream = bool(getattr(request, "stream", False))
    response_format = getattr(request, "response_format", None)
    user_stop = normalize_stop(getattr(request, "stop", None))

    gen_params = {
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        gen_params["temperature"] = temperature
    if top_p is not None:
        gen_params["top_p"] = top_p
    if response_format is not None:
        gen_params["response_format"] = response_format

    created = int(time.time())
    completion_id = f"chatcmpl-{created}"

    # 1) Explicit reasoning disable -> reuse ollama-style template render
    if explicit_think is False:
        class _ChatReqShim:
            def __init__(self, messages, options):
                self.messages = messages
                self.options = options

        shim_messages = []
        for msg in request.messages:
            shim_messages.append(msg)

        full_prompt, rendered_stop = render_chat_prompt_with_explicit_think(
            llm=llm,
            metadata_info=metadata_info,
            messages=messages,
            think_enabled=explicit_think,
            user_stop=user_stop,
        )

        prompt_eval_count = estimate_completion_prompt_eval_count(llm, full_prompt)

        if stream:
            async def generate():
                response_iter = llm.create_completion(
                    prompt=full_prompt,
                    stream=True,
                    stop=rendered_stop,
                    **gen_params
                )

                finish_reason = None

                for chunk in response_iter:
                    choice = chunk["choices"][0]
                    text = choice.get("text", "")
                    chunk_finish_reason = choice.get("finish_reason")

                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason

                    if text:
                        yield f"data: {json.dumps({
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': text},
                                'finish_reason': None
                            }]
                        })}\n\n"

                yield f"data: {json.dumps({
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': request.model,
                    'choices': [{
                        'index': 0,
                        'delta': {},
                        'finish_reason': finish_reason
                    }]
                })}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        response = llm.create_completion(
            prompt=full_prompt,
            stream=False,
            stop=rendered_stop,
            **gen_params
        )

        text = response["choices"][0].get("text", "")
        finish_reason = response["choices"][0].get("finish_reason")
        usage = response.get("usage", {})

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", prompt_eval_count),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": (
                    (usage.get("prompt_tokens", prompt_eval_count) or 0) +
                    (usage.get("completion_tokens") or 0)
                )
            }
        }

    # 2) Default OpenAI-compatible chat path
    if stream:
        async def generate():
            response_iter = llm.create_chat_completion(
                messages=messages,
                stream=True,
                stop=user_stop,
                **gen_params
            )

            for chunk in response_iter:
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    response = llm.create_chat_completion(
        messages=messages,
        stream=False,
        stop=user_stop,
        **gen_params
    )

    return response
