import json
import time
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from tllama.schemas.openai import ChatCompletionRequest
from tllama.backend import model_manager
from tllama.lib.llama_wrap import create_chat_completion_ex

from tllama.helpers.common import (
    get_iso_time,
    normalize_stop,
    normalize_max_tokens_from_options,
    estimate_completion_prompt_eval_count,
)
from tllama.helpers.prompt_render import (
    render_chat_prompt_with_explicit_think,
    render_generate_prompt,
)
from tllama.helpers.reasoning_split import (
    detect_reasoning_format,
    split_full_text_by_reasoning_format,
    ReasoningStreamSplitter,
)
from tllama.helpers.openai_compat import openai_reasoning_effort_to_explicit_think, build_openai_chat_messages
from tllama.helpers.chat import build_think_kwargs_ex


router = APIRouter(
    prefix="/v1",
    tags=["OpenAI API"]
)


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
    llm = await model_manager.get_model(request.model)
    metadata_info = await model_manager.get_model_metadata(request.model) or {}

    messages = build_openai_chat_messages(request)
    explicit_think = openai_reasoning_effort_to_explicit_think(request)
    reasoning_format = detect_reasoning_format(request.model, metadata_info)
    kwargs_ex = build_think_kwargs_ex(explicit_think)

    max_tokens = normalize_max_tokens_from_options({
        "num_predict": getattr(request, "max_tokens", None)
    })

    temperature = getattr(request, "temperature", None)
    top_p = getattr(request, "top_p", None)
    stream = bool(getattr(request, "stream", False))
    response_format = getattr(request, "response_format", None)
    user_stop = normalize_stop(getattr(request, "stop", None))
    tools = getattr(request, "tools", None)
    tool_choice = getattr(request, "tool_choice", None)

    gen_params = {
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        gen_params["temperature"] = temperature
    if top_p is not None:
        gen_params["top_p"] = top_p
    if response_format is not None:
        gen_params["response_format"] = response_format
    if user_stop:
        gen_params["stop"] = user_stop
    if tools is not None:
        gen_params["tools"] = tools
    if tool_choice is not None:
        gen_params["tool_choice"] = tool_choice

    created = int(time.time())
    completion_id = f"chatcmpl-{created}"

    if stream:
        def generate():
            response_iter = create_chat_completion_ex(
                llm,
                messages=messages,
                stream=True,
                **gen_params,
                **kwargs_ex
            )

            finish_reason = None
            role_sent = False
            splitter = ReasoningStreamSplitter(reasoning_format, think_value=explicit_think)

            for chunk in response_iter:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                chunk_finish_reason = choice.get("finish_reason")

                if chunk_finish_reason is not None:
                    finish_reason = chunk_finish_reason

                if not role_sent:
                    yield f"data: {json.dumps({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'role': 'assistant'},
                            'finish_reason': None
                        }]
                    })}\n\n"
                    role_sent = True

                if delta.get("tool_calls") is not None:
                    yield f"data: {json.dumps({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'tool_calls': delta['tool_calls']},
                            'finish_reason': None
                        }]
                    })}\n\n"

                content = delta.get("content", "")
                for kind, piece in splitter.push(content):
                    if not piece:
                        continue

                    yield f"data: {json.dumps({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': piece},
                            'finish_reason': None
                        }]
                    })}\n\n"

            for kind, piece in splitter.finish():
                if not piece:
                    continue

                yield f"data: {json.dumps({
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': request.model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': piece},
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

    response = create_chat_completion_ex(
        llm,
        messages=messages,
        stream=False,
        **gen_params,
        **kwargs_ex
    )

    choice = response["choices"][0]
    choice_message = choice.get("message", {}) or {}
    full_content = choice_message.get("content", "") or ""

    _thinking_text, response_text = split_full_text_by_reasoning_format(
        full_content,
        reasoning_format,
        think_value=explicit_think,
    )
    content_text = response_text if response_text else _thinking_text

    message = {
        "role": "assistant",
        "content": content_text,
    }

    if choice_message.get("tool_calls") is not None:
        message["tool_calls"] = choice_message["tool_calls"]

    usage = response.get("usage", {})

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": choice.get("finish_reason"),
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": (
                (usage.get("prompt_tokens") or 0) +
                (usage.get("completion_tokens") or 0)
            ),
        },
    }
