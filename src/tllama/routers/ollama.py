import logging
import json
import time

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import LlamaGrammar
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.backend import model_manager

from tllama.helpers.common import (
    get_iso_time,
    normalize_stop,
    normalize_max_tokens_from_options,
    normalize_message_content,
    estimate_completion_prompt_eval_count,
    build_completion_format_kwargs
)
from tllama.helpers.prompt_render import (
    render_chat_prompt_with_explicit_think,
    render_generate_prompt,
)
from tllama.helpers.reasoning_stream import (
    detect_reasoning_format,
    ReasoningStreamSplitter,
    split_full_text_by_reasoning_format,
)

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)


@router.get("/version")
async def get_version():
    return {"version": "0.0.0"}  # Return version, that client expect


@router.get("/tags")
async def list_models_ollama():
    local_models = model_manager.list_local_models()

    formatted_models = []
    for m in local_models:
        metadata_info = model_manager.get_model_metadata(m['id'])
        if not metadata_info:
            continue

        p_size = "unknown"
        if isinstance(metadata_info["params"], (int, str)) and int(metadata_info["params"]) > 0:
            p_size = f"{round(int(metadata_info['params']) / 1e9)}b"

        formatted_models.append({
            "name": f"{m['id']}",
            "model": f"{m['id']}",
            "modified_at": datetime.fromtimestamp(m["mtime"], timezone.utc).isoformat(),
            "size": m["size"],
            "digest": f"{m['sha256']}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": metadata_info["arch"],
                "families": [metadata_info["arch"]],
                "parameter_size": p_size,
                "quantization_level": metadata_info["bits"]
            }
        })

    return {"models": formatted_models}


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest):
    opts = request.options or {}
    llm = await model_manager.get_model(
        request.model,
        num_ctx=opts.get("num_ctx")
    )
    metadata_info = model_manager.get_model_metadata(request.model) or {}

    explicit_think = None
    user_stop = normalize_stop(opts.get("stop"))
    request_format = getattr(request, "format", None)
    has_structured_format = request_format is not None

    if has_structured_format and explicit_think not in (None, False):
        raise HTTPException(
            status_code=400,
            detail="format cannot be combined with think=true in /api/chat. Structured output uses completion path with thinking disabled."
        )

    force_completion_path = (explicit_think is False) or has_structured_format

    messages = [
        {
            "role": m.role,
            "content": normalize_message_content(m.content),
            "images": m.images,
            "thinking": m.thinking,
            "tool_calls": m.tool_calls,
            "tool_name": m.tool_name,
            "tool_call_id": m.tool_call_id,
        }
        for m in request.messages
    ]

    base_gen_params = {
        "max_tokens": normalize_max_tokens_from_options(opts),
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
    }

    start_time = time.time_ns()

    if force_completion_path:
        full_prompt, stop = render_chat_prompt_with_explicit_think(
            llm=llm,
            metadata_info=metadata_info,
            messages=messages,
            think_enabled=False if has_structured_format else explicit_think,
            user_stop=user_stop,
        )

        prompt_eval_count = estimate_completion_prompt_eval_count(llm, full_prompt)

        gen_params = {
            **base_gen_params,
            "stop": stop,
        }

        try:
            gen_params.update(build_completion_format_kwargs(request_format))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid format schema: {e}")

        if request.stream:
            def chat_stream_generator():
                response_iter = llm.create_completion(
                    prompt=full_prompt,
                    stream=True,
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
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': datetime.now(timezone.utc).isoformat(),
                            'message': {'role': 'assistant', 'content': text},
                            'done': False
                        })}\n"

                end_time = time.time_ns()
                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': end_time - start_time,
                    'load_duration': 0,
                    'prompt_eval_count': prompt_eval_count,
                    'eval_count': None
                })}\n"

            return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")

        response = llm.create_completion(
            prompt=full_prompt,
            stream=False,
            **gen_params
        )
        end_time = time.time_ns()

        return {
            "model": request.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": response["choices"][0].get("text", "")
            },
            "done": True,
            "done_reason": response["choices"][0].get("finish_reason"),
            "total_duration": end_time - start_time,
            "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
            "eval_count": response.get("usage", {}).get("completion_tokens")
        }

    gen_params = {
        **base_gen_params,
        "stop": user_stop
    }

    if request.stream:
        def chat_stream_generator():
            response_iter = llm.create_chat_completion(
                messages=messages,
                stream=True,
                **gen_params
            )

            finish_reason = None

            for chunk in response_iter:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                chunk_finish_reason = choice.get("finish_reason")

                if chunk_finish_reason is not None:
                    finish_reason = chunk_finish_reason

                if content:
                    yield f"{json.dumps({
                        'model': request.model,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'message': {'role': 'assistant', 'content': content},
                        'done': False
                    })}\n"

            end_time = time.time_ns()
            yield f"{json.dumps({
                'model': request.model,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'done': True,
                'done_reason': finish_reason,
                'total_duration': end_time - start_time,
                'load_duration': 0,
                'prompt_eval_count': None,
                'eval_count': None
            })}\n"

        return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")

    response = llm.create_chat_completion(
        messages=messages,
        stream=False,
        **gen_params
    )
    end_time = time.time_ns()

    return {
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": response["choices"][0]["message"],
        "done": True,
        "done_reason": response["choices"][0].get("finish_reason"),
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
        "eval_count": response.get("usage", {}).get("completion_tokens")
    }


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """Handle Ollama-compatible /generate requests using the llama.cpp completion API.

    Returns:
        dict | StreamingResponse:
            A standard JSON response for non-stream requests, or NDJSON stream output
            for stream requests.

    Raises:
        HTTPException:
            Raised when model loading fails, the format schema is invalid, template
            rendering fails, or an unsupported input combination is used.
    """
    opts = request.options or {}

    try:
        keep_alive_seconds = model_manager._normalize_keep_alive(request.keep_alive)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid keep_alive value: {str(e)}")

    request_format = getattr(request, "format", None)
    is_raw = request.raw is True
    user_stop = normalize_stop(opts.get("stop"))
    suffix_text = request.suffix or None

    if is_raw and (
        request.template is not None
        or request.system is not None
        or request.context is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="raw mode does not support template, system, or context"
        )

    if not request.prompt:
        if keep_alive_seconds == 0:
            model_manager.unload_model(request.model)
            return {
                "model": request.model,
                "created_at": get_iso_time(),
                "response": "",
                "done": True,
                "done_reason": "unload"
            }

        try:
            await model_manager.get_model(
                request.model,
                num_ctx=opts.get("num_ctx"),
                keep_alive=request.keep_alive,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")

        return {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": "",
            "done": True,
            "done_reason": "load"
        }

    try:
        llm = await model_manager.get_model(
            request.model,
            num_ctx=opts.get("num_ctx"),
            keep_alive=request.keep_alive,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")

    generation_kwargs = {
        "max_tokens": normalize_max_tokens_from_options(opts),
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
        "echo": False,
    }

    try:
        generation_kwargs.update(build_completion_format_kwargs(request_format))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format schema: {e}")

    if is_raw:
        prompt_for_completion = request.prompt or ""
        stop_tokens = user_stop
    else:
        prompt_for_completion, stop_tokens = render_generate_prompt(
            llm=llm,
            metadata_info=model_manager.get_model_metadata(request.model) or {},
            request=request,
        )

    prompt_eval_count = estimate_completion_prompt_eval_count(llm, prompt_for_completion)

    start_time = time.time_ns()

    if request.stream:
        def generate_stream():
            response_iter = llm.create_completion(
                prompt=prompt_for_completion,
                suffix=suffix_text,
                stream=True,
                stop=stop_tokens,
                **generation_kwargs
            )

            finish_reason = None
            eval_count = None

            try:
                for chunk in response_iter:
                    choice = chunk["choices"][0]
                    text = choice.get("text", "")
                    chunk_finish_reason = choice.get("finish_reason")

                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason

                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        eval_count = usage.get("completion_tokens")

                    if text:
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'response': text,
                            'done': False
                        })}\n"

                end_time = time.time_ns()

                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': get_iso_time(),
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': end_time - start_time,
                    'prompt_eval_count': prompt_eval_count,
                    'eval_count': eval_count,
                    'context': []
                })}\n"
            finally:
                if keep_alive_seconds == 0:
                    model_manager.unload_model(request.model)

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    try:
        response = llm.create_completion(
            prompt=prompt_for_completion,
            suffix=suffix_text,
            stream=False,
            stop=stop_tokens,
            **generation_kwargs
        )
    finally:
        if keep_alive_seconds == 0:
            model_manager.unload_model(request.model)

    end_time = time.time_ns()

    response_text = response["choices"][0].get("text", "")
    done_reason = response["choices"][0].get("finish_reason")

    result = {
        "model": request.model,
        "created_at": get_iso_time(),
        "response": response_text,
        "done": True,
        "done_reason": done_reason,
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
        "eval_count": response.get("usage", {}).get("completion_tokens"),
        "context": []
    }

    return result


@router.post("/show")
async def show_model_info(request: dict):
    model_name = request.get("name", "") or request.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    metadata_info = model_manager.get_model_metadata(model_name)
    if not metadata_info:
        raise HTTPException(status_code=404, detail="Model doesn't exist")

    template = metadata_info.get("template", None) or "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: "

    return {
        "modelfile": f"FROM {model_name}.gguf\nTEMPLATE \"\"\"{template}\"\"\"",
        "parameters": "stop                           \"<|end_of_text|>\"",
        "template": template,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": metadata_info["arch"],
            "families": [metadata_info["arch"]],
            "parameter_size": f"{round(int(metadata_info['params']) / 1e9)}B" if metadata_info["params"] else "unknown",
            "quantization_level": metadata_info["bits"]
        }
    }


@router.get("/ps")
async def list_running_models():
    loaded_models = model_manager.list_loaded_models()

    formatted = []
    for m in loaded_models:
        metadata_info = model_manager.get_model_metadata(m["id"]) or {}

        p_size = "unknown"
        params = metadata_info.get("params", 0)
        if isinstance(params, (int, str)):
            try:
                if int(params) > 0:
                    p_size = f"{round(int(params) / 1e9)}b"
            except Exception:
                pass

        formatted.append({
            "name": m["model"],
            "model": m["model"],
            "size": m["size"],
            "digest": m["sha256"],
            "context_length": m["n_ctx"],
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": metadata_info.get("arch", "unknown"),
                "families": [metadata_info.get("arch", "unknown")],
                "parameter_size": p_size,
                "quantization_level": metadata_info.get("bits", "unknown"),
            },
            "expires_at": m["expires_at"],
            "size_vram": int(m.get("gpu_model_mib", 0.0) * 1024**2)
        })

    return {"models": formatted}
