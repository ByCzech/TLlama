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
from tllama.lib.llama_wrap import create_chat_completion_ex

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
from tllama.helpers.reasoning_split import (
    detect_reasoning_format,
    ReasoningStreamSplitter,
    split_full_text_by_reasoning_format,
)
from tllama.helpers.chat import (
    normalize_chat_messages,
    build_chat_kwargs_ex,
    build_chat_response_format_kwargs
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
    local_models = await model_manager.list_local_models()

    formatted_models = []
    for m in local_models:
        metadata_info = await model_manager.get_model_metadata(m['id'])
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
    """Handle Ollama-compatible /chat requests via create_chat_completion_ex()."""
    opts = request.options or {}

    try:
        keep_alive_seconds = model_manager._normalize_keep_alive(request.keep_alive)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid keep_alive value: {str(e)}")

    if not request.messages:
        if keep_alive_seconds == 0:
            model_manager.unload_model(request.model)
            return {
                "model": request.model,
                "created_at": get_iso_time(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "unload",
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
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "load",
        }

    try:
        llm = await model_manager.get_model(
            request.model,
            num_ctx=opts.get("num_ctx"),
            keep_alive=request.keep_alive,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")

    metadata_info = await model_manager.get_model_metadata(request.model) or {}
    reasoning_format = detect_reasoning_format(request.model, metadata_info)
    messages = normalize_chat_messages(request.messages)
    kwargs_ex = build_chat_kwargs_ex(request)

    gen_params = {
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
        "top_k": opts.get("top_k", 40),
        "min_p": opts.get("min_p", 0.05),
        "typical_p": opts.get("typical_p", 1.0),
        "presence_penalty": opts.get("presence_penalty", 0.0),
        "frequency_penalty": opts.get("frequency_penalty", 0.0),
        "repeat_penalty": opts.get("repeat_penalty", 1.0),
        "tfs_z": opts.get("tfs_z", 1.0),
        "mirostat_mode": opts.get("mirostat", 0),
        "mirostat_tau": opts.get("mirostat_tau", 5.0),
        "mirostat_eta": opts.get("mirostat_eta", 0.1),
        "seed": opts.get("seed"),
        "max_tokens": normalize_max_tokens_from_options(opts),
    }

    user_stop = normalize_stop(opts.get("stop"))
    if user_stop:
        gen_params["stop"] = user_stop

    if request.tools:
        gen_params["tools"] = request.tools

    try:
        gen_params.update(build_chat_response_format_kwargs(getattr(request, "format", None)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format schema: {e}")

    start_time = time.time_ns()

    if request.stream:
        def chat_stream_generator():
            finish_reason = None
            eval_count = None
            splitter = ReasoningStreamSplitter(reasoning_format, think_value=request.think)

            try:
                response_iter = create_chat_completion_ex(
                    llm,
                    messages=messages,
                    stream=True,
                    **gen_params,
                    **kwargs_ex
                )

                for chunk in response_iter:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    chunk_finish_reason = choice.get("finish_reason")

                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason

                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        eval_count = usage.get("completion_tokens")

                    if delta.get("tool_calls") is not None:
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'message': {
                                'role': 'assistant',
                                'tool_calls': delta['tool_calls']
                            },
                            'done': False
                        })}\n"

                    content = delta.get("content", "")
                    for kind, piece in splitter.push(content):
                        if kind == "thinking":
                            payload = {
                                "model": request.model,
                                "created_at": get_iso_time(),
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "thinking": piece,
                                },
                                "done": False,
                            }
                        else:
                            payload = {
                                "model": request.model,
                                "created_at": get_iso_time(),
                                "message": {
                                    "role": "assistant",
                                    "content": piece,
                                },
                                "done": False,
                            }

                        yield f"{json.dumps(payload)}\n"

                for kind, piece in splitter.finish():
                    if kind == "thinking":
                        payload = {
                            "model": request.model,
                            "created_at": get_iso_time(),
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": piece,
                            },
                            "done": False,
                        }
                    else:
                        payload = {
                            "model": request.model,
                            "created_at": get_iso_time(),
                            "message": {
                                "role": "assistant",
                                "content": piece,
                            },
                            "done": False,
                        }

                    yield f"{json.dumps(payload)}\n"

                end_time = time.time_ns()
                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': get_iso_time(),
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': end_time - start_time,
                    'prompt_eval_count': None,
                    'eval_count': eval_count,
                })}\n"

            finally:
                if keep_alive_seconds == 0:
                    model_manager.unload_model(request.model)

        return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")

    try:
        response = create_chat_completion_ex(
            llm,
            messages=messages,
            stream=False,
            **gen_params,
            **kwargs_ex
        )
    finally:
        if keep_alive_seconds == 0:
            model_manager.unload_model(request.model)

    end_time = time.time_ns()

    choice = response["choices"][0]
    choice_message = choice.get("message", {}) or {}
    full_content = choice_message.get("content", "") or ""

    thinking_text, response_text = split_full_text_by_reasoning_format(
        full_content,
        reasoning_format,
        think_value=request.think,
    )

    message = {
        "role": "assistant",
        "content": response_text,
    }

    if thinking_text:
        message["thinking"] = thinking_text

    if choice_message.get("tool_calls") is not None:
        message["tool_calls"] = choice_message["tool_calls"]

    return {
        "model": request.model,
        "created_at": get_iso_time(),
        "message": message,
        "done": True,
        "done_reason": choice.get("finish_reason"),
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
        "eval_count": response.get("usage", {}).get("completion_tokens"),
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

    reasoning_format = detect_reasoning_format(request.model, model_manager.get_model_metadata(request.model) or {})

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
            splitter = ReasoningStreamSplitter(reasoning_format, think_value=request.think)

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

                    for kind, piece in splitter.push(text):
                        if kind == "thinking":
                            yield f"{json.dumps({
                                'model': request.model,
                                'created_at': get_iso_time(),
                                'response': '',
                                'thinking': piece,
                                'done': False
                            })}\n"
                        else:
                            yield f"{json.dumps({
                                'model': request.model,
                                'created_at': get_iso_time(),
                                'response': piece,
                                'done': False
                            })}\n"

                for kind, piece in splitter.finish():
                    if kind == "thinking":
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'response': '',
                            'thinking': piece,
                            'done': False
                        })}\n"
                    else:
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'response': piece,
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

    full_text = response["choices"][0].get("text", "")
    thinking_text, response_text = split_full_text_by_reasoning_format(
        full_text,
        reasoning_format,
        think_value=request.think,
    )
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

    if thinking_text:
        result["thinking"] = thinking_text

    return result


@router.post("/show")
async def show_model_info(request: dict):
    model_name = request.get("name", "") or request.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    metadata_info = await model_manager.get_model_metadata(model_name)
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
        metadata_info = await model_manager.get_model_metadata(m["id"]) or {}

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
            "size": m["size"],
            "size_vram": m["size_vram"]
        })

    return {"models": formatted}
