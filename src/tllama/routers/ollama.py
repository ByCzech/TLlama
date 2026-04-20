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
    resolve_think_flag,
    estimate_completion_prompt_eval_count,
)
from tllama.helpers.prompt_render import (
    render_chat_prompt_with_explicit_think,
    render_generate_prompt,
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

    explicit_think = resolve_think_flag(request)
    user_stop = normalize_stop(opts.get("stop"))

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    base_gen_params = {
        "max_tokens": normalize_max_tokens_from_options(opts),
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
    }

    if request.format == "json":
        base_gen_params["response_format"] = {"type": "json_object"}

    start_time = time.time_ns()

    # think=false => explicitní render promptu
    if explicit_think is False:
        full_prompt, stop = render_chat_prompt_with_explicit_think(
            llm=llm,
            metadata_info=metadata_info,
            messages=messages,
            think_enabled=explicit_think,
            user_stop=user_stop,
        )

        prompt_eval_count = estimate_completion_prompt_eval_count(llm, full_prompt)

        gen_params = {
            **base_gen_params,
            "stop": stop,
        }

        if request.stream:
            async def chat_stream_generator():
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
            "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
            "eval_count": response.get("usage", {}).get("completion_tokens")
        }

    # default / think=true => původní fungující chat path

    gen_params = {
        **base_gen_params,
        "stop": user_stop
    }

    if request.stream:
        async def chat_stream_generator():
            response_iter = llm.create_chat_completion(
                messages=messages,
                stream=True,
                **gen_params
            )

            for chunk in response_iter:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")

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
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
        "eval_count": response.get("usage", {}).get("completion_tokens")
    }


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    opts = request.options or {}
    try:
        llm = await model_manager.get_model(
            request.model,
            num_ctx=opts.get("num_ctx")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    metadata_info = model_manager.get_model_metadata(request.model) or {}

    if not request.prompt.strip():
        response_data = {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": "",
            "done": True
        }

        if request.stream:
            async def preload_stream():
                yield f"{json.dumps(response_data)}\n"
            return StreamingResponse(preload_stream(), media_type="application/x-ndjson")

        return response_data

    explicit_think = resolve_think_flag(request)
    user_stop = normalize_stop(opts.get("stop"))
    is_raw = bool(getattr(request, "raw", False))
    has_template_override = bool(getattr(request, "template", None))

    # raw=true => bez templatingu
    if is_raw:
        full_prompt = request.prompt
    else:
        full_prompt = request.prompt
        if request.system:
            full_prompt = f"{request.system}\n\n{request.prompt}"

    generation_kwargs = {
        "max_tokens": normalize_max_tokens_from_options(opts),
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
        "echo": False,
    }

    # Ollama supports "json" or a JSON schema object
    request_format = getattr(request, "format", None)
    try:
        if request_format == "json":
            generation_kwargs["grammar"] = LlamaGrammar.from_json_schema(
                json.dumps({"type": "object"})
            )
        elif isinstance(request_format, dict):
            generation_kwargs["grammar"] = LlamaGrammar.from_json_schema(
                json.dumps(request_format)
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format schema: {e}")

    start_time = time.time_ns()

    # raw + explicit think nejde v tomhle backendu implementovat čistě
    if is_raw and explicit_think is not None:
        raise HTTPException(
            status_code=400,
            detail="raw=true cannot be combined with explicit think in this backend, because think control is implemented through prompt templating."
        )

    def split_thinking_and_response(text: str) -> tuple[str, str]:
        thinking_parts = THINK_BLOCK_RE.findall(text)
        thinking = "\n".join(part.strip() for part in thinking_parts if part.strip())
        response_text = THINK_BLOCK_RE.sub("", text).strip()
        return thinking, response_text

    # 1) RAW branch
    if is_raw:
        prompt_eval_count = estimate_completion_prompt_eval_count(llm, full_prompt)

        if request.stream:
            async def generate_stream():
                response_iter = llm.create_completion(
                    prompt=full_prompt,
                    stream=True,
                    stop=user_stop,
                    **generation_kwargs
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
                    'context': []
                })}\n"

            return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

        response = llm.create_completion(
            prompt=full_prompt,
            stream=False,
            stop=user_stop,
            **generation_kwargs
        )
        end_time = time.time_ns()

        return {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": response["choices"][0].get("text", ""),
            "done": True,
            "done_reason": response["choices"][0].get("finish_reason"),
            "total_duration": end_time - start_time,
            "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
            "eval_count": response.get("usage", {}).get("completion_tokens"),
            "context": []
        }

    # 2) Rendered branch:
    #    a) explicit think set
    #    b) or template override present
    should_render = explicit_think is not None or has_template_override

    if should_render:
        rendered_prompt, stop = render_generate_prompt(
            llm=llm,
            metadata_info=metadata_info,
            request=request,
            full_prompt=full_prompt,
            explicit_think=explicit_think,
        )

        prompt_eval_count = estimate_completion_prompt_eval_count(llm, rendered_prompt)

        # explicit think=true / "low|medium|high" -> separate thinking where possible
        if explicit_think not in (None, False):
            if request.stream:
                async def generate_stream():
                    response_iter = llm.create_completion(
                        prompt=rendered_prompt,
                        stream=True,
                        stop=stop,
                        **generation_kwargs
                    )

                    finish_reason = None
                    text_buffer = ""

                    for chunk in response_iter:
                        choice = chunk["choices"][0]
                        text = choice.get("text", "")
                        chunk_finish_reason = choice.get("finish_reason")

                        if chunk_finish_reason is not None:
                            finish_reason = chunk_finish_reason

                        if text:
                            text_buffer += text

                    thinking, response_text = split_thinking_and_response(text_buffer)

                    if thinking:
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'thinking': thinking,
                            'response': '',
                            'done': False
                        })}\n"

                    if response_text:
                        yield f"{json.dumps({
                            'model': request.model,
                            'created_at': get_iso_time(),
                            'response': response_text,
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
                        'context': []
                    })}\n"

                return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

            response = llm.create_completion(
                prompt=rendered_prompt,
                stream=False,
                stop=stop,
                **generation_kwargs
            )
            end_time = time.time_ns()

            full_text = response["choices"][0].get("text", "")
            thinking, response_text = split_thinking_and_response(full_text)

            return {
                "model": request.model,
                "created_at": get_iso_time(),
                "response": response_text,
                "thinking": thinking,
                "done": True,
                "done_reason": response["choices"][0].get("finish_reason"),
                "total_duration": end_time - start_time,
                "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
                "eval_count": response.get("usage", {}).get("completion_tokens"),
                "context": []
            }

        # explicit think=false OR template override without explicit think
        if request.stream:
            async def generate_stream():
                response_iter = llm.create_completion(
                    prompt=rendered_prompt,
                    stream=True,
                    stop=stop,
                    **generation_kwargs
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
                    'context': []
                })}\n"

            return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

        response = llm.create_completion(
            prompt=rendered_prompt,
            stream=False,
            stop=stop,
            **generation_kwargs
        )
        end_time = time.time_ns()

        return {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": response["choices"][0].get("text", ""),
            "done": True,
            "done_reason": response["choices"][0].get("finish_reason"),
            "total_duration": end_time - start_time,
            "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
            "eval_count": response.get("usage", {}).get("completion_tokens"),
            "context": []
        }

    # 3) Default plain completion branch
    prompt_eval_count = estimate_completion_prompt_eval_count(llm, full_prompt)

    if request.stream:
        async def generate_stream():
            response_iter = llm.create_completion(
                prompt=full_prompt,
                stream=True,
                stop=user_stop,
                **generation_kwargs
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
                'context': []
            })}\n"

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    response = llm.create_completion(
        prompt=full_prompt,
        stream=False,
        stop=user_stop,
        **generation_kwargs
    )
    end_time = time.time_ns()

    return {
        "model": request.model,
        "created_at": get_iso_time(),
        "response": response["choices"][0].get("text", ""),
        "done": True,
        "done_reason": response["choices"][0].get("finish_reason"),
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", prompt_eval_count),
        "eval_count": response.get("usage", {}).get("completion_tokens"),
        "context": []
    }


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
