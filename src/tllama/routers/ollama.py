import asyncio
import json
import time

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
# Only iterate_in_threadpool is taken from starlette. Its run_in_threadpool
# goes through anyio, whose worker held on to the last callable it ran until
# 4.9, and a reference to a multi-gigabyte model does not belong in a queue
# whose housekeeping we do not control. asyncio.to_thread uses the standard
# library ThreadPoolExecutor, which drops its work item as soon as it has run.
from starlette.concurrency import iterate_in_threadpool
from llama_cpp import LlamaGrammar
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.backend import model_manager, HF_LOOKUP_MISSING, PullProgress
from tllama._ext import counted_for_request, reset_eval_counters
from tllama.lib.llama_wrap import create_chat_completion_ex

from tllama.errors import ollama_stream_error_line
from tllama.helpers.model_toml import TomlModelError

from tllama.helpers.common import (
    get_iso_time,
    never_expires_at,
    normalize_stop,
    normalize_message_content,
    estimate_completion_prompt_eval_count,
    build_completion_format_kwargs,
    build_sampling_kwargs,
)

from tllama.helpers.prompt_render import (
    render_generate_prompt,
)
from tllama.helpers.reasoning_split import (
    detect_reasoning_format,
    ReasoningStreamSplitter,
    split_full_text_by_reasoning_format,
)
from tllama.helpers.chat import (
    normalize_chat_messages,
    apply_default_system_prompt,
    build_chat_kwargs_ex,
    build_chat_response_format_kwargs
)

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)

# How often the pull stream reports download progress to the client.
# Ollama updates its progress bar far more often than this; a request every
# few hundred milliseconds is plenty for a status line and keeps the ndjson
# stream from being dominated by progress noise.
_PULL_PROGRESS_INTERVAL_SECONDS = 0.5


@router.get("/version")
async def get_version():
    return {"version": "0.0.0"}  # Return version, that client expect


@router.get("/tags")
async def list_models_ollama():
    local_models = await model_manager.list_local_models_with_metadata()

    formatted_models = []
    for m in local_models:
        family = m.get("arch", "unknown")

        formatted_models.append({
            "name": f"{m['id']}",
            "model": f"{m['id']}",
            "modified_at": datetime.fromtimestamp(m["mtime"], timezone.utc).isoformat(),
            "size": m["size"],
            "digest": m.get("digest", ""),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": family,
                "families": [family],
                "parameter_size": m.get("parameter_size", "unknown"),
                "quantization_level": m.get("bits", "unknown"),
            }
        })

    return {"models": formatted_models}


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest):
    """Handle Ollama-compatible /chat requests via create_chat_completion_ex()."""
    opts = request.options or {}

    try:
        keep_alive_seconds = model_manager.resolve_keep_alive(request.keep_alive)
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
        except TomlModelError:
            # Not the request's fault and not a load failure: the virtual
            # model's own definition is broken. Reported as 500 by the
            # registered handler instead of being flattened into a 400 that
            # blames the client.
            raise
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
    except TomlModelError:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")

    metadata_info = await model_manager.get_model_metadata(request.model) or {}
    reasoning_format = detect_reasoning_format(request.model, metadata_info)
    messages = apply_default_system_prompt(normalize_chat_messages(request.messages), metadata_info)
    kwargs_ex = build_chat_kwargs_ex(request)

    gen_params = build_sampling_kwargs(opts, metadata_info)

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
            counters_reset = reset_eval_counters(llm)
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

                # llama-cpp-python reports usage only on a non-streaming
                # response, so ask llama.cpp for its own counters instead.
                #
                # n_p_eval counts tokens actually evaluated, which in principle
                # could fall below the prompt length once a prefix is reused.
                # It does not today: llama.cpp cannot remove part of a KV cache
                # entry, so it re-evaluates the whole prompt and the two agree.
                counted_prompt, counted_eval = counted_for_request(counters_reset, llm)

                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': get_iso_time(),
                    # Required, not optional, in the official client's
                    # ChatResponse -- a final line without it fails
                    # validation before generation stats are ever seen.
                    # Real Ollama sends the same empty-content message here.
                    'message': {'role': 'assistant', 'content': ''},
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': end_time - start_time,
                    'prompt_eval_count': counted_prompt,
                    'eval_count': eval_count if eval_count is not None else counted_eval,
                })}\n"

            except Exception as e:
                # The status code went out with the first chunk, so an error
                # line in the body is the only way left to report a failure.
                yield ollama_stream_error_line(str(e))

            finally:
                if keep_alive_seconds == 0:
                    model_manager.unload_model(request.model)

        async def guarded_chat_stream():
            # The slot is held for the whole stream, not just its creation.
            async with model_manager.acquire_inference_slot(request.model):
                async for chunk in iterate_in_threadpool(chat_stream_generator()):
                    yield chunk

        return StreamingResponse(guarded_chat_stream(), media_type="application/x-ndjson")

    try:
        async with model_manager.acquire_inference_slot(request.model):
            response = await asyncio.to_thread(
                create_chat_completion_ex,
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
        keep_alive_seconds = model_manager.resolve_keep_alive(request.keep_alive)
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
        except TomlModelError:
            # Not the request's fault and not a load failure: the virtual
            # model's own definition is broken. Reported as 500 by the
            # registered handler instead of being flattened into a 400 that
            # blames the client.
            raise
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
    except TomlModelError:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")

    metadata_info = await model_manager.get_model_metadata(request.model) or {}

    generation_kwargs = build_sampling_kwargs(opts, metadata_info)
    generation_kwargs.pop("stop", None)  # stop is computed below, alongside eos_token
    generation_kwargs["echo"] = False

    try:
        generation_kwargs.update(build_completion_format_kwargs(request_format))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format schema: {e}")

    if is_raw:
        prompt_for_completion = request.prompt or ""
        stop_tokens = user_stop or list(metadata_info.get("stop_defaults") or [])
    else:
        prompt_for_completion, stop_tokens = render_generate_prompt(
            llm=llm,
            metadata_info=metadata_info,
            request=request,
        )

    reasoning_format = detect_reasoning_format(request.model, metadata_info)

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
            counters_reset = reset_eval_counters(llm)
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

                # llama-cpp-python reports usage only on a non-streaming
                # response, so ask llama.cpp for its own counters instead.
                #
                # n_p_eval counts tokens actually evaluated, which in principle
                # could fall below the prompt length once a prefix is reused.
                # It does not today: llama.cpp cannot remove part of a KV cache
                # entry, so it re-evaluates the whole prompt and the two agree.
                counted_prompt, counted_eval = counted_for_request(counters_reset, llm)

                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': get_iso_time(),
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': end_time - start_time,
                    'prompt_eval_count': counted_prompt if counted_prompt is not None else prompt_eval_count,
                    'eval_count': eval_count if eval_count is not None else counted_eval,
                    'context': []
                })}\n"
            except Exception as e:
                # The status code went out with the first chunk, so an error
                # line in the body is the only way left to report a failure.
                yield ollama_stream_error_line(str(e))

            finally:
                if keep_alive_seconds == 0:
                    model_manager.unload_model(request.model)

        async def guarded_generate_stream():
            async with model_manager.acquire_inference_slot(request.model):
                async for chunk in iterate_in_threadpool(generate_stream()):
                    yield chunk

        return StreamingResponse(guarded_generate_stream(), media_type="application/x-ndjson")

    try:
        async with model_manager.acquire_inference_slot(request.model):
            response = await asyncio.to_thread(
                llm.create_completion,
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

    metadata_info = await model_manager.get_model_metadata(model_name) or {}

    if not metadata_info:
        model_info = model_manager._build_model_file_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model doesn't exist")

    template = metadata_info.get("template") or "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: "

    family = metadata_info.get("arch", "unknown")

    # Real Ollama uses this list to tell a client what a model can do, and
    # clients built against it (any bot using the ollama Python client, for
    # instance) act on it directly -- an absent or empty list reads as "this
    # model can't do any of that", including things TLlama actually
    # supports. "completion" always holds; "thinking" is reported only when
    # the model's own template is one detect_reasoning_format recognizes as
    # having a thinking format, so a plain non-reasoning model is not
    # claimed to think just because it exists.
    #
    # "tools" is deliberately left out: TLlama can only force a tool call
    # today, not detect one from the template the way real Ollama's chat.*
    # layer does, so claiming the capability would overstate what actually
    # works. "vision" is left out because there is no vision support yet.
    capabilities = ["completion"]
    if detect_reasoning_format(model_name, metadata_info) != "none":
        capabilities.append("thinking")

    return {
        "modelfile": f'FROM {model_name}\nTEMPLATE """{template}"""',
        "parameters": "stop                           \"<|end_of_text|>\"",
        "template": template,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": metadata_info.get("parameter_size", "unknown"),
            "quantization_level": metadata_info.get("bits", "unknown"),
        },
        # The official Ollama client's ShowResponse declares this field
        # Optional but without a pydantic default, which in practice makes
        # it required on the wire: a response missing the key fails
        # validation before the client ever looks at what's in it. Real
        # Ollama fills it with the model's raw GGUF key/value metadata;
        # metadata_raw is the same style of data, only over a narrower,
        # whitelisted set of keys, since that is what TLlama currently
        # reads from a GGUF header.
        "model_info": metadata_info.get("metadata_raw") or {},
        "capabilities": capabilities,
    }


@router.get("/ps")
async def list_running_models():
    loaded_models = model_manager.list_loaded_models()

    formatted = []
    for m in loaded_models:
        # A model that is resident right now stays in this listing even if
        # its .toml has since been edited into something unparseable: it is
        # still loaded and still holding memory, so omitting it would
        # misreport the state of the machine. Its metadata degrades to
        # unknown instead.
        metadata_info = await model_manager.get_model_metadata_best_effort(m["id"]) or {}
        digest_info = await model_manager.get_model_digest(m["path"]) or {}

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
            "digest": digest_info.get("content_sha256", ""),
            "context_length": m["n_ctx"],
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": metadata_info.get("arch", "unknown"),
                "families": [metadata_info.get("arch", "unknown")],
                "parameter_size": p_size,
                "quantization_level": metadata_info.get("bits", "unknown"),
            },
            "expires_at": m["expires_at"] or never_expires_at(),
            "size_vram": m["size_vram"]
        })

    loaded_names = {m["model"] for m in loaded_models}
    for model_name in model_manager.list_loading_models():
        if model_name in loaded_names:
            # Being reloaded with a different context size, and already
            # represented above by its (soon to be replaced) loaded entry.
            continue

        model_info = model_manager.build_model_file_info_best_effort(model_name)
        if not model_info:
            # Gone from disk, or its .toml turned unparseable, between
            # starting the load and this request.
            continue

        metadata_info = await model_manager.get_model_metadata_best_effort(model_name) or {}
        digest_info = await model_manager.get_model_digest(model_info["path"]) or {}

        p_size = "unknown"
        params = metadata_info.get("params", 0)
        if isinstance(params, (int, str)):
            try:
                if int(params) > 0:
                    p_size = f"{round(int(params) / 1e9)}b"
            except Exception:
                pass

        formatted.append({
            "name": model_name,
            "model": model_name,
            "size": model_info["size"],
            "digest": digest_info.get("content_sha256", ""),
            "context_length": 0,
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": metadata_info.get("arch", "unknown"),
                "families": [metadata_info.get("arch", "unknown")],
                "parameter_size": p_size,
                "quantization_level": metadata_info.get("bits", "unknown"),
            },
            "expires_at": never_expires_at(),
            "size_vram": 0,
        })

    return {"models": formatted}


@router.post("/pull")
async def pull_model_ollama(request: Request):
    payload = await request.json()

    model_ref = (payload.get("model") or payload.get("name") or "").strip()
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    stream = payload.get("stream", True)

    if not model_ref:
        raise HTTPException(status_code=400, detail="Missing model name")

    try:
        target_info = model_manager.resolve_hf_pull_target(model_ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Hugging Face uses token auth. For now, allow the password field
    # to act as a token placeholder if provided.
    hf_token = password or None

    # Published sha256 and size, looked up before the transfer. Best effort:
    # when unavailable the digest is simply computed from the file later.
    hf_file_info = await model_manager.fetch_hf_file_info(
        repo_id=target_info["repo_id"],
        filename=target_info["filename"],
        token=hf_token,
    )

    # hf_hub_download creates the whole directory chain implied by the file
    # name before it checks that the file exists, and leaves it behind when
    # the check fails. A reference built from a browser URL therefore litters
    # the model store with empty blob/main and resolve/main trees. Refuse here
    # instead, but only when the repository actually answered.
    if hf_file_info.status == HF_LOOKUP_MISSING:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{target_info['filename']}' not found in repository "
                f"'{target_info['repo_id']}'"
            ),
        )

    if stream is False:
        try:
            local_path = await model_manager.pull_hf_file(
                repo_id=target_info["repo_id"],
                filename=target_info["filename"],
                token=hf_token,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if await model_manager.store_hf_digest(local_path, hf_file_info) is None:
            await model_manager.get_model_digest(local_path)

        await model_manager.ensure_metadata_cache_for_path(local_path)

        return JSONResponse({
            "status": "success",
            "path": local_path,
        })

    async def generate():
        yield json.dumps({"status": "resolving repository"}) + "\n"

        # Seed with what the pre-download lookup already told us (Content-
        # Length only arrives once the transfer itself starts), so the first
        # progress line does not have to wait for that.
        progress = PullProgress(total=hf_file_info.size or 0)

        download_task = asyncio.ensure_future(
            model_manager.pull_hf_file(
                repo_id=target_info["repo_id"],
                filename=target_info["filename"],
                token=hf_token,
                progress=progress,
            )
        )

        try:
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {download_task}, timeout=_PULL_PROGRESS_INTERVAL_SECONDS
                    )

                    completed, total = progress.snapshot()
                    yield json.dumps({
                        "status": "downloading model",
                        "completed": completed,
                        "total": total,
                    }) + "\n"

                    if done:
                        break

                local_path = await download_task
            except Exception as e:
                yield ollama_stream_error_line(str(e))
                return
        finally:
            # A client that disconnects mid-download (GeneratorExit) must not
            # leave this task dangling and unretrieved. The underlying thread
            # is not forcibly interrupted -- asyncio.to_thread offers no such
            # thing -- but the task itself is cleaned up promptly instead of
            # lingering until the transfer happens to finish on its own.
            if not download_task.done():
                download_task.cancel()

        if await model_manager.store_hf_digest(local_path, hf_file_info) is None:
            # The published digest was unusable. Hash the file here, while it
            # is still warm, rather than leaving the cost to a later listing.
            yield json.dumps({"status": "computing sha256 digest"}) + "\n"

            if await model_manager.get_model_digest(local_path) is None:
                yield json.dumps({"status": "digest unavailable"}) + "\n"

        yield json.dumps({"status": "creating metadata cache"}) + "\n"

        metadata = await model_manager.ensure_metadata_cache_for_path(local_path)
        if metadata is None:
            yield json.dumps({"status": "metadata cache unavailable"}) + "\n"
        else:
            yield json.dumps({"status": "metadata cache ready"}) + "\n"

        yield json.dumps({
            "status": "success",
            "path": local_path,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.delete("/delete")
async def delete_model_ollama(request: Request):
    payload = await request.json()

    model_ref = (payload.get("name") or payload.get("model") or "").strip()
    if not model_ref:
        raise HTTPException(status_code=400, detail="Missing model name")

    # Best-effort unload if the loaded model key matches the incoming ref.
    # If not loaded, this is harmless.
    try:
        if model_manager.is_model_loaded(model_ref):
            model_manager.unload_model(model_ref)
    except Exception:
        pass

    try:
        result = await model_manager.delete_model(model_ref)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "deleted": result["deleted_path"],
    }
