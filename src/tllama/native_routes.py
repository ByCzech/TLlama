"""Shared ownership and preparation for public native text/image endpoints."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, aclosing
from dataclasses import replace
import time

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect
from tllama.errors import ollama_stream_error_line, openai_stream_error_frame

from tllama.helpers.common import get_iso_time, validate_format, validate_sampling_options
from tllama.helpers.prompt_render import render_generate_text
from tllama.helpers.reasoning_split import detect_reasoning_format
from tllama.helpers.vision import content_carries_images, request_carries_images, NO_MULTIMODAL_SUPPORT
from tllama.native_api import (NativeAPI, ReplyContext, ollama_response, openai_response,
                              ollama_stream, openai_stream, public_error)
from tllama.native_config import _plain
from tllama.native_fim import prepare_fim, validate_fim_input
from tllama.native_requests import (NativeRequest, prepare_ollama_chat, prepare_openai_chat,
                                   prepare_completion, prepare_ollama_generate_images,
                                   _response_format, normalize_chat_input)
from tllama.residency import _settle_owned
from tllama.worker import WorkerProtocolError


class NativeStreamingResponse(StreamingResponse):
    """Own the acquired lease even if the body never starts or send fails."""

    def __init__(self, content, stack, *, media_type):
        super().__init__(content, media_type=media_type)
        self._stack, self._cleanup_task = stack, None

    async def _cleanup(self, tasks):
        try:
            # Wait for the producer to stop before closing its generator.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.body_iterator.aclose()
        finally:
            await self._stack.aclose()

    async def __call__(self, scope, receive, send):
        tasks = []
        try:
            # Observe disconnects even under ASGI 2.4: a producer can be
            # waiting for its next native token without calling send again.
            stream = asyncio.create_task(self.stream_response(send))
            disconnected = asyncio.create_task(self.listen_for_disconnect(receive))
            tasks.extend((stream, disconnected))
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if stream in done:
                try:
                    await stream
                except OSError as exc:
                    raise ClientDisconnect() from exc
            else:
                await disconnected
        finally:
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup(tasks))
            await _settle_owned(self._cleanup_task)
        if self.background is not None:
            await self.background()


def _validate(request, api, kind):
    if api == "ollama":
        validate_format(request.format)
        validate_sampling_options(request.options)
        if kind == "completion":
            if request.suffix is not None:
                validate_fim_input(request)
            if request.context:
                raise ValueError("Nonempty generate context is unsupported; supply explicit messages instead")
            if request.raw and any(value is not None for value in (request.template, request.system, request.context)):
                raise ValueError("raw mode does not support template, system, or context")
            if request_carries_images(request.images):
                if request.raw:
                    raise ValueError("Images in /api/generate do not support raw mode")
                if request.template is not None:
                    raise ValueError("Images in /api/generate use the model template; request template is unsupported")
                messages = []
                if request.system is not None:
                    messages.append({"role": "system", "content": request.system})
                messages.append({"role": "user", "content": request.prompt, "images": request.images})
                return normalize_chat_input(messages)
    else:
        validate_sampling_options({key: value for key, value in {
            "temperature": request.temperature, "top_p": request.top_p, "num_predict": request.max_tokens,
        }.items() if value is not None})
        if request.response_format is not None:
            _response_format(request.response_format)
        if request.functions is not None or request.function_call is not None:
            raise ValueError("Deprecated functions/function_call is unsupported; use tools/tool_choice")
    if kind == "chat":
        if any(getattr(message, "function_call", None) is not None or message.role == "function"
               for message in request.messages):
            raise ValueError("Deprecated function-call history is unsupported; use tool messages")
        return normalize_chat_input(request.messages)


def lease_metadata(lease):
    metadata = _plain(lease.definition.metadata)
    props = lease.worker.props or {}
    # Worker props reflect the actual selected template. TOML is already part
    # of the generation; no mutable definition/header lookup occurs here.
    template = lease.resolved.launch.template
    if template is None:
        template = props.get("chat_template", metadata.get("template"))
    metadata["template"] = template
    metadata["default_system_prompt"] = lease.resolved.system_prompt
    if lease.resolved.model_sampling is not None:
        metadata["sampling_defaults"] = _plain(lease.resolved.model_sampling)
        metadata["stop_defaults"] = list(lease.resolved.model_stop or ())
    return metadata


async def _completion_request(request, lease, metadata) -> NativeRequest:
    if request.suffix is not None:
        return await prepare_fim(request, lease, metadata)
    props = lease.worker.props or {}
    bos, eos = props.get("bos_token", ""), props.get("eos_token", "")
    if not isinstance(bos, str) or not isinstance(eos, str):
        raise WorkerProtocolError("Native BOS/EOS properties must be strings")
    if request.raw:
        prompt, stop = request.prompt, None
    else:
        prompt, stop = render_generate_text(metadata, request, bos_token=bos, eos_token=eos)
    prepared = prepare_completion(prompt, request.options, resolved=lease.resolved, metadata_info=metadata,
                                  format_value=request.format,
                                  reasoning_format=detect_reasoning_format(request.model, metadata), think=request.think)
    body = prepared.body()
    if stop is not None:
        body["stop"] = stop
    tokens = await lease.worker.tokenize(prompt, add_special=True)
    # Preserve the former completion tokenizer's initial duplicate-BOS rule.
    if bos and prompt.startswith(bos) and len(tokens) > 1 and tokens[0] == tokens[1]:
        tokens = tokens[1:]
    body["prompt"] = tokens  # Native mixed token input does not add another BOS.
    return replace(prepared, payload=body)


async def serve_native(manager, request, *, api, kind="chat"):
    stack, transferred = AsyncExitStack(), False
    phase = "prepare"
    try:
        normalized = _validate(request, api, kind)
        keep_alive = request.keep_alive if api == "ollama" else None
        if api == "ollama":
            seconds = manager.resolve_keep_alive(keep_alive)
            empty = (not request.messages if kind == "chat" else
                     not request.prompt and request.suffix is None and not normalized)
            if empty:
                phase = "load"
                if seconds == 0:
                    await manager.unload_native_model(request.model)
                else:
                    await manager.preload_native_model(request.model, num_ctx=request.options.get("num_ctx"), keep_alive=keep_alive)
                payload = {"model": request.model, "created_at": get_iso_time(), "done": True,
                           "done_reason": "unload" if seconds == 0 else "load"}
                payload.update({"message": {"role": "assistant", "content": ""}} if kind == "chat" else {"response": ""})
                return payload
        started = time.monotonic_ns()
        phase = "load"
        lease = await stack.enter_async_context(manager.acquire_native_model(
            request.model, num_ctx=request.options.get("num_ctx") if api == "ollama" else None, keep_alive=keep_alive))
        context = ReplyContext(request.model, started_ns=started, acquired_ns=time.monotonic_ns())
        metadata = lease_metadata(lease)
        phase = "prepare"
        if normalized and any(content_carries_images(message['content']) for message in normalized):
            modalities = (lease.worker.props or {}).get('modalities')
            if not isinstance(modalities, dict) or modalities.get('vision') is not True:
                raise ValueError(NO_MULTIMODAL_SUPPORT)
        if kind == "completion" and normalized is not None:
            prepared = prepare_ollama_generate_images(request, resolved=lease.resolved, metadata_info=metadata,
                                                      normalized_messages=normalized)
        elif kind == "completion":
            prepared = await _completion_request(request, lease, metadata)
        else:
            builder = prepare_ollama_chat if api == "ollama" else prepare_openai_chat
            prepared = builder(request, resolved=lease.resolved, metadata_info=metadata, normalized_messages=normalized)
        native = NativeAPI(lease.worker, prepared)
        phase = "inference"
        if not request.stream:
            result = await native.result()
            return ollama_response(result, context, kind=kind) if api == "ollama" else openai_response(result, context)

        async def body():
            try:
                async with native.stream() as events:
                    frames = ollama_stream(events, context, kind=kind) if api == "ollama" else openai_stream(events, context)
                    async with aclosing(frames):
                        async for frame in frames:
                            yield frame
            except Exception as exc:
                # Opening an upstream response can fail before the serializer
                # begins, but the public streaming headers are already sent.
                error = public_error(exc, api=api)
                yield ollama_stream_error_line(error.message) if api == "ollama" else openai_stream_error_frame(error.message, error.status)

        response = NativeStreamingResponse(body(), stack.pop_all(),
                                           media_type="application/x-ndjson" if api == "ollama" else "text/event-stream")
        transferred = True
        return response
    except HTTPException:
        raise
    except Exception as exc:
        error = public_error(exc, api=api, phase=phase)
        raise HTTPException(error.status, error.message) from exc
    finally:
        if not transferred:
            await _settle_owned(asyncio.create_task(stack.aclose()))
