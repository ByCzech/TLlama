"""Request-local native results and public JSON/stream adapters.

The caller owns the model lease and keeps it around the complete call/stream.
This layer owns only its upstream response context, never model residency.
Public ownership and routing policy is implemented in native_routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields, replace
import json
import math
import time
from typing import Callable
from uuid import uuid4

from tllama.errors import ollama_stream_error_line, openai_stream_error_frame, openai_error_type
from tllama.helpers.common import get_iso_time
from tllama.helpers.model_toml import TomlModelError
from tllama.helpers.reasoning_split import ReasoningStreamSplitter
from tllama.native_config import _plain
from tllama.native_requests import NativeRequest
from tllama.native_tools import ToolCalls, full_calls, ollama_calls
from tllama.residency import ResidencyClosed
from tllama.worker import WorkerProtocolError, WorkerRequestError


def _object(value, label):
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"Native {label} must be an object")
    return value


def _count(value, label):
    if value is not None and (type(value) is not int or value < 0):
        raise WorkerProtocolError(f"Native {label} must be a nonnegative integer")
    return value


def _nanoseconds(value, label):
    if value is None:
        return None
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0 and math.isfinite(value * 1_000_000)
    except OverflowError:
        valid = False
    if not valid:
        raise WorkerProtocolError(f"Native {label} must be nonnegative finite milliseconds")
    return round(value * 1_000_000)


def _text(value, label):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise WorkerProtocolError(f"Native {label} must be a string or null")
    return value


def _finish(value):
    if not isinstance(value, str) or value not in {"stop", "length", "tool_calls"}:
        raise WorkerProtocolError(f"Invalid native finish_reason: {value!r}")
    return value


def _raw_finish(payload):
    if payload.get("stop") is not True:
        raise WorkerProtocolError("Native completion lacks its terminal stop flag")
    stop_type = payload.get("stop_type")
    reason = {"eos": "stop", "word": "stop", "limit": "length"}.get(stop_type) if isinstance(stop_type, str) else None
    if reason is None:
        raise WorkerProtocolError("Native completion has an unknown terminal stop_type")
    return reason


@dataclass(frozen=True)
class RequestStats:
    input_tokens: int | None = None
    evaluated_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_duration_ns: int | None = None
    output_duration_ns: int | None = None
    draft_tokens: int | None = None
    accepted_draft_tokens: int | None = None

    def __post_init__(self):
        drafted = _count(self.draft_tokens, "draft token count")
        accepted = _count(self.accepted_draft_tokens, "accepted draft token count")
        if drafted is not None and accepted is not None and accepted > drafted:
            raise WorkerProtocolError("Native accepted draft count exceeds drafted count")

    @classmethod
    def read(cls, payload):
        usage = _object(payload.get("usage", {}), "usage")
        timings = _object(payload.get("timings", {}), "timings")
        details = _object(usage.get("prompt_tokens_details", {}), "prompt_tokens_details")
        input_tokens = _count(usage.get("prompt_tokens", payload.get("tokens_evaluated")), "input token count")
        output_tokens = _count(usage.get("completion_tokens", payload.get("tokens_predicted", timings.get("predicted_n"))),
                               "output token count")
        total_tokens = _count(usage.get("total_tokens"), "total token count")
        if input_tokens is not None and output_tokens is not None:
            if total_tokens is not None and total_tokens != input_tokens + output_tokens:
                raise WorkerProtocolError("Native usage total disagrees with input/output counts")
            total_tokens = input_tokens + output_tokens
        cached = _count(details.get("cached_tokens", timings.get("cache_n")), "cached token count")
        evaluated = _count(timings.get("prompt_n"), "evaluated token count")
        # tokens_cached is slot occupancy AFTER generation, not input reuse.
        return cls(input_tokens, evaluated, cached, output_tokens, total_tokens,
                   _nanoseconds(timings.get("prompt_ms"), "prompt_ms"),
                   _nanoseconds(timings.get("predicted_ms"), "predicted_ms"),
                   timings.get("draft_n"), timings.get("draft_n_accepted"))

    def merge(self, newer):
        return replace(self, **{f.name: getattr(newer, f.name) for f in fields(self)
                                if getattr(newer, f.name) is not None})

    def ollama(self):
        # Count the tokens whose evaluation time is reported; keep the full
        # input separately for OpenAI usage. Missing measurement stays absent.
        return {key: value for key, value in {
            "prompt_eval_count": self.evaluated_tokens, "prompt_eval_cached_count": self.cached_tokens,
            "eval_count": self.output_tokens, "prompt_eval_duration": self.prompt_duration_ns,
            "eval_duration": self.output_duration_ns,
        }.items() if value is not None}

    def openai(self):
        result = {key: value for key, value in {
            "prompt_tokens": self.input_tokens, "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }.items() if value is not None}
        if self.cached_tokens is not None:
            result["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
        return result

    def extensions(self):
        """Measured request counters, separate from either API's token usage."""
        speculative = {key: value for key, value in {
            "draft_tokens": self.draft_tokens, "accepted_tokens": self.accepted_draft_tokens,
        }.items() if value is not None}
        if not speculative:
            return {}
        if self.draft_tokens and self.accepted_draft_tokens is not None:
            speculative["acceptance"] = self.accepted_draft_tokens / self.draft_tokens
        return {"tllama": {"speculative": speculative}}


@dataclass(frozen=True)
class NativeResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: tuple = ()
    finish_reason: str | None = None
    stats: RequestStats = field(default_factory=RequestStats)
    terminal: bool = False


def _choice(payload, *, stream):
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) > 1 or not stream and len(choices) != 1:
        raise WorkerProtocolError("Native chat must return one choice, or an empty stream usage choice list")
    if not choices:
        if "usage" not in payload and "timings" not in payload:
            raise WorkerProtocolError("Empty native choices without usage/timings")
        return None
    choice = _object(choices[0], "choice")
    if type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0:
        raise WorkerProtocolError("Native chat returned a nonzero choice index")
    return choice


def read_result(payload, request: NativeRequest):
    stats = RequestStats.read(payload)
    if request.kind == "chat":
        choice = _choice(payload, stream=False)
        message = _object(choice.get("message"), "message")
        content = _text(message.get("content"), "content")
        reasoning = _text(message.get("reasoning_content"), "reasoning_content")
        finish = _finish(choice.get("finish_reason"))
        return NativeResult(content, reasoning, full_calls(message.get("tool_calls"), finish),
                            finish, stats, True)
    content = _text(payload.get("content"), "completion content")
    splitter = ReasoningStreamSplitter(request.reasoning_format, think_value=request.think)
    pieces = splitter.push(content) + splitter.finish()
    reasoning = "".join(piece for kind, piece in pieces if kind == "thinking")
    content = "".join(piece for kind, piece in pieces if kind == "response")
    return NativeResult(content, reasoning, (), _raw_finish(payload), stats, True)


async def _read_events(events, request):
    stats, finish = RequestStats(), None
    tools = ToolCalls()
    splitter = ReasoningStreamSplitter(request.reasoning_format, think_value=request.think)
    async for payload in events:
        stats = stats.merge(RequestStats.read(payload))
        if request.kind == "chat":
            choice = _choice(payload, stream=True)
            if choice is None:
                continue
            delta = _object(choice.get("delta"), "delta")
            content = _text(delta.get("content"), "delta content")
            reasoning = _text(delta.get("reasoning_content"), "delta reasoning_content")
            if finish is not None:
                raise WorkerProtocolError("Native chat choice arrived after its finish_reason")
            calls = tools.push(delta.get("tool_calls"))
            if choice.get("finish_reason") is not None:
                finish = _finish(choice["finish_reason"])
            if content or reasoning or calls:
                yield NativeResult(content, reasoning, calls)
        else:
            if finish is not None:
                raise WorkerProtocolError("Native completion arrived after its terminal stop")
            for kind, piece in splitter.push(_text(payload.get("content"), "completion content")):
                yield NativeResult(content=piece if kind == "response" else "",
                                   reasoning=piece if kind == "thinking" else "")
            if payload.get("stop") is True:
                finish = _raw_finish(payload)
    # The transport must validate [DONE]/raw stop before a public success.
    if finish is None:
        raise WorkerProtocolError("Native stream ended without a finish reason")
    if request.kind == "completion":
        for kind, piece in splitter.finish():
            yield NativeResult(content=piece if kind == "response" else "",
                               reasoning=piece if kind == "thinking" else "")
    calls = tools.finish(finish) if request.kind == "chat" else ()
    yield NativeResult(tool_calls=calls, finish_reason=finish, stats=stats, terminal=True)


class NativeAPI:
    def __init__(self, worker, request: NativeRequest):
        self.worker, self.request = worker, request

    async def result(self):
        call = self.worker.chat if self.request.kind == "chat" else self.worker.completion
        return read_result(await call(self.request.body()), self.request)

    @asynccontextmanager
    async def stream(self):
        call = self.worker.chat_stream if self.request.kind == "chat" else self.worker.completion_stream
        async with call(self.request.body()) as upstream:
            events = _read_events(upstream, self.request)
            try:
                yield events
            finally:
                await events.aclose()


@dataclass(frozen=True)
class ReplyContext:
    model: str
    completion_id: str = field(default_factory=lambda: "chatcmpl-" + uuid4().hex)
    created: int = field(default_factory=lambda: int(time.time()))
    started_ns: int | None = None
    acquired_ns: int | None = None
    clock: Callable[[], int] = time.monotonic_ns

    def durations(self):
        if self.started_ns is None:
            return {}
        result = {"total_duration": max(0, self.clock() - self.started_ns)}
        if self.acquired_ns is not None:
            result["load_duration"] = max(0, self.acquired_ns - self.started_ns)
        return result


def _message(result, *, ollama):
    message = {"role": "assistant", "content": result.content}
    if result.reasoning:
        message["thinking" if ollama else "reasoning_content"] = result.reasoning
    if result.tool_calls:
        if not ollama:
            message["tool_calls"] = _plain(result.tool_calls)
        elif result.finish_reason == "tool_calls":
            message["tool_calls"] = ollama_calls(result.tool_calls)
    return message


def ollama_response(result, context, *, kind="chat"):
    payload = {"model": context.model, "created_at": get_iso_time(), "done": result.terminal}
    if kind == "chat":
        payload["message"] = _message(result, ollama=True)
    elif kind == "completion":
        payload["response"] = result.content
        if result.reasoning:
            payload["thinking"] = result.reasoning
        if result.terminal:
            payload["context"] = []  # Compatibility placeholder, not conversation/KV history.
    else:
        raise ValueError("Ollama response kind must be chat or completion")
    if result.terminal:
        reason = "stop" if result.finish_reason == "tool_calls" else result.finish_reason
        payload.update(done_reason=reason, **context.durations(), **result.stats.ollama(),
                       **result.stats.extensions())
    return payload


def openai_response(result, context):
    return {"id": context.completion_id, "object": "chat.completion", "created": context.created,
            "model": context.model, "choices": [{"index": 0, "message": _message(result, ollama=False),
                                                  "finish_reason": result.finish_reason}],
            "usage": result.stats.openai(), **result.stats.extensions()}


@dataclass(frozen=True)
class PublicError:
    status: int
    message: str

    def body(self, api):
        if api == "ollama":
            return {"error": self.message}
        if api != "openai":
            raise ValueError("Unknown public API")
        return {"error": {"message": self.message, "type": openai_error_type(self.status),
                          "param": None, "code": None}}


def public_error(error, *, api, phase="inference"):
    if api not in {"ollama", "openai"} or phase not in {"prepare", "load", "inference"}:
        raise ValueError("Unknown error API or phase")
    status = 500
    if isinstance(error, TomlModelError):
        pass
    elif isinstance(error, ResidencyClosed):
        status = 503
    elif isinstance(error, WorkerRequestError):
        status = error.status_code
    elif phase == "prepare" and isinstance(error, ValueError):
        status = 400
    elif phase == "load":
        if isinstance(error, FileNotFoundError):
            status = 404 if api == "openai" else 400
        elif isinstance(error, ValueError) or api == "ollama":
            status = 400
    return PublicError(status, str(error))


async def ollama_stream(events, context, *, kind="chat"):
    try:
        async for event in events:
            if kind == "chat" and event.tool_calls:
                if event.terminal and event.finish_reason == "tool_calls":
                    # The transport has now verified the complete native stream.
                    # Deliver each object once, before the separate done/stats line.
                    calls = replace(event, content="", reasoning="", terminal=False)
                    yield json.dumps(ollama_response(calls, context), ensure_ascii=False) + "\n"
                event = replace(event, tool_calls=())
                if not event.terminal and not event.content and not event.reasoning:
                    continue
            yield json.dumps(ollama_response(event, context, kind=kind), ensure_ascii=False) + "\n"
    except Exception as exc:
        yield ollama_stream_error_line(public_error(exc, api="ollama").message)


def _sse(context, choices, **extra):
    body = {"id": context.completion_id, "object": "chat.completion.chunk", "created": context.created,
            "model": context.model, "choices": choices, **extra}
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


async def openai_stream(events, context, *, include_usage=False):
    try:
        role_sent = False
        async for event in events:
            if not role_sent:
                yield _sse(context, [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
                role_sent = True
            if event.terminal:
                yield _sse(context, [{"index": 0, "delta": {}, "finish_reason": event.finish_reason}],
                           **event.stats.extensions())
                if include_usage:
                    yield _sse(context, [], usage=event.stats.openai())
                yield "data: [DONE]\n\n"
            else:
                delta = _message(event, ollama=False)
                delta.pop("role")
                if not event.content:
                    delta.pop("content")
                yield _sse(context, [{"index": 0, "delta": delta, "finish_reason": None}])
    except Exception as exc:
        error = public_error(exc, api="openai")
        yield openai_stream_error_frame(error.message, error.status)
