"""Request-local native tool state and completed Ollama object projection.

Native names are cumulative revisions; arguments are string suffixes. There is
no per-call commit marker, so object consumers wait for a verified response end.
This module never executes tools or repairs generated JSON.
"""

from dataclasses import dataclass, field
import json
import math

from tllama.native_config import _plain
from tllama.native_params import freeze_values
from tllama.worker import WorkerProtocolError


MAX_CALLS = 128
MAX_CALL_BYTES = 16 * 1024 * 1024


def _fail(message):
    raise WorkerProtocolError("Native tool calls: " + message)


def _string(value, label):
    if not isinstance(value, str):
        _fail(label + " must be a string")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate argument key")
        result[key] = value
    return result


def _constant(value):
    _fail("nonfinite argument constant: " + value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        _fail("nonfinite argument number")
    return result


def _bytes(value):
    try:
        return len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise WorkerProtocolError("Native tool calls: invalid Unicode") from exc


def argument_object(text):
    try:
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    except (ValueError, RecursionError) as exc:
        raise WorkerProtocolError("Native tool calls: arguments must be complete object JSON") from exc
    if not isinstance(result, dict):
        _fail("arguments must be an object")
    # JSON permits lone surrogate escapes, but public UTF-8 JSON cannot carry
    # the resulting strings. Validate nested strings before emitting any call.
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            _bytes(value)
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value)
            pending.extend(value.values())
    return result


@dataclass
class _Call:
    id: str = ""
    name: str = ""
    arguments: list = field(default_factory=list)


class ToolCalls:
    def __init__(self):
        self.calls = {}
        self.ids = {}
        self.size = 0

    def push(self, values):
        if values is None:
            return ()
        if not isinstance(values, list):
            _fail("tool_calls must be an array")
        deltas = []
        for value in values:
            if not isinstance(value, dict):
                _fail("each call must be an object")
            index = value.get("index")
            if type(index) is not int or not 0 <= index < MAX_CALLS:
                _fail("index is outside the supported range")
            call = self.calls.setdefault(index, _Call())
            delta = {"index": index}
            if "type" in value and value["type"] != "function":
                _fail("only function calls are supported")
            if "id" in value:
                identity = _string(value["id"], "id")
                if not identity or call.id and call.id != identity:
                    _fail("call identity changed or is empty")
                if identity in self.ids and self.ids[identity] != index:
                    _fail("call IDs must be unique")
                self.ids[identity] = index
                if not call.id:
                    delta.update(id=identity, type="function")
                    self.size += _bytes(identity)
                call.id = identity
            function = value.get("function", {})
            if not isinstance(function, dict):
                _fail("function must be an object")
            fragment = {}
            if "name" in function:
                name = _string(function["name"], "name")
                if not name.startswith(call.name):
                    _fail("name revision must extend its previous prefix")
                suffix = name[len(call.name):]
                if suffix:
                    fragment["name"] = suffix
                    self.size += _bytes(suffix)
                call.name = name
            if "arguments" in function:
                text = _string(function["arguments"], "arguments")
                self.size += _bytes(text)
                # Check before retaining another fragment of an unbounded stream.
                if self.size > MAX_CALL_BYTES:
                    _fail("accumulated call data exceeds the byte limit")
                if text:
                    call.arguments.append(text)
                    fragment["arguments"] = text
            if self.size > MAX_CALL_BYTES:
                _fail("accumulated call data exceeds the byte limit")
            if fragment:
                delta["function"] = fragment
            if len(delta) > 1:
                deltas.append(delta)
        return freeze_values(deltas)

    def finish(self, reason):
        if reason == "tool_calls":
            if not self.calls or sorted(self.calls) != list(range(len(self.calls))):
                _fail("completed calls require contiguous indices")
            for call in self.calls.values():
                if not call.id or not call.name:
                    _fail("completed calls require an ID and a name")
                argument_object("".join(call.arguments))
        elif self.calls and reason != "length":
            _fail("calls require a tool_calls or length finish reason")
        return freeze_values([
            {"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": "".join(call.arguments)}}
            for _, call in sorted(self.calls.items())])


def full_calls(values, reason):
    state = ToolCalls()
    if values is not None:
        if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
            _fail("tool_calls must be an array of objects")
        state.push([{**value, "index": index} for index, value in enumerate(values)])
    return state.finish(reason)


def ollama_calls(values):
    result = _plain(values)
    for index, call in enumerate(result):
        call["function"]["arguments"] = argument_object(call["function"]["arguments"])
        call["function"]["index"] = index
    return result
