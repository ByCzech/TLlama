"""Measured statistics and public output semantics independent of model/GPU."""

import asyncio
from contextlib import aclosing
import json

import pytest

from tllama.helpers.model_toml import TomlModelError
from tllama.native_api import (NativeAPI, NativeResult, RequestStats, ReplyContext, read_result,
    ollama_response, openai_response, ollama_stream, openai_stream, public_error, _read_events)
from tllama.native_requests import NativeRequest
from tllama.residency import ResidencyClosed
from tllama.worker import WorkerProtocolError, WorkerRequestError, WorkerExited


CHAT = NativeRequest("chat", {"messages": []})
RAW = NativeRequest("completion", {"prompt": "hi"})
USAGE = {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35,
         "prompt_tokens_details": {"cached_tokens": 21}}
TIMINGS = {"prompt_n": 4, "cache_n": 21, "predicted_n": 10, "prompt_ms": 176.128, "predicted_ms": 728.803}


def chat_result(content="answer", reasoning="thought", finish="stop", **extra):
    return {"choices": [{"index": 0, "message": {"content": content, "reasoning_content": reasoning},
                          "finish_reason": finish}], "usage": USAGE, "timings": TIMINGS, **extra}


async def sequence(*values):
    for value in values:
        if isinstance(value, BaseException):
            raise value
        yield value


async def collected(values, request=CHAT):
    async with aclosing(sequence(*values)) as upstream:
        async with aclosing(_read_events(upstream, request)) as events:
            return [event async for event in events]


def context():
    return ReplyContext("public/model", completion_id="chatcmpl-public-1", created=123,
                        started_ns=100, acquired_ns=130, clock=lambda: 210)


def test_cache_hit_preserves_logical_and_evaluated_counts_with_exact_ns():
    result = read_result(chat_result(), CHAT)
    assert result.stats.input_tokens == 25 and result.stats.evaluated_tokens == 4
    assert result.stats.cached_tokens == 21 and result.stats.output_tokens == 10
    ollama = ollama_response(result, context())
    assert ollama["prompt_eval_count"] == 4 and ollama["prompt_eval_cached_count"] == 21
    assert ollama["prompt_eval_duration"] == 176_128_000 and ollama["eval_duration"] == 728_803_000
    assert ollama["total_duration"] == 110 and ollama["load_duration"] == 30
    assert openai_response(result, context())["usage"] == USAGE


def test_raw_tokens_cached_cannot_masquerade_as_input_cache_measurement():
    result = read_result({"content": "hi", "stop": True, "stop_type": "limit", "tokens_evaluated": 1,
                          "tokens_predicted": 8, "tokens_cached": 8,
                          "timings": {"prompt_n": 1, "predicted_n": 8}}, RAW)
    assert result.finish_reason == "length" and result.stats.input_tokens == 1
    assert result.stats.total_tokens == 9 and result.stats.cached_tokens is None
    assert "prompt_eval_cached_count" not in result.stats.ollama()


def test_missing_stats_do_not_become_unknown_zero_or_prompt_estimate():
    result = read_result({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}, CHAT)
    assert result.stats.ollama() == result.stats.openai() == {}
    assert "prompt_eval_count" not in ollama_response(result, ReplyContext("m"))
    stats = RequestStats.read({"usage": {"prompt_tokens": 25}})
    assert stats.input_tokens == 25 and stats.evaluated_tokens is None
    assert "prompt_eval_count" not in stats.ollama() and "total_tokens" not in stats.openai()


def test_measured_zero_is_preserved_and_merge_does_not_drop_it():
    stats = RequestStats(25, 4, 21).merge(RequestStats.read({"timings": {"prompt_n": 0, "cache_n": 0, "prompt_ms": 0}}))
    assert stats.ollama() == {"prompt_eval_count": 0, "prompt_eval_cached_count": 0, "prompt_eval_duration": 0}
    assert stats.input_tokens == 25


@pytest.mark.parametrize("payload", [{"usage": {"prompt_tokens": True}}, {"timings": {"cache_n": -1}},
    {"timings": {"predicted_n": 1.5}}, {"timings": {"prompt_ms": -1}}, {"timings": {"prompt_ms": float("nan")}},
    {"timings": {"predicted_ms": float("inf")}}, {"timings": {"prompt_ms": 10**400}},
    {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}}, {"usage": []},
    {"timings": "bad"}, {"usage": {"prompt_tokens_details": []}}])
def test_invalid_native_measurements_fail_as_protocol_errors(payload):
    with pytest.raises(WorkerProtocolError):
        RequestStats.read(payload)


def test_structured_chat_reasoning_does_not_reparse_literal_markers_in_content():
    result = read_result(chat_result(content="Literal <think>example</think> answer"), CHAT)
    assert result.reasoning == "thought" and result.content == "Literal <think>example</think> answer"
    assert ollama_response(result, context())["message"]["thinking"] == "thought"
    message = openai_response(result, context())["choices"][0]["message"]
    assert message["reasoning_content"] == "thought" and message["content"] == result.content


def test_reasoning_only_length_is_success_with_empty_answer():
    result = read_result(chat_result(content=None, reasoning="unfinished thought", finish="length"), CHAT)
    message = openai_response(result, context())["choices"][0]["message"]
    assert result.finish_reason == "length" and message["content"] == ""
    assert message["reasoning_content"] == "unfinished thought"


def test_tool_call_result_keeps_native_arguments_and_length_semantics():
    payload = chat_result(finish="tool_calls")
    payload["choices"][0]["message"]["tool_calls"] = [{"id": "call-1", "function": {"name": "f", "arguments": "{}"}}]
    result = read_result(payload, CHAT)
    payload["choices"][0]["message"]["tool_calls"][0]["id"] = "mutated"
    assert openai_response(result, context())["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"


@pytest.mark.parametrize("stop_type,reason", [("eos", "stop"), ("word", "stop"), ("limit", "length")])
def test_raw_terminal_reasons_are_translated_explicitly(stop_type, reason):
    result = read_result({"content": "x", "stop": True, "stop_type": stop_type}, RAW)
    payload = ollama_response(result, context(), kind="completion")
    assert payload["done_reason"] == reason and payload["context"] == []


@pytest.mark.parametrize("payload", [{"content": "x"}, {"stop": True, "stop_type": "none"},
    {"stop": True, "stop_type": []}, {"stop": True, "stop_type": "limit", "content": []}])
def test_invalid_raw_terminal_is_not_a_success(payload):
    with pytest.raises(WorkerProtocolError):
        read_result(payload, RAW)


@pytest.mark.parametrize("payload", [{"choices": []}, {"choices": [{}, {}]},
    {"choices": [{"message": {}, "finish_reason": None}]},
    {"choices": [{"message": {}, "finish_reason": []}]},
    {"choices": [{"index": True, "message": {}, "finish_reason": "stop"}]},
    {"choices": [{"message": "bad", "finish_reason": "stop"}]},
    {"choices": [{"message": {"tool_calls": "bad"}, "finish_reason": "stop"}]}])
def test_invalid_chat_shape_is_a_typed_protocol_error(payload):
    with pytest.raises(WorkerProtocolError):
        read_result(payload, CHAT)


async def test_stream_stats_can_follow_finish_in_empty_choices_and_are_request_local():
    values = [{"choices": [{"delta": {"reasoning_content": "thought"}}]},
              {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
              {"choices": [], "usage": USAGE, "timings": TIMINGS}]
    first, second = await asyncio.gather(collected(values), collected([
        {"choices": [{"delta": {}, "finish_reason": "length"}]}, {"choices": [], "usage": {"completion_tokens": 0}}]))
    assert first[0].reasoning == "thought" and first[1].content == "answer"
    assert first[-1].terminal and first[-1].stats.cached_tokens == 21
    assert second[-1].stats.output_tokens == 0 and second[-1].stats.cached_tokens is None


async def test_raw_reasoning_marker_split_across_events_matches_non_stream():
    request = NativeRequest("completion", {}, reasoning_format="think_prefill")
    values = [{"content": "<thi", "stop": False}, {"content": "nk>thought</thi", "stop": False},
              {"content": "nk>answer", "stop": True, "stop_type": "limit", "timings": TIMINGS}]
    events = await collected(values, request)
    result = read_result({"content": "<think>thought</think>answer", "stop": True, "stop_type": "limit"}, request)
    assert "".join(e.reasoning for e in events) == result.reasoning == "thought"
    assert "".join(e.content for e in events) == result.content == "answer"
    assert events[-1].terminal and events[-1].finish_reason == "length"


async def test_raw_split_preserves_whitespace_equally_in_stream_and_full_result():
    request = NativeRequest("completion", {}, reasoning_format="think_prefill")
    text = "<think> thought \n</think> answer \n"
    result = read_result({"content": text, "stop": True, "stop_type": "eos"}, request)
    events = await collected([{"content": char, "stop": False} for char in text] + [
        {"content": "", "stop": True, "stop_type": "eos"}], request)
    assert result.reasoning == " thought \n" == "".join(event.reasoning for event in events)
    assert result.content == " answer \n" == "".join(event.content for event in events)


@pytest.mark.parametrize("values", [[{"choices": [{"delta": {"content": "x"}}]}],
    [{"choices": []}], [{"choices": [{"delta": {}, "finish_reason": "stop"}]},
                         {"choices": [{"delta": {"content": "late"}}]}]])
async def test_incomplete_or_out_of_order_stream_does_not_invent_done(values):
    with pytest.raises(WorkerProtocolError):
        await collected(values)


@pytest.mark.parametrize("include_usage", [False, True])
async def test_openai_frames_keep_public_identity_reasoning_tools_and_terminal_order(include_usage):
    events = sequence(NativeResult(reasoning="thought"), NativeResult(content="answer"),
        NativeResult(tool_calls=({"index": 0, "function": {"arguments": '{"x":'}},)),
        NativeResult(finish_reason="tool_calls", stats=RequestStats.read({"usage": USAGE}), terminal=True))
    frames = [f async for f in openai_stream(events, context(), include_usage=include_usage)]
    assert frames[-1] == "data: [DONE]\n\n"
    bodies = [json.loads(f[6:]) for f in frames[:-1]]
    assert all(b["id"] == "chatcmpl-public-1" and b["model"] == "public/model" and b["created"] == 123 for b in bodies)
    assert bodies[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert bodies[1]["choices"][0]["delta"] == {"reasoning_content": "thought"}
    assert bodies[2]["choices"][0]["delta"] == {"content": "answer"}
    assert bodies[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"x":'
    final = bodies[-2] if include_usage else bodies[-1]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    if include_usage:
        assert bodies[-1]["choices"] == [] and bodies[-1]["usage"] == USAGE


async def test_ollama_ndjson_final_has_required_empty_message_and_measured_stats():
    events = sequence(NativeResult(reasoning="thought"), NativeResult(content="answer"),
                      NativeResult(finish_reason="stop", stats=RequestStats.read({"usage": USAGE, "timings": TIMINGS}), terminal=True))
    lines = [line async for line in ollama_stream(events, context())]
    bodies = [json.loads(line) for line in lines]
    assert all(line.endswith("\n") and not line.startswith("data:") for line in lines)
    assert bodies[0]["message"] == {"role": "assistant", "content": "", "thinking": "thought"}
    assert bodies[-1]["message"] == {"role": "assistant", "content": ""}
    assert bodies[-1]["done"] and bodies[-1]["prompt_eval_count"] == 4


@pytest.mark.parametrize("presenter", [ollama_stream, openai_stream])
async def test_error_after_partial_output_has_error_envelope_and_no_success_terminal(presenter):
    error = WorkerRequestError(200, {"error": {"code": 400, "message": "bad grammar"}})
    frames = [f async for f in presenter(sequence(NativeResult(content="partial"), error), context())]
    assert "partial" in "".join(frames) and "bad grammar" in frames[-1]
    assert not any('[DONE]' in f or '"done": true' in f for f in frames)
    if presenter is openai_stream:
        assert json.loads(frames[-1][6:])["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("presenter", [ollama_stream, openai_stream])
async def test_cancellation_is_not_rewritten_as_a_stream_error(presenter):
    with pytest.raises(asyncio.CancelledError):
        [f async for f in presenter(sequence(asyncio.CancelledError()), context())]


@pytest.mark.parametrize("error,phase,oa,ol", [(ValueError("bad input"), "prepare", 400, 400),
    (TomlModelError("bad TOML"), "load", 500, 500), (FileNotFoundError("missing"), "load", 404, 400),
    (RuntimeError("load failed"), "load", 500, 400), (WorkerExited(71, "crashed"), "inference", 500, 500),
    (WorkerProtocolError("bad native response"), "inference", 500, 500),
    (ResidencyClosed("shutdown"), "load", 503, 503),
    (WorkerRequestError(400, {"error": {"message": "native invalid"}}), "inference", 400, 400)])
def test_pre_header_error_status_and_public_envelope_matrix(error, phase, oa, ol):
    for api, expected in [("openai", oa), ("ollama", ol)]:
        translated = public_error(error, api=api, phase=phase)
        assert translated.status == expected and translated.message == str(error)
        if api == "openai":
            assert "param" in translated.body(api)["error"]
        else:
            assert isinstance(translated.body(api)["error"], str)
