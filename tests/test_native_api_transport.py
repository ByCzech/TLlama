"""Private API adapters use real TCP, subprocesses and upstream cancellation."""

import asyncio
import json
from pathlib import Path

import pytest

from tllama.config import BackendConfig
from tllama.native_config import resolve_worker_config
from tllama.native_requests import NativeRequest, prepare_chat, prepare_completion
from tllama.native_api import NativeAPI, ReplyContext, ollama_response, openai_response, ollama_stream, openai_stream
from tllama.worker import WorkerError, WorkerRequestError

# Reuse the existing socket/process fixture without changing adapter/lifecycle tests.
from test_native_worker import factory, records, wait_marker


def prepared(*, raw=False, stub=None):
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={"device": "none"}), Path("/models/m.gguf"))
    request = (prepare_completion("raw exact", {"temperature": 0, "num_predict": 10}, resolved=resolved) if raw
               else prepare_chat([{"role": "user", "content": "hello"}], {"temperature": 0, "num_predict": 10},
                                 resolved=resolved, think=False))
    return NativeRequest(request.kind, {**request.body(), **({"stub": stub} if stub else {})})


@pytest.mark.parametrize("raw", [False, True])
async def test_real_transport_result_and_stream_have_matching_content_and_measurements(factory, raw):
    worker, root = factory()
    await worker.start()
    api = NativeAPI(worker, prepared(raw=raw))
    result = await api.result()
    async with api.stream() as events:
        streamed = [event async for event in events]
    assert "".join(e.content for e in streamed) == result.content == "Ahoj světe"
    assert "".join(e.reasoning for e in streamed) == result.reasoning == ("" if raw else "myšlenka")
    assert streamed[-1].stats == result.stats
    assert streamed[-1].stats.input_tokens == 25 and streamed[-1].stats.evaluated_tokens == 6
    assert streamed[-1].stats.cached_tokens == 19
    requests = [r for r in records(root, "requests") if r["path"] not in {"/health", "/props"}]
    assert len(requests) == 2 and all(r["body"]["n_predict"] == 10 for r in requests)
    if not raw:
        assert requests[-1]["body"]["stream_options"]["include_usage"] is True
        assert requests[-1]["body"]["reasoning_format"] == "deepseek"
        assert requests[-1]["body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert all(r["body"]["model"] == "tllama-worker" for r in requests)
    assert all("x-conversation-id" not in r["header_names"] for r in requests)


@pytest.mark.parametrize("api_name", ["ollama", "openai"])
async def test_wire_serialization_uses_public_model_and_no_internal_timings(factory, api_name):
    worker, _ = factory()
    await worker.start()
    api = NativeAPI(worker, prepared())
    context = ReplyContext("public/reference")
    response = await api.result()
    serializer = ollama_response if api_name == "ollama" else openai_response
    assert serializer(response, context)["model"] == "public/reference"
    async with api.stream() as events:
        serializer = ollama_stream if api_name == "ollama" else openai_stream
        kwargs = {"include_usage": True} if api_name == "openai" else {}
        frames = [frame async for frame in serializer(events, context, **kwargs)]
    assert "tllama-worker" not in "".join(frames) and '"timings"' not in "".join(frames)
    if api_name == "ollama":
        final = json.loads(frames[-1])
        assert final["done"] and final["prompt_eval_count"] == 6 and final["prompt_eval_cached_count"] == 19
    else:
        assert frames[-1] == "data: [DONE]\n\n"
        usage = json.loads(frames[-2][6:])["usage"]
        assert usage["prompt_tokens"] == 25 and usage["prompt_tokens_details"]["cached_tokens"] == 19


@pytest.mark.parametrize("mode", ["error-stream", "bad-json", "truncated", "finish-without-done"])
@pytest.mark.parametrize("serializer", [ollama_stream, openai_stream])
async def test_transport_failure_cannot_emit_success_even_after_native_finish(factory, mode, serializer):
    worker, root = factory()
    await worker.start()
    api = NativeAPI(worker, prepared(stub=mode))
    async with api.stream() as events:
        frames = [frame async for frame in serializer(events, ReplyContext("m"))]
    assert '"error"' in frames[-1]
    assert not any('[DONE]' in f or '"done": true' in f for f in frames)
    inference = [r for r in records(root, "requests") if r["path"] == "/v1/chat/completions"]
    assert len(inference) == 1  # Never replay after a partial response.
    assert (await NativeAPI(worker, prepared()).result()).content == "Ahoj světe"


async def test_pre_header_native_http_error_stays_typed(factory):
    worker, _ = factory()
    await worker.start()
    with pytest.raises(WorkerRequestError) as raised:
        await NativeAPI(worker, prepared(stub="error")).result()
    assert raised.value.status_code == 400


async def test_crash_does_not_restart_or_replay_in_adapter(factory):
    worker, root = factory()
    await worker.start()
    with pytest.raises(WorkerError):
        await NativeAPI(worker, prepared(stub="crash")).result()
    assert await asyncio.wait_for(worker.wait_for_exit(), timeout=3) == 71
    requests = [r for r in records(root, "requests") if r["path"] == "/v1/chat/completions"]
    assert len(requests) == 1 and len(records(root, "launch")) == 1


async def test_breaking_public_stream_context_closes_upstream_and_reuses_worker(factory):
    worker, root = factory()
    await worker.start()
    async with NativeAPI(worker, prepared(stub="hold")).stream() as events:
        frames = ollama_stream(events, ReplyContext("m"))
        assert "Ahoj" in await anext(frames)
        await frames.aclose()
    await wait_marker(root, "cancelled")
    assert (await NativeAPI(worker, prepared()).result()).content == "Ahoj světe"


@pytest.mark.parametrize("stream", [False, True])
async def test_client_cancellation_unblocks_upstream_and_preserves_resident_worker(factory, stream):
    worker, root = factory()
    await worker.start()

    async def call():
        api = NativeAPI(worker, prepared(stub="hold"))
        if stream:
            async with api.stream() as events:
                async for _ in ollama_stream(events, ReplyContext("m")):
                    pass
        else:
            await api.result()

    task = asyncio.create_task(call())
    await wait_marker(root, "started")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_marker(root, "cancelled")
    assert worker.process.returncode is None
    assert (await NativeAPI(worker, prepared()).result()).content == "Ahoj světe"
