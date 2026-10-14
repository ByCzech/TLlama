"""The durations an Ollama reply carries beside its token counts.

`ollama run --verbose` prints tokens per second, which it works out by
dividing a count by a duration. TLlama sent the counts and only
total_duration, so the client had nothing to divide and printed bare
counts -- against real Ollama on the same machine it printed rates.

llama.cpp keeps the figures in llama_perf_context_data and
llama-cpp-python binds the call. _ext.py already reads that struct for
n_p_eval and n_eval; t_p_eval_ms and t_eval_ms are two more fields of the
same reading. They mean one request because reset_eval_counters() zeroes
them first, exactly as the counts rely on.

t_load_ms is in the struct too and is not used. perf_reset() leaves it
alone, so it holds the time the model took to load and would be sent
again on every request against that context for as long as it lives.
"""

import pytest
from fastapi.testclient import TestClient

from tllama._ext import timed_for_request
from tllama.main import app


class FakePerf:
    """llama_perf_context_data, as far as anything here reads it."""

    def __init__(self, t_p_eval_ms, t_eval_ms, t_load_ms=98765.0):
        self.t_p_eval_ms = t_p_eval_ms
        self.t_eval_ms = t_eval_ms
        self.t_load_ms = t_load_ms
        self.n_p_eval = 15
        self.n_eval = 542


class TestReadingTheCounters:
    def test_milliseconds_become_nanoseconds(self, monkeypatch):
        """Every other duration in an Ollama reply is in nanoseconds."""
        monkeypatch.setattr(
            "tllama._ext._perf_context_data",
            lambda llm: FakePerf(t_p_eval_ms=172.90, t_eval_ms=5192.20),
        )

        prompt, generation = timed_for_request(True, object())

        assert prompt == 172_900_000
        assert generation == 5_192_200_000

    def test_nothing_is_reported_without_a_reset(self, monkeypatch):
        """The reading would span an unknown number of earlier requests.

        A rate computed from that is wrong in a way a client cannot see,
        which is worse than the field being absent.
        """
        monkeypatch.setattr(
            "tllama._ext._perf_context_data",
            lambda llm: FakePerf(t_p_eval_ms=172.90, t_eval_ms=5192.20),
        )

        assert timed_for_request(False, object()) == (None, None)

    def test_the_load_timer_is_not_one_of_them(self, monkeypatch):
        """It survives perf_reset() and describes the context, not the request."""
        monkeypatch.setattr(
            "tllama._ext._perf_context_data",
            lambda llm: FakePerf(t_p_eval_ms=1.0, t_eval_ms=2.0, t_load_ms=98765.0),
        )

        assert timed_for_request(True, object()) == (1_000_000, 2_000_000)

    def test_a_context_that_cannot_be_read_reports_nothing(self, monkeypatch):
        monkeypatch.setattr("tllama._ext._perf_context_data", lambda llm: None)

        assert timed_for_request(True, object()) == (None, None)


SIMPLE_TEMPLATE = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"


class FakeLlm:
    """Enough of a model for both endpoints to reach their reply."""

    chat_handler = None

    def token_bos(self):
        return -1

    def token_eos(self):
        return -1

    def tokenize(self, text, add_bos=False, special=False):
        return list(range(15))

    def create_completion(self, *args, **kwargs):
        return {
            "choices": [{"text": "ok", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 542},
        }


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def answered(monkeypatch):
    """Answer a request with fixed timings from the context."""
    def factory(timings=(172_900_000, 5_192_200_000)):
        from tllama.routers import ollama as ollama_router

        llm = FakeLlm()

        async def fake_get_model(*args, **kwargs):
            return llm

        async def fake_metadata(*args, **kwargs):
            return {"template": SIMPLE_TEMPLATE}

        def fake_completion(*args, **kwargs):
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "text": "ok",
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 15, "completion_tokens": 542},
            }

        monkeypatch.setattr(ollama_router.model_manager, "get_model", fake_get_model)
        monkeypatch.setattr(
            ollama_router.model_manager, "get_model_metadata", fake_metadata
        )
        monkeypatch.setattr(ollama_router, "create_chat_completion_ex", fake_completion)
        monkeypatch.setattr(ollama_router, "reset_eval_counters", lambda llm: True)
        monkeypatch.setattr(
            ollama_router, "counted_for_request", lambda was_reset, llm: (15, 542)
        )
        monkeypatch.setattr(
            ollama_router, "timed_for_request", lambda was_reset, llm: timings
        )

        return llm

    return factory


CHAT = {"model": "m", "stream": False,
        "messages": [{"role": "user", "content": "ahoj"}]}
GENERATE = {"model": "m", "stream": False, "prompt": "ahoj"}


class TestWhatTheReplyCarries:
    @pytest.mark.parametrize("path,body", [("/api/chat", CHAT),
                                           ("/api/generate", GENERATE)])
    def test_the_client_can_work_out_a_rate(self, client, answered, path, body):
        answered()

        reply = client.post(path, json=body).json()

        assert reply["prompt_eval_duration"] == 172_900_000
        assert reply["eval_duration"] == 5_192_200_000

    @pytest.mark.parametrize("path,body", [("/api/chat", CHAT),
                                           ("/api/generate", GENERATE)])
    def test_the_load_is_timed_too(self, client, answered, path, body):
        """Measured around fetching the model, whether or not one loaded.

        A resident model still costs a lookup and Ollama reports that
        cost rather than a zero.
        """
        answered()

        reply = client.post(path, json=body).json()

        assert reply["load_duration"] >= 0

    @pytest.mark.parametrize("path,body", [("/api/chat", CHAT),
                                           ("/api/generate", GENERATE)])
    def test_the_pieces_fit_inside_the_total(
        self, client, answered, path, body, monkeypatch
    ):
        """total_duration used to start after the model was fetched.

        A client adding load_duration to the evaluation times expects
        them to fit within the total, and on a cold start they did not:
        the load was reported and then excluded from the figure it was
        supposed to be part of. The delay here stands in for that load,
        which is otherwise too quick for the difference to show.
        """
        answered(timings=(1_000, 2_000))

        from tllama.routers import ollama as ollama_router
        real_get_model = ollama_router.model_manager.get_model

        async def slow_get_model(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.05)
            return await real_get_model(*args, **kwargs)

        monkeypatch.setattr(
            ollama_router.model_manager, "get_model", slow_get_model
        )

        reply = client.post(path, json=body).json()

        assert reply["load_duration"] >= 50_000_000
        assert reply["total_duration"] >= reply["load_duration"]

    @pytest.mark.parametrize("path,body", [("/api/chat", CHAT),
                                           ("/api/generate", GENERATE)])
    def test_an_unreadable_context_leaves_them_absent_not_wrong(
        self, client, answered, path, body
    ):
        answered(timings=(None, None))

        reply = client.post(path, json=body).json()

        assert reply["prompt_eval_duration"] is None
        assert reply["eval_duration"] is None


async def test_waiting_generate_does_not_reset_running_inference(
    manager, monkeypatch
):
    import asyncio
    import threading
    from contextlib import asynccontextmanager

    from tllama.routers import ollama as router
    from tllama.schemas.ollama import OllamaGenerateRequest

    started = asyncio.Event()
    queued = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    events = []

    class BlockingLlm(FakeLlm):
        def create_completion(self, *, prompt, **kwargs):
            events.append(f"{prompt}:start")
            if prompt == "first":
                loop.call_soon_threadsafe(started.set)
                assert release.wait(timeout=5)
            events.append(f"{prompt}:end")
            return super().create_completion()

    llm = BlockingLlm()

    async def get_model(*args, **kwargs):
        return llm

    async def metadata(*args, **kwargs):
        return {}

    original_slot = manager.acquire_inference_slot

    @asynccontextmanager
    async def tracked_slot(name):
        if started.is_set():
            queued.set()
        async with original_slot(name):
            yield

    def reset(model):
        events.append("reset")
        return True

    monkeypatch.setattr(router, "model_manager", manager)
    monkeypatch.setattr(manager, "get_model", get_model)
    monkeypatch.setattr(manager, "get_model_metadata", metadata)
    monkeypatch.setattr(manager, "acquire_inference_slot", tracked_slot)
    monkeypatch.setattr(router, "reset_eval_counters", reset)
    monkeypatch.setattr(router, "timed_for_request", lambda *args: (1, 2))

    def request(prompt):
        return OllamaGenerateRequest(
            model="m", prompt=prompt, raw=True, stream=False
        )

    first = asyncio.create_task(router.ollama_generate(request("first")))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(router.ollama_generate(request("second")))
        await asyncio.wait_for(queued.wait(), timeout=2)
        assert events == ["reset", "first:start"]
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []))

    assert events == [
        "reset", "first:start", "first:end",
        "reset", "second:start", "second:end",
    ]
