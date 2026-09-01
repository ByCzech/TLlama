"""Token counts a request carrying an image reports.

MTMDChatHandler evaluates the picture through mtmd and only then hands
create_completion llama.input_ids, so the image's tokens never reach the
usage it returns. Measured against a real vision model: 39 reported for a
request llama.cpp's own counters put at 179, the 140 missing being the
image. The client is told the picture was nearly free.

The context's counters do count it, which is what llama.cpp prints as
"prompt eval time / N tokens". The streaming paths read them already;
these cover the non-streaming siblings that settled for usage.
"""

import pytest
from fastapi.testclient import TestClient

from tllama.main import app


class FakeLlm:
    chat_handler = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def answered(monkeypatch):
    """Answer a request with fixed usage and fixed context counters."""
    def factory(usage, counters):
        from tllama.routers import ollama as ollama_router
        from tllama.routers import openai as openai_router

        llm = FakeLlm()

        async def fake_get_model(*args, **kwargs):
            return llm

        async def fake_metadata(*args, **kwargs):
            return {}

        def fake_completion(*args, **kwargs):
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "text": "ok",
                    "finish_reason": "stop",
                }],
                "usage": usage,
            }

        for router in (ollama_router, openai_router):
            monkeypatch.setattr(router.model_manager, "get_model", fake_get_model)
            monkeypatch.setattr(
                router.model_manager, "get_model_metadata", fake_metadata
            )
            monkeypatch.setattr(router, "create_chat_completion_ex", fake_completion)
            monkeypatch.setattr(router, "reset_eval_counters", lambda llm: True)
            monkeypatch.setattr(
                router, "counted_for_request", lambda was_reset, llm: counters
            )

        return llm

    return factory


CHAT_BODY = {
    "model": "m",
    "stream": False,
    "messages": [{"role": "user", "content": "co je na obrazku"}],
}

OPENAI_BODY = {
    "model": "m",
    "messages": [{"role": "user", "content": "co je na obrazku"}],
}


class TestOllamaChat:
    def test_the_counted_prompt_wins_over_usage(self, client, answered):
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1217},
                 counters=(179, 1217))

        body = client.post("/api/chat", json=CHAT_BODY).json()

        assert body["prompt_eval_count"] == 179

    def test_the_counted_completion_is_used_too(self, client, answered):
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1000},
                 counters=(179, 1217))

        body = client.post("/api/chat", json=CHAT_BODY).json()

        assert body["eval_count"] == 1217

    def test_usage_still_answers_when_the_counters_cannot(
        self, client, answered
    ):
        # Without a reset the reading spans earlier requests, so
        # counted_for_request declines rather than guessing.
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1217},
                 counters=(None, None))

        body = client.post("/api/chat", json=CHAT_BODY).json()

        assert body["prompt_eval_count"] == 39
        assert body["eval_count"] == 1217


class TestOpenAiChat:
    def test_the_counted_prompt_wins_over_usage(self, client, answered):
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1217},
                 counters=(179, 1217))

        body = client.post("/v1/chat/completions", json=OPENAI_BODY).json()

        assert body["usage"]["prompt_tokens"] == 179

    def test_the_total_adds_up_to_what_was_reported(self, client, answered):
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1217},
                 counters=(179, 1217))

        usage = client.post("/v1/chat/completions", json=OPENAI_BODY).json()["usage"]

        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_usage_still_answers_when_the_counters_cannot(self, client, answered):
        answered(usage={"prompt_tokens": 39, "completion_tokens": 1217},
                 counters=(None, None))

        body = client.post("/v1/chat/completions", json=OPENAI_BODY).json()

        assert body["usage"]["prompt_tokens"] == 39
