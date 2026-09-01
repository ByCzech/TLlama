"""A non-streaming inference failure must be reported like every other
error on its surface, not as a bare unhandled 500.

The streaming branches of /api/chat and /api/generate both catch and emit
an error line, because by then the status code has already gone out and a
line in the body is all that is left. Their non-streaming siblings had
try/finally with no except at all, so anything raised out of llama.cpp
escaped to FastAPI unhandled and the client got an unformatted response
that no Ollama client can parse.

Nothing has been written to the response yet in the non-streaming case, so
a status code is still available and is the right way to say it.
"""

import pytest

from tllama.backend import model_manager
from tllama.routers import ollama as ollama_router


class ExplodingLlama:
    """Stands in for a loaded model whose generation fails.

    Only the surface /api/chat and /api/generate actually touch is
    implemented; the failure is the point, not the arithmetic.
    """

    def create_completion(self, *args, **kwargs):
        raise RuntimeError("llama_decode failed")

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def tokenize(self, *args, **kwargs):
        return [1, 2, 3]

    def n_ctx(self):
        return 4096


@pytest.fixture
def a_model_that_fails_to_generate(monkeypatch):
    """A model that loads fine and then blows up during generation."""

    async def fake_get_model(model_name, *args, **kwargs):
        return ExplodingLlama()

    async def fake_get_model_metadata(model_name, *args, **kwargs):
        return {}

    def fake_chat_completion(*args, **kwargs):
        raise RuntimeError("llama_decode failed")

    monkeypatch.setattr(model_manager, "get_model", fake_get_model)
    monkeypatch.setattr(model_manager, "get_model_metadata", fake_get_model_metadata)
    monkeypatch.setattr(ollama_router, "create_chat_completion_ex", fake_chat_completion)

    return "exploding"


class TestChat:
    def test_the_failure_is_a_server_error_not_an_unhandled_one(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/api/chat",
            json={
                "model": a_model_that_fails_to_generate,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 500

    def test_the_body_is_the_shape_an_ollama_client_reads(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/api/chat",
            json={
                "model": a_model_that_fails_to_generate,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        body = response.json()

        assert list(body) == ["error"]
        assert isinstance(body["error"], str)

    def test_the_underlying_reason_is_not_swallowed(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/api/chat",
            json={
                "model": a_model_that_fails_to_generate,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert "llama_decode failed" in response.json()["error"]


class TestGenerate:
    def test_the_failure_is_a_server_error_not_an_unhandled_one(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/api/generate",
            json={
                "model": a_model_that_fails_to_generate,
                "prompt": "hi",
                "raw": True,
                "stream": False,
            },
        )

        assert response.status_code == 500

    def test_the_body_is_the_shape_an_ollama_client_reads(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/api/generate",
            json={
                "model": a_model_that_fails_to_generate,
                "prompt": "hi",
                "raw": True,
                "stream": False,
            },
        )
        body = response.json()

        assert list(body) == ["error"]
        assert isinstance(body["error"], str)


class TestNoRegression:
    def test_a_bad_request_is_still_a_client_error(
        self, client, a_model_that_fails_to_generate
    ):
        """raw mode with a template is rejected before generation is ever
        reached, and must stay a 400."""
        response = client.post(
            "/api/generate",
            json={
                "model": a_model_that_fails_to_generate,
                "prompt": "hi",
                "raw": True,
                "template": "{{ .Prompt }}",
                "stream": False,
            },
        )

        assert response.status_code == 400
