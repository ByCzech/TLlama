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
from tllama.routers import openai as openai_router


@pytest.fixture
def a_model_that_fails_to_generate(native_reply):
    state = native_reply()
    state.worker.error = RuntimeError("llama_decode failed")
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


class TestOpenAiSurface:
    """The same gap existed on /v1/chat/completions, where the streaming
    branch catches and emits an error frame and the non-streaming one had
    no try around the call at all.
    """

    def test_the_failure_is_a_server_error_not_an_unhandled_one(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": a_model_that_fails_to_generate,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 500

    def test_the_body_is_the_openai_error_object(
        self, client, a_model_that_fails_to_generate
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": a_model_that_fails_to_generate,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = response.json()

        assert set(body["error"]) == {"message", "type", "param", "code"}
        assert body["error"]["type"] == "server_error"

    def test_a_broken_toml_is_still_reported_as_such(
        self, client, native_reply
    ):
        """The new except must not swallow the TomlModelError passthrough
        added alongside it: that one is raised before generation, from
        get_model, and still belongs to the registered handler.
        """
        from tllama.helpers.model_toml import TomlModelError

        state = native_reply()
        state.load_error = TomlModelError("Local/x.toml: invalid TOML")

        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 500
        assert "invalid TOML" in response.json()["error"]["message"]
