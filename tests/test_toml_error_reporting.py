"""A malformed virtual-model .toml must be reported in the shape the
surface it was asked through uses, not as a bare unhandled 500.

TomlModelError matched neither registered handler, so it reached FastAPI
unhandled and the client got an unformatted traceback response that no
Ollama or OpenAI client can parse. It also subclasses ValueError, which
meant the endpoints that catch ValueError themselves turned it into a 400
-- telling the client to fix its request, which cannot help, when the
problem is a file on the server.

The scenario is routine rather than exotic: a .toml is an ordinary file a
person edits in place, so it can become invalid at any moment, including
while a model built from it is loaded and serving.
"""

import json

import pytest

from tllama.backend import model_manager


BROKEN_TOML = '[llm]\nmodel = "Local/whatever.gguf"\nthis line is not valid toml\n'

SEMANTICALLY_BROKEN_TOML = '[llm]\nmodel = "Local/a.gguf"\nfrom = "/outside/b.gguf"\n'


@pytest.fixture
def broken_toml_model():
    """Give the running application a model whose .toml does not parse.

    Written into the live model_manager's store rather than a fixture
    manager, because the point is to go through the real application the
    way a client does.
    """
    path = model_manager.local_models_dir / "broken.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BROKEN_TOML, encoding="utf-8")
    try:
        yield "broken"
    finally:
        path.unlink(missing_ok=True)
        model_manager._invalidate_metadata_cache_entry("broken")


@pytest.fixture
def contradictory_toml_model():
    """A .toml that parses as TOML but is rejected on its own terms."""
    path = model_manager.local_models_dir / "contradictory.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SEMANTICALLY_BROKEN_TOML, encoding="utf-8")
    try:
        yield "contradictory"
    finally:
        path.unlink(missing_ok=True)
        model_manager._invalidate_metadata_cache_entry("contradictory")


class TestOllamaSurface:
    def test_show_reports_a_broken_toml_as_a_server_error(self, client, broken_toml_model):
        response = client.post("/api/show", json={"model": broken_toml_model})

        assert response.status_code == 500
        assert list(response.json()) == ["error"]

    def test_the_message_is_a_flat_string_an_ollama_client_can_read(
        self, client, broken_toml_model
    ):
        response = client.post("/api/show", json={"model": broken_toml_model})
        body = json.loads(response.text)

        assert isinstance(body["error"], str)

    def test_the_message_names_the_offending_file(self, client, broken_toml_model):
        response = client.post("/api/show", json={"model": broken_toml_model})

        assert "broken.toml" in response.json()["error"]

    def test_chat_does_not_flatten_it_into_a_client_error(
        self, client, broken_toml_model
    ):
        """The broad `except Exception` around get_model() reports every
        load failure as 400. A broken .toml is not a load failure and must
        pass through it."""
        response = client.post(
            "/api/chat",
            json={
                "model": broken_toml_model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 500

    def test_a_semantic_rejection_is_reported_the_same_way(
        self, client, contradictory_toml_model
    ):
        response = client.post("/api/show", json={"model": contradictory_toml_model})

        assert response.status_code == 500
        assert list(response.json()) == ["error"]


class TestOpenAiSurface:
    def test_chat_completions_uses_the_openai_error_shape(
        self, client, broken_toml_model
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": broken_toml_model,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = response.json()

        assert response.status_code == 500
        assert set(body["error"]) == {"message", "type", "param", "code"}

    def test_the_error_type_says_server_error_not_invalid_request(
        self, client, broken_toml_model
    ):
        """The official OpenAI client picks its exception class from the
        status, and a caller that retries on 5xx but not on 4xx must see
        this as the server's problem."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": broken_toml_model,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.json()["error"]["type"] == "server_error"


class TestNoRegression:
    def test_a_model_that_simply_does_not_exist_is_still_a_404(self, client):
        response = client.post("/api/show", json={"model": "no-such-model-at-all"})

        assert response.status_code == 404
