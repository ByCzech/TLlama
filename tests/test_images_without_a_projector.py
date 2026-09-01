"""An image a model cannot use is refused, not quietly dropped.

Dropping it is what llama-cpp-python's own server does, and the result is
a model answering about a picture it never received: measured against a
text-only model, a request carrying an image came back with a confident
description and a prompt_eval_count that accounted for the text alone.
Real Ollama refuses the request instead, which is the behaviour copied
here down to the wording.
"""

import pytest
from fastapi.testclient import TestClient

from llama_cpp.llama_chat_format import MTMDChatHandler

from tllama.backend import model_has_projector
from tllama.helpers.vision import NO_MULTIMODAL_SUPPORT
from tllama.main import app


ONE_PIXEL = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
    "BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeLlm:
    def __init__(self, chat_handler=None):
        self.chat_handler = chat_handler


class FakeProjectorHandler(MTMDChatHandler):
    def __init__(self):
        # Deliberately not calling super().__init__: it wants a real file
        # on disk, and nothing here loads one.
        self.mtmd_ctx = object()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded(monkeypatch):
    """Put a model behind the endpoints without loading anything."""
    def factory(chat_handler=None):
        from tllama.routers import ollama as ollama_router

        llm = FakeLlm(chat_handler=chat_handler)

        async def fake_get_model(*args, **kwargs):
            return llm

        async def fake_metadata(*args, **kwargs):
            return {}

        monkeypatch.setattr(
            ollama_router.model_manager, "get_model", fake_get_model
        )
        monkeypatch.setattr(
            ollama_router.model_manager, "get_model_metadata", fake_metadata
        )
        return llm

    return factory


class TestWhatCountsAsAbleToSee:
    def test_a_projector_handler_counts(self):
        assert model_has_projector(FakeLlm(chat_handler=FakeProjectorHandler()))

    def test_no_handler_at_all_does_not(self):
        assert not model_has_projector(FakeLlm(chat_handler=None))

    def test_a_plain_text_handler_does_not(self):
        # What a [template] override leaves behind: an opaque callable.
        assert not model_has_projector(FakeLlm(chat_handler=lambda **kw: None))


class TestChatRefusesAnImageItCannotUse:
    def test_an_image_to_a_text_model_is_refused(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert response.status_code == 400

    def test_the_refusal_says_why(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert NO_MULTIMODAL_SUPPORT in response.json()["error"]

    def test_an_image_on_a_later_message_is_found_too(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "ahoj"},
                {"role": "assistant", "content": "zdravim"},
                {"role": "user", "content": "a co tohle", "images": [ONE_PIXEL]},
            ],
        })

        assert response.status_code == 400

    def test_an_empty_images_list_is_not_an_image(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "ahoj", "images": []}],
        })

        # Past the guard. What it fails on afterwards is inference against
        # a model that cannot run, which is not what this is measuring.
        assert response.status_code != 400

    def test_text_alone_is_untouched(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "ahoj"}],
        })

        assert response.status_code != 400

    def test_a_model_with_a_projector_is_not_refused(self, client, loaded):
        loaded(chat_handler=FakeProjectorHandler())

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert response.status_code != 400


class TestGenerateRefusesImagesForNow:
    def test_an_image_is_refused(self, client, loaded):
        loaded(chat_handler=FakeProjectorHandler())

        response = client.post("/api/generate", json={
            "model": "m",
            "stream": False,
            "prompt": "co je to",
            "images": [ONE_PIXEL],
        })

        assert response.status_code == 400
        assert NO_MULTIMODAL_SUPPORT in response.json()["error"]

    def test_a_prompt_without_images_is_untouched(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/api/generate", json={
            "model": "m",
            "stream": False,
            "prompt": "ahoj",
        })

        assert response.status_code != 400
