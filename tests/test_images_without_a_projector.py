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


from tllama.helpers.vision import NO_MULTIMODAL_SUPPORT
from tllama.main import app


ONE_PIXEL = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
    "BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded(native_reply):
    def factory(vision=False):
        return native_reply(vision=vision)
    return factory


class TestChatRefusesAnImageItCannotUse:
    def test_an_image_to_a_text_model_is_refused(self, client, loaded):
        loaded(vision=False)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert response.status_code == 400

    def test_the_refusal_says_why(self, client, loaded):
        loaded(vision=False)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert NO_MULTIMODAL_SUPPORT in response.json()["error"]

    def test_an_image_on_a_later_message_is_found_too(self, client, loaded):
        loaded(vision=False)

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
        loaded(vision=False)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "ahoj", "images": []}],
        })

        # Past the guard. What it fails on afterwards is inference against
        # a model that cannot run, which is not what this is measuring.
        assert response.status_code != 400

    def test_text_alone_is_untouched(self, client, loaded):
        loaded(vision=False)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "ahoj"}],
        })

        assert response.status_code != 400

    def test_worker_vision_capability_allows_image_inference(self, client, loaded):
        loaded(vision=True)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [
                {"role": "user", "content": "co je to", "images": [ONE_PIXEL]}
            ],
        })

        assert response.status_code == 200


class TestGenerateRequiresVision:
    @pytest.mark.parametrize("vision", [False, True])
    def test_an_image_uses_actual_worker_capability(self, client, loaded, vision):
        loaded(vision=vision)

        response = client.post("/api/generate", json={
            "model": "m",
            "stream": False,
            "prompt": "co je to",
            "images": [ONE_PIXEL],
        })

        assert response.status_code == (200 if vision else 400)
        if vision:
            assert response.json()["response"] == "ok"
        else:
            assert NO_MULTIMODAL_SUPPORT in response.json()["error"]

    @pytest.mark.parametrize("images", [None, []])
    def test_a_prompt_without_images_is_untouched(self, client, loaded, images):
        loaded(vision=False)

        response = client.post("/api/generate", json={
            "model": "m",
            "stream": False,
            "prompt": "ahoj",
            "images": images,
        })

        assert response.status_code == 200 and response.json()["response"] == "ok"


class TestAnUnusableImageIsAClientError:
    def test_url_is_refused_without_fetch_at_native_text_checkpoint(self, client, loaded):
        loaded(vision=True)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{
                "role": "user",
                "content": "co to je",
                "images": ["http://127.0.0.1:9/nic.png"],
            }],
        })

        assert response.status_code == 400
        assert "Image URLs are not supported" in response.json()["error"]

    def test_undecodable_media_is_refused_at_native_text_checkpoint(self, client, loaded):
        loaded(vision=True)

        response = client.post("/api/chat", json={
            "model": "m",
            "stream": False,
            "messages": [{
                "role": "user",
                "content": "co to je",
                "images": ["nonsense!!"],
            }],
        })

        assert response.status_code == 400
