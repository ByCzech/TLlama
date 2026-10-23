"""Images on /v1/chat/completions.

OpenAI carries an image inside the content, as an image_url part; there
is no field beside it the way Ollama has one. build_openai_chat_messages
flattened any list to its text parts, so that was the one route an image
could take and it was closed.

image_url.url is a real URL field in the OpenAI spec, which is the part
that needs care here rather than on the Ollama side: a spec-abiding
client may put https:// in it, and hosted OpenAI would fetch that from
its own infrastructure. Here the fetch would come out of this server.
"""

import base64

import pytest
from fastapi.testclient import TestClient


from tllama.native_requests import normalize_chat_input
from tllama.helpers.vision import InvalidImageError, NO_MULTIMODAL_SUPPORT
from tllama.main import app


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
PNG_URL = f"data:image/png;base64,{PNG}"


class FakeMessage:
    def __init__(self, role="user", content=""):
        self.role = role
        self.content = content

    def model_dump(self, **kwargs):
        return vars(self).copy()


class FakeRequest:
    def __init__(self, messages):
        self.messages = messages


def image_message(url=PNG_URL, text="co je na obrazku"):
    return FakeMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": url}},
    ])


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded(native_reply):
    def factory(vision=False):
        return native_reply(vision=vision)
    return factory


class TestImagePartsSurviveTheTrip:
    def test_the_content_stays_a_list(self):
        [message] = normalize_chat_input([image_message()])

        assert isinstance(message["content"], list)

    def test_the_image_part_is_still_there(self):
        [message] = normalize_chat_input([image_message()])

        assert message["content"][1] == {
            "type": "image_url",
            "image_url": {"url": PNG_URL},
        }

    def test_the_text_part_is_kept_alongside_it(self):
        [message] = normalize_chat_input([image_message()])

        assert message["content"][0] == {"type": "text", "text": "co je na obrazku"}

    def test_bare_base64_in_the_url_field_gets_a_prefix(self):
        [message] = normalize_chat_input([image_message(url=PNG)])

        assert message["content"][1]["image_url"]["url"] == PNG_URL

    def test_a_string_image_url_is_accepted_too(self):
        # llama-cpp-python's own get_image_urls tolerates this shape.
        message = FakeMessage(content=[{"type": "image_url", "image_url": PNG_URL}])

        [built] = normalize_chat_input([message])

        assert built["content"][0]["image_url"] == {"url": PNG_URL}


class TestWhatIsStillFlattened:
    def test_a_text_only_parts_list_becomes_a_string(self):
        message = FakeMessage(content=[
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ])

        [built] = normalize_chat_input([message])

        assert built["content"] == "ab"

    def test_a_plain_string_is_untouched(self):
        [built] = normalize_chat_input([FakeMessage(content="ahoj")])

        assert built["content"] == "ahoj"


class TestRemoteUrlsAreRefused:
    @pytest.mark.parametrize("url", [
        "https://example.com/cat.png",
        "http://127.0.0.1:9/cat.png",
        "file:///etc/passwd",
    ])
    def test_a_url_never_becomes_a_fetch(self, url):
        with pytest.raises(InvalidImageError):
            normalize_chat_input([image_message(url=url)])

    def test_a_part_without_a_url_is_refused(self):
        message = FakeMessage(content=[{"type": "image_url", "image_url": {}}])

        with pytest.raises(InvalidImageError):
            normalize_chat_input([message])


class TestTheEndpoint:
    def test_an_image_to_a_text_model_is_refused(self, client, loaded):
        loaded(vision=False)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "co je to"},
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ]}],
        })

        assert response.status_code == 400
        assert "Multimodal data provided" in response.text

    def test_remote_url_is_refused_at_native_text_checkpoint(self, client, loaded):
        loaded(vision=True)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "http://127.0.0.1:9/cat.png"}},
            ]}],
        })

        assert response.status_code == 400
        assert "Image URLs are not supported" in response.text

    def test_worker_vision_capability_allows_image_inference(self, client, loaded):
        loaded(vision=True)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ]}],
        })

        assert response.status_code == 200

    def test_text_alone_is_untouched(self, client, loaded):
        loaded(vision=False)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "ahoj"}],
        })

        assert response.status_code != 400
