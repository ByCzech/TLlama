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

from llama_cpp.llama_chat_format import MTMDChatHandler

from tllama.helpers.openai_compat import build_openai_chat_messages
from tllama.helpers.vision import InvalidImageError, NO_MULTIMODAL_SUPPORT
from tllama.main import app


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
PNG_URL = f"data:image/png;base64,{PNG}"


class FakeLlm:
    def __init__(self, chat_handler=None):
        self.chat_handler = chat_handler


class FakeProjectorHandler(MTMDChatHandler):
    def __init__(self):
        self.mtmd_ctx = object()


class FakeMessage:
    def __init__(self, role="user", content=""):
        self.role = role
        self.content = content


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
def loaded(monkeypatch):
    def factory(chat_handler=None):
        from tllama.routers import openai as openai_router

        llm = FakeLlm(chat_handler=chat_handler)

        async def fake_get_model(*args, **kwargs):
            return llm

        async def fake_metadata(*args, **kwargs):
            return {}

        monkeypatch.setattr(openai_router.model_manager, "get_model", fake_get_model)
        monkeypatch.setattr(
            openai_router.model_manager, "get_model_metadata", fake_metadata
        )
        return llm

    return factory


class TestImagePartsSurviveTheTrip:
    def test_the_content_stays_a_list(self):
        [message] = build_openai_chat_messages(FakeRequest([image_message()]))

        assert isinstance(message["content"], list)

    def test_the_image_part_is_still_there(self):
        [message] = build_openai_chat_messages(FakeRequest([image_message()]))

        assert message["content"][1] == {
            "type": "image_url",
            "image_url": {"url": PNG_URL},
        }

    def test_the_text_part_is_kept_alongside_it(self):
        [message] = build_openai_chat_messages(FakeRequest([image_message()]))

        assert message["content"][0] == {"type": "text", "text": "co je na obrazku"}

    def test_bare_base64_in_the_url_field_gets_a_prefix(self):
        [message] = build_openai_chat_messages(FakeRequest([image_message(url=PNG)]))

        assert message["content"][1]["image_url"]["url"] == PNG_URL

    def test_a_string_image_url_is_accepted_too(self):
        # llama-cpp-python's own get_image_urls tolerates this shape.
        message = FakeMessage(content=[{"type": "image_url", "image_url": PNG_URL}])

        [built] = build_openai_chat_messages(FakeRequest([message]))

        assert built["content"][0]["image_url"] == {"url": PNG_URL}


class TestWhatIsStillFlattened:
    def test_a_text_only_parts_list_becomes_a_string(self):
        message = FakeMessage(content=[
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ])

        [built] = build_openai_chat_messages(FakeRequest([message]))

        assert built["content"] == "ab"

    def test_a_plain_string_is_untouched(self):
        [built] = build_openai_chat_messages(
            FakeRequest([FakeMessage(content="ahoj")])
        )

        assert built["content"] == "ahoj"


class TestRemoteUrlsAreRefused:
    @pytest.mark.parametrize("url", [
        "https://example.com/cat.png",
        "http://127.0.0.1:9/cat.png",
        "file:///etc/passwd",
    ])
    def test_a_url_never_becomes_a_fetch(self, url):
        with pytest.raises(InvalidImageError):
            build_openai_chat_messages(FakeRequest([image_message(url=url)]))

    def test_a_part_without_a_url_is_refused(self):
        message = FakeMessage(content=[{"type": "image_url", "image_url": {}}])

        with pytest.raises(InvalidImageError):
            build_openai_chat_messages(FakeRequest([message]))


class TestTheEndpoint:
    def test_an_image_to_a_text_model_is_refused(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "co je to"},
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ]}],
        })

        assert response.status_code == 400
        assert NO_MULTIMODAL_SUPPORT in response.text

    def test_a_remote_url_gets_a_400(self, client, loaded):
        loaded(chat_handler=FakeProjectorHandler())

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "http://127.0.0.1:9/cat.png"}},
            ]}],
        })

        assert response.status_code == 400
        assert "base64" in response.text

    def test_a_model_with_a_projector_is_not_refused(self, client, loaded):
        loaded(chat_handler=FakeProjectorHandler())

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ]}],
        })

        assert response.status_code != 400

    def test_text_alone_is_untouched(self, client, loaded):
        loaded(chat_handler=None)

        response = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "ahoj"}],
        })

        assert response.status_code != 400
