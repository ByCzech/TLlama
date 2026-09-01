"""Ollama's images field, converted into what the handlers actually read.

Ollama puts bare base64 in message.images. MTMDChatHandler looks for
OpenAI image_url content parts and finds them only on user messages.
Between the two, the image used to be dropped: measured against a real
vision model, a request carrying a picture came back with
prompt_eval_count 13 -- the text and nothing else.

The prefix is not decoration. llama-cpp-python's _load_image treats
anything that is not a data: URL as something to fetch with
urllib.request.urlopen, in the server's process.
"""

import base64

import pytest

from tllama.helpers.chat import normalize_chat_messages
from tllama.helpers.vision import (
    InvalidImageError,
    NO_REMOTE_IMAGES,
    build_multimodal_content,
    messages_carry_images,
    to_image_data_url,
)


PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
).decode()
JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()
GIF = base64.b64encode(b"GIF89a" + b"\x00" * 32).decode()
WEBP = base64.b64encode(
    b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"VP8 " + b"\x00" * 16
).decode()
BMP = base64.b64encode(b"BM" + b"\x00" * 32).decode()
TIFF = base64.b64encode(b"II\x2a\x00" + b"\x00" * 32).decode()


class FakeMessage:
    def __init__(self, role="user", content="", images=None):
        self.role = role
        self.content = content
        self.images = images
        self.thinking = None
        self.tool_calls = None
        self.tool_name = None
        self.tool_call_id = None


class TestTheDataUrlPrefix:
    def test_bare_base64_gets_a_prefix(self):
        assert to_image_data_url(PNG) == f"data:image/png;base64,{PNG}"

    def test_the_payload_is_left_byte_for_byte(self):
        # Re-encoding would be a chance to corrupt it for no reason.
        assert to_image_data_url(PNG).split(",", 1)[1] == PNG

    @pytest.mark.parametrize("payload,expected", [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (WEBP, "image/webp"),
        # Neither of these was recognised while the media type came from a
        # hand-written table of four formats; the library knows a good
        # many more, and adding one is no longer a code change.
        (BMP, "image/bmp"),
        (TIFF, "image/tiff"),
    ])
    def test_the_label_is_read_from_the_bytes(self, payload, expected):
        assert to_image_data_url(payload).startswith(f"data:{expected};base64,")

    def test_an_existing_prefix_is_not_added_twice(self):
        already = f"data:image/png;base64,{PNG}"

        # Wrapping it again gives a string with two commas, and
        # _load_image's split(",")[1] would then hand the decoder the
        # middle of it.
        assert to_image_data_url(already) == already


class TestWhatIsRefused:
    @pytest.mark.parametrize("url", [
        "http://example.com/cat.png",
        "https://example.com/cat.png",
        "file:///etc/passwd",
        "ftp://example.com/cat.png",
    ])
    def test_a_url_is_refused_rather_than_fetched(self, url):
        with pytest.raises(InvalidImageError) as raised:
            to_image_data_url(url)

        assert NO_REMOTE_IMAGES in str(raised.value)

    def test_undecodable_base64_is_refused(self):
        with pytest.raises(InvalidImageError):
            to_image_data_url("not base64 at all!!")

    def test_base64_of_something_that_is_not_an_image_is_refused(self):
        with pytest.raises(InvalidImageError):
            to_image_data_url(base64.b64encode(b"just some text").decode())

    def test_a_data_url_with_two_commas_is_refused(self):
        with pytest.raises(InvalidImageError):
            to_image_data_url(f"data:image/png;base64,{PNG},extra")


class TestTheContentParts:
    def test_the_text_comes_first(self):
        parts = build_multimodal_content("co to je", [PNG])

        assert parts[0] == {"type": "text", "text": "co to je"}

    def test_the_image_is_an_openai_image_url_part(self):
        parts = build_multimodal_content("co to je", [PNG])

        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG}"},
        }

    def test_every_image_gets_its_own_part(self):
        parts = build_multimodal_content("dva", [PNG, JPEG])

        assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]


class TestTheMessageAsTheHandlerSeesIt:
    def test_images_become_content_parts(self):
        [message] = normalize_chat_messages(
            [FakeMessage(content="co to je", images=[PNG])]
        )

        assert isinstance(message["content"], list)
        assert message["content"][1]["type"] == "image_url"

    def test_the_images_field_does_not_survive(self):
        # It is represented in content now, and a leftover key would be a
        # second copy of the same picture that nothing renders.
        [message] = normalize_chat_messages(
            [FakeMessage(content="co to je", images=[PNG])]
        )

        assert "images" not in message

    def test_a_message_without_images_still_gets_plain_text(self):
        [message] = normalize_chat_messages([FakeMessage(content="ahoj")])

        assert message["content"] == "ahoj"

    def test_openai_parts_sent_to_the_ollama_endpoint_survive(self):
        content = [
            {"type": "text", "text": "co to je"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        ]

        [message] = normalize_chat_messages([FakeMessage(content=content)])

        assert message["content"] == content

    def test_a_plain_text_parts_list_is_still_flattened(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

        [message] = normalize_chat_messages([FakeMessage(content=content)])

        assert message["content"] == "ab"


class TestBothShapesCountAsAnImage:
    def test_the_images_field_counts(self):
        assert messages_carry_images([FakeMessage(content="x", images=[PNG])])

    def test_an_image_url_part_counts_too(self):
        # Otherwise a client using the OpenAI shape against /api/chat
        # walks straight past the guard and into a text-only model.
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        ]

        assert messages_carry_images([FakeMessage(content=content)])

    def test_plain_text_does_not(self):
        assert not messages_carry_images([FakeMessage(content="ahoj")])
