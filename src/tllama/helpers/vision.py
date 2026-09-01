from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Iterable, Optional

import filetype


# Ollama's own wording, kept verbatim: a client that already knows how to
# read this from real Ollama gets the same sentence from TLlama.
NO_MULTIMODAL_SUPPORT = (
    "Multimodal data provided, but model does not support multimodal requests."
)

# Ollama refuses remote image URLs outright and says so; TLlama does the
# same, and the reason is worth more than parity. llama-cpp-python's
# _load_image hands anything that is not a data: URL to
# urllib.request.urlopen, in the server's own process, with no allowlist,
# no timeout and no size limit -- and urlopen speaks file:// as readily as
# http://. Accepting a URL here would mean fetching whatever a client
# names, from wherever the server can reach, as the server. Only data:
# URLs get through, so that branch is never reached at all.
NO_REMOTE_IMAGES = (
    "Image URLs are not supported, please use base64 encoded data instead."
)

UNRECOGNISED_IMAGE = "Image data is not a recognisable image."

UNDECODABLE_IMAGE = "Image data is not valid base64."

_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


class InvalidImageError(ValueError):
    """An image a client sent cannot be used, and saying so is the answer."""


def _media_type_for(payload: bytes) -> Optional[str]:
    """The media type the bytes themselves say they are.

    Read rather than taken on trust. The label is not what decides how the
    image is read -- mtmd sniffs the real thing -- so a label that
    disagreed with the content would be a lie told for no gain, and bytes
    that are no image at all are worth catching here rather than several
    layers down.

    None when the bytes match nothing filetype knows, which is the signal
    to refuse.
    """
    kind = filetype.image_match(payload)
    if kind is None:
        return None

    return kind.mime


def to_image_data_url(value: str) -> str:
    """Turn one of Ollama's bare base64 images into a data: URL.

    Ollama puts raw base64 in message.images with nothing around it, while
    the handlers downstream read OpenAI's image_url shape. The prefix is
    what bridges the two, and it has to be there: without it the string
    falls through to the URL branch of _load_image and gets fetched.

    A value that already carries a prefix is left alone, so a client that
    sends the OpenAI shape through Ollama's field is not double-wrapped
    into something that decodes to nothing.
    """
    if value.startswith("data:"):
        # _load_image splits on the comma and takes element 1, so a second
        # comma silently hands it the middle of the string instead of the
        # payload. Better to say so than to decode garbage.
        if value.count(",") != 1:
            raise InvalidImageError(
                "Image data URL is malformed (expected exactly one comma)."
            )
        return value

    if _URL_SCHEME.match(value):
        raise InvalidImageError(NO_REMOTE_IMAGES)

    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidImageError(UNDECODABLE_IMAGE)

    media_type = _media_type_for(payload)
    if media_type is None:
        raise InvalidImageError(UNRECOGNISED_IMAGE)

    return f"data:{media_type};base64,{value}"


def build_multimodal_content(text: str, images: Iterable[str]) -> list[dict]:
    """Ollama's split text/images message as OpenAI content parts.

    The text part comes first because that is the order the message was
    written in, and every handler that reads these parts renders them in
    sequence.
    """
    parts: list[dict] = [{"type": "text", "text": text}]

    for image in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": to_image_data_url(image)},
        })

    return parts


def content_carries_images(content: Any) -> bool:
    """Whether a message's content holds OpenAI image parts."""
    if not isinstance(content, list):
        return False

    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def messages_carry_images(messages: Iterable[Any]) -> bool:
    """Whether any message in a chat request came with images attached.

    Both shapes count. A client can use Ollama's images field or send
    OpenAI content parts to the Ollama endpoint, and an image that arrives
    the second way is no less an image than one that arrives the first.
    """
    for message in messages:
        if getattr(message, "images", None):
            return True
        if content_carries_images(getattr(message, "content", None)):
            return True
    return False


def request_carries_images(images: Optional[Iterable[Any]]) -> bool:
    """Whether a generate request came with images attached."""
    return bool(images)
