from __future__ import annotations

from typing import Any, Iterable, Optional


# Ollama's own wording, kept verbatim: a client that already knows how to
# read this from real Ollama gets the same sentence from TLlama.
NO_MULTIMODAL_SUPPORT = (
    "Multimodal data provided, but model does not support multimodal requests."
)


def messages_carry_images(messages: Iterable[Any]) -> bool:
    """Whether any message in a chat request came with images attached."""
    for message in messages:
        if getattr(message, "images", None):
            return True
    return False


def request_carries_images(images: Optional[Iterable[Any]]) -> bool:
    """Whether a generate request came with images attached."""
    return bool(images)
