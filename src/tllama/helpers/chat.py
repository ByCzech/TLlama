from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .schemas import OllamaChatRequest, Message

from tllama.helpers.common import normalize_message_content
from tllama.helpers.vision import build_multimodal_content, content_carries_images


def build_think_kwargs_ex(think_value) -> dict[str, Any]:
    if think_value is None:
        return {}

    kwargs_ex: dict[str, Any] = {
        "IsThinkSet": True,
    }

    think_disabled = (
        think_value is False
        or (isinstance(think_value, str) and think_value.strip().lower() in {"false", "none"})
    )
    think_enabled = not think_disabled

    kwargs_ex["enable_thinking"] = think_enabled
    kwargs_ex["thinking"] = think_enabled
    kwargs_ex["Think"] = think_enabled

    if think_disabled:
        kwargs_ex["ThinkLevel"] = "none"
        kwargs_ex["reasoning_effort"] = "none"
    elif isinstance(think_value, str):
        level = think_value.strip().lower()
        if level not in {"", "true"}:
            kwargs_ex["ThinkLevel"] = level
            kwargs_ex["reasoning_effort"] = level

    return kwargs_ex


def build_chat_kwargs_ex(request: OllamaChatRequest) -> dict[str, Any]:
    return build_think_kwargs_ex(getattr(request, "think", None))


def build_chat_response_format_kwargs(format_value) -> dict[str, Any]:
    """
    Minimal response_format mapping for chat-completion style calls.

    If you already have a shared helper that returns chat-compatible response_format,
    reuse it here.
    """
    if format_value is None:
        return {}

    if format_value == "json":
        return {
            "response_format": {
                "type": "json_object"
            }
        }

    if isinstance(format_value, dict):
        return {
            "response_format": {
                "type": "json_object",
                "schema": format_value,
            }
        }

    raise ValueError("Invalid format schema")


def normalize_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for m in messages:
        images = getattr(m, "images", None)

        if images:
            # Ollama keeps images beside the text, in their own field, as
            # bare base64. Every handler downstream reads OpenAI content
            # parts instead, so this is where the two shapes meet. Passing
            # the field through untouched, as this used to, meant the
            # image reached the template as a key nothing rendered and the
            # model answered about a picture it never got.
            content: Any = build_multimodal_content(
                normalize_message_content(m.content), images
            )
        elif content_carries_images(m.content):
            # Already in the shape the handlers read. Flattening it to
            # text, which is what normalize_message_content does to any
            # list, would keep the words and drop the picture.
            content = m.content
        else:
            content = normalize_message_content(m.content)

        msg = {
            "role": m.role,
            "content": content,
        }

        if getattr(m, "thinking", None):
            msg["thinking"] = m.thinking
        if getattr(m, "tool_calls", None):
            msg["tool_calls"] = m.tool_calls
        if getattr(m, "tool_name", None):
            msg["tool_name"] = m.tool_name
        if getattr(m, "tool_call_id", None):
            msg["tool_call_id"] = m.tool_call_id

        normalized.append(msg)

    return normalized


def apply_default_system_prompt(messages: list[dict[str, Any]], metadata_info: dict) -> list[dict[str, Any]]:
    """Prepend a virtual model's [system].prompt default, only if nothing
    in messages already carries system/developer content.

    Presence, not truthiness, decides whether a default is needed: an
    explicit empty string ("") is a deliberate request for no system
    content and must count as already provided, the same as a non-empty
    one -- it must not be treated as absent. metadata_info.get(
    "default_system_prompt") follows the same "is not None" rule for the
    same reason on the toml side.

    Returns a new list; never mutates the one passed in.
    """
    has_system_message = any(
        message.get("role") in ("system", "developer")
        and isinstance(message.get("content"), str)
        for message in messages
    )
    if has_system_message:
        return list(messages)

    default_system_text = metadata_info.get("default_system_prompt")
    if default_system_text is None:
        return list(messages)

    return [{"role": "system", "content": default_system_text}] + list(messages)
