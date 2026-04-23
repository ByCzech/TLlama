from __future__ import annotations

from typing import Any

from llama_cpp import Llama, llama_chat_format

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .schemas import OllamaChatRequest, Message

from tllama.helpers.common import normalize_message_content


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
        msg = {
            "role": m.role,
            "content": normalize_message_content(m.content),
        }

        if getattr(m, "images", None):
            msg["images"] = m.images
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
