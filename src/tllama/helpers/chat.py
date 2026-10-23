from __future__ import annotations

from typing import Any


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
        for message in messages
    )
    if has_system_message:
        return list(messages)

    default_system_text = metadata_info.get("default_system_prompt")
    if default_system_text is None:
        return list(messages)

    return [{"role": "system", "content": default_system_text}] + list(messages)
