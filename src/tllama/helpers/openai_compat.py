def openai_reasoning_effort_to_explicit_think(request) -> bool | None:
    effort = getattr(request, "reasoning_effort", None)

    # Some client shapes may carry request.reasoning.effort instead.
    reasoning_obj = getattr(request, "reasoning", None)
    if effort is None and reasoning_obj is not None:
        if isinstance(reasoning_obj, dict):
            effort = reasoning_obj.get("effort")
        else:
            effort = getattr(reasoning_obj, "effort", None)

    if effort is None:
        return None

    effort = str(effort).strip().lower()

    # Conservative first step: only "none" maps to think=False.
    if effort == "none":
        return False

    # Everything else is left without an explicit override for now.
    return None


def build_openai_chat_messages(request, metadata_info: dict | None = None) -> list[dict]:
    messages = []
    for m in request.messages:
        content = m.content

        # If the schema ever carries content parts, flatten them to text.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            content = "".join(parts)

        messages.append({
            "role": m.role,
            "content": content if isinstance(content, str) else ""
        })

    has_system_message = any(m["role"] in ("system", "developer") for m in messages)
    if not has_system_message and metadata_info:
        default_system_text = metadata_info.get("default_system_prompt")
        if default_system_text is not None:
            messages = [{"role": "system", "content": default_system_text}] + messages

    return messages
