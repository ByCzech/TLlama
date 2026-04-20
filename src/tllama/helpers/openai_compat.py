def openai_reasoning_effort_to_explicit_think(request) -> bool | None:
    effort = getattr(request, "reasoning_effort", None)

    # některé klientské struktury mohou mít request.reasoning.effort
    reasoning_obj = getattr(request, "reasoning", None)
    if effort is None and reasoning_obj is not None:
        if isinstance(reasoning_obj, dict):
            effort = reasoning_obj.get("effort")
        else:
            effort = getattr(reasoning_obj, "effort", None)

    if effort is None:
        return None

    effort = str(effort).strip().lower()

    # bezpečný první krok:
    # jen "none" mapujeme na think=False
    if effort == "none":
        return False

    # ostatní zatím necháme bez explicitního override
    return None


def build_openai_chat_messages(request) -> list[dict]:
    messages = []
    for m in request.messages:
        content = m.content

        # pokud schema někdy používá content parts, zjednoduš na text
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
    return messages
