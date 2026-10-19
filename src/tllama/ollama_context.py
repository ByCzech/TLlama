"""Bound Ollama input using the admitted worker's tokenizer and context."""

from dataclasses import replace

from tllama.worker import WorkerProtocolError


def truncate_completion(tokens, context, keep, *, add_bos=False):
    if len(tokens) < context:
        return tokens
    if context < 5:
        raise ValueError("Context is too small for prompt truncation")
    keep = len(tokens) if keep < 0 else keep
    keep = min(keep + int(add_bos), context - 4)
    limit = context - max((context - keep) // 2, 1)
    return tokens[:keep] + tokens[keep + len(tokens) - limit:]


async def prepare_ollama_context(prepared, lease):
    body = prepared.body()
    if "n_keep" not in body:
        return prepared
    context = (lease.worker.props or {}).get("default_generation_settings", {}).get("n_ctx")
    if type(context) is not int or context < 1:
        raise WorkerProtocolError("num_keep requires the effective native context")
    if prepared.kind != "chat":
        return prepared  # Completion tokens are bounded before this call; FIM preserves its recipe.
    messages = body["messages"]
    while True:
        prompt = await lease.worker.render_chat(body)
        tokens = await lease.worker.tokenize(prompt, add_special=True)
        if len(tokens) < context:
            break
        # Remove one complete old user turn, including its tool exchanges.
        users = [i for i, message in enumerate(messages) if message["role"] == "user"]
        if len(users) < 2:
            raise ValueError("The latest chat turn and system prompt exceed the effective context")
        first, following = users[:2]
        messages = [message for i, message in enumerate(messages)
                    if not first <= i < following or message["role"] == "system"]
        body["messages"] = messages
    return replace(prepared, payload=body)
