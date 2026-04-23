from __future__ import annotations

from typing import Any

from llama_cpp import Llama, llama_chat_format


def _resolve_chat_completion_handler(llm: Llama):
    """
    Resolve the chat completion handler used by llama-cpp-python.

    Order:
    1. Explicit custom handler attached to the model instance
    2. Handler registered for llm.chat_format
    3. Generic handler resolver from llama_chat_format

    Raises:
        ValueError: If no handler can be resolved.
    """
    if getattr(llm, "chat_handler", None) is not None:
        return llm.chat_handler

    chat_format = getattr(llm, "chat_format", None)

    if chat_format:
        internal_handlers = getattr(llm, "_chat_handlers", None) or {}
        handler = internal_handlers.get(chat_format)
        if handler is not None:
            return handler

        handler = llama_chat_format.get_chat_completion_handler(chat_format)
        if handler is not None:
            return handler

    raise ValueError(
        "Unable to resolve chat completion handler. "
        "The model instance has no usable chat_handler/chat_format."
    )


def create_chat_completion_ex(
    llm: Llama,
    *,
    messages,
    functions=None,
    function_call=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop=None,
    seed: int | None = None,
    max_tokens: int | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.0,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: str | None = None,
    logits_processor=None,
    grammar=None,
    logit_bias=None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    **kwargs_ex: Any,
):
    """
    Extended variant of llama-cpp-python create_chat_completion().

    It preserves the usual chat-completion call shape, but also forwards extra
    keyword arguments into the underlying chat formatter/template layer.

    This makes it possible to pass custom template variables like:
    - enable_thinking
    - Think
    - ThinkLevel
    - reasoning_effort
    - developer_instructions
    - model_identity
    - any other template-specific flags

    Returns:
        Same type/shape as the underlying chat handler:
        - non-stream: chat completion dict
        - stream: iterator of chat completion chunks
    """
    handler = _resolve_chat_completion_handler(llm)

    return handler(
        llama=llm,
        messages=messages,
        functions=functions,
        function_call=function_call,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=stream,
        stop=stop,
        seed=seed,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,
        logit_bias=logit_bias,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        **kwargs_ex,
    )
