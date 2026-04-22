import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from fastapi import HTTPException
from typing import Literal

from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.helpers.common import normalize_stop, strftime_now


def _get_bos_eos_tokens(llm):
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()

    bos_token = llm.detokenize([bos_id]).decode("utf-8", errors="ignore") if bos_id != -1 else ""
    eos_token = llm.detokenize([eos_id]).decode("utf-8", errors="ignore") if eos_id != -1 else ""

    return bos_token, eos_token


def render_chat_prompt_with_explicit_think(
    llm,
    metadata_info: dict,
    messages: list[dict],
    think_enabled: bool,
    user_stop: list[str],
):
    template = metadata_info.get("template")
    if not template:
        raise HTTPException(
            status_code=501,
            detail="Model template is not available; cannot render chat prompt."
        )

    bos_token, eos_token = _get_bos_eos_tokens(llm)

    context = {
        "messages": messages,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_generation_prompt": True,
        "tools": [],
        "functions": None,
        "function_call": None,
        "tool_choice": None,
        "enable_thinking": think_enabled,
        "thinking": think_enabled,

        "Messages": messages,
        "Tools": [],
        "Response": "",
        "Think": think_enabled,
        "ThinkLevel": "",
        "IsThinkSet": True,
    }

    try:
        env = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.ChainableUndefined,
        )
        env.globals["strftime_now"] = strftime_now
        tmpl = env.from_string(template)
        prompt = tmpl.render(**context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"chat template render failed: {type(e).__name__}: {e}"
        )

    stop = list(user_stop or [])
    if eos_token and eos_token not in stop:
        stop.append(eos_token)

    return prompt, stop


def render_generate_prompt(
    llm,
    metadata_info: dict,
    request,
    mode: Literal["prompt", "messages"] = "prompt",
):
    """Render a completion prompt from a model template and request data.

    Returns:
        tuple[str, list[str]]:
            The rendered prompt string and the final stop token list to use for
            completion generation.

    Raises:
        HTTPException:
            Raised when the model template is unavailable or template rendering
            fails.
    """
    template = getattr(request, "template", None) or metadata_info.get("template")
    if not template:
        raise HTTPException(
            status_code=501,
            detail="Model template is not available; cannot render prompt for this model."
        )

    bos_token, eos_token = _get_bos_eos_tokens(llm)

    if mode == "prompt":
        prompt_text = getattr(request, "prompt", None) or ""
        system_text = getattr(request, "system", None) or ""

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt_text})

        tools_value = []
    else:
        messages = list(getattr(request, "messages", None) or [])
        tools_value = list(getattr(request, "tools", None) or [])

        system_text = ""
        prompt_text = ""

        for message in messages:
            if (
                message.get("role") in ("system", "developer")
                and isinstance(message.get("content"), str)
                and not system_text
            ):
                system_text = message["content"]

        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                prompt_text = message["content"]
                break

    think_value = getattr(request, "think", None)
    think_is_set = think_value is not None

    think_enabled = None
    think_level = ""
    reasoning_effort = None

    if think_value is True:
        think_enabled = True
    elif think_value is False:
        think_enabled = False
        think_level = "none"
        reasoning_effort = "none"
    elif isinstance(think_value, str):
        normalized_think = think_value.strip().lower()
        if normalized_think == "none":
            think_enabled = False
            think_level = "none"
            reasoning_effort = "none"
        else:
            think_enabled = True
            think_level = normalized_think
            reasoning_effort = normalized_think

    tllama_options = {}
    options_value = getattr(request, "options", None)
    if isinstance(options_value, dict):
        raw_tllama_options = options_value.get("tllama", {})
        if isinstance(raw_tllama_options, dict):
            tllama_options = raw_tllama_options

    developer_instructions = tllama_options.get("developer_instructions")
    model_identity = tllama_options.get("model_identity")

    context = {
        "prompt": prompt_text,
        "Prompt": prompt_text,
        "system": system_text,
        "System": system_text,
        "messages": messages,
        "Messages": messages,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_generation_prompt": True,

        "tools": tools_value,
        "available_tools": tools_value,
        "Tools": tools_value,
        "documents": [],
        "controls": [],
        "add_vision_id": False,
        "preserve_thinking": False,

        "Response": "",
        "IsThinkSet": think_is_set
    }

    if developer_instructions is not None:
        context["developer_instructions"] = developer_instructions
    if model_identity is not None:
        context["model_identity"] = model_identity

    if think_is_set:
        context.update(
            {
                "enable_thinking": think_enabled,
                "thinking": think_enabled,
                "Think": think_enabled,
                "ThinkLevel": think_level
            }
        )
        if reasoning_effort is not None:
            context["reasoning_effort"] = reasoning_effort

    def raise_exception(message: str):
        raise ValueError(message)

    try:
        env = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.ChainableUndefined,
        )
        env.globals["strftime_now"] = strftime_now
        env.globals["raise_exception"] = raise_exception

        tmpl = env.from_string(template)
        prompt = tmpl.render(**context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"template render failed: {type(e).__name__}: {e}"
        )

    stop = normalize_stop((getattr(request, "options", None) or {}).get("stop"))
    if eos_token and eos_token not in stop:
        stop.append(eos_token)

    return prompt, stop
