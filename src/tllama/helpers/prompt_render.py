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

        request_system = getattr(request, "system", None)
        if request_system is not None:
            system_text = request_system
            has_system_content = True
        else:
            default_system_text = metadata_info.get("default_system_prompt")
            if default_system_text is not None:
                system_text = default_system_text
                has_system_content = True
            else:
                system_text = ""
                has_system_content = False

        messages = []
        if has_system_content:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt_text})

        tools_value = []
    else:
        messages = list(getattr(request, "messages", None) or [])
        tools_value = list(getattr(request, "tools", None) or [])

        system_text = ""
        has_system_content = False
        prompt_text = ""

        for message in messages:
            if (
                message.get("role") in ("system", "developer")
                and isinstance(message.get("content"), str)
                and not has_system_content
            ):
                system_text = message["content"]
                has_system_content = True

        if not has_system_content:
            default_system_text = metadata_info.get("default_system_prompt")
            if default_system_text is not None:
                messages = [{"role": "system", "content": default_system_text}] + messages
                system_text = default_system_text
                has_system_content = True

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
    if not stop:
        stop = list(metadata_info.get("stop_defaults") or [])
    if eos_token and eos_token not in stop:
        stop.append(eos_token)

    return prompt, stop
