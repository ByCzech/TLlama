import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from fastapi import HTTPException
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.helpers.common import normalize_stop


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
    request: OllamaGenerateRequest,
    full_prompt: str,
    explicit_think: bool | str | None,
):
    template = getattr(request, "template", None) or metadata_info.get("template")
    if not template:
        raise HTTPException(
            status_code=501,
            detail="Model template is not available; cannot render generate prompt for this model."
        )

    bos_id = llm.token_bos()
    eos_id = llm.token_eos()

    bos_token = llm.detokenize([bos_id]).decode("utf-8", errors="ignore") if bos_id != -1 else ""
    eos_token = llm.detokenize([eos_id]).decode("utf-8", errors="ignore") if eos_id != -1 else ""

    think_enabled = explicit_think is not False if explicit_think is not None else True
    think_level = explicit_think if isinstance(explicit_think, str) else ""

    context = {
        "prompt": full_prompt,
        "messages": [{"role": "user", "content": full_prompt}],
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_generation_prompt": True,
        "enable_thinking": think_enabled,
        "thinking": think_enabled,

        "Messages": [{"role": "user", "content": full_prompt}],
        "Tools": [],
        "Response": "",
        "Think": think_enabled,
        "ThinkLevel": think_level,
        "IsThinkSet": explicit_think is not None,
    }

    try:
        env = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.ChainableUndefined,
        )
        tmpl = env.from_string(template)
        prompt = tmpl.render(**context)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"generate template render failed: {type(e).__name__}: {e}"
        )

    stop = normalize_stop(request.options.get("stop"))
    if eos_token and eos_token not in stop:
        stop.append(eos_token)

    return prompt, stop
