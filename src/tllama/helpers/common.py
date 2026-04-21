import json
from datetime import datetime, timezone
from llama_cpp import LlamaGrammar


def get_iso_time():
    """Ollama wants specific time format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_stop(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s for s in value if isinstance(s, str) and s != ""]


def normalize_max_tokens_from_options(opts: dict):
    num_predict = opts.get("num_predict", None)

    if num_predict is None:
        return None

    if isinstance(num_predict, int) and num_predict <= 0:
        return None

    return num_predict


def normalize_optional_max_tokens(value):
    if value is None:
        return None
    if isinstance(value, int) and value <= 0:
        return None
    return value


def normalize_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def resolve_think_flag(request) -> bool | str | None:
    if getattr(request, "think", None) is not None:
        return request.think

    options = getattr(request, "options", {}) or {}
    opt_think = options.get("think")

    if isinstance(opt_think, (bool, str)):
        return opt_think

    return None


def estimate_completion_prompt_eval_count(llm, prompt: str) -> int:
    bos_token_id = llm.token_bos()
    eos_token_id = llm.token_eos()

    try:
        add_bos = bool(llm._model.add_bos_token()) and bos_token_id != -1
    except Exception:
        add_bos = bos_token_id != -1

    try:
        add_eos = bool(llm._model.add_eos_token()) and eos_token_id != -1
    except Exception:
        add_eos = eos_token_id != -1

    prompt_tokens = llm.tokenize(
        prompt.encode("utf-8"),
        add_bos=False,
        special=True,
    )

    return len(prompt_tokens) + (1 if add_bos else 0) + (1 if add_eos else 0)


def build_completion_format_kwargs(request_format):
    """
    Completion path uses grammar, not response_format.
    Supports:
    - format == "json"
    - format == {...json schema...}
    """
    if request_format == "json":
        return {
            "grammar": LlamaGrammar.from_json_schema(
                json.dumps({"type": "object"})
            )
        }

    if isinstance(request_format, dict):
        return {
            "grammar": LlamaGrammar.from_json_schema(
                json.dumps(request_format)
            )
        }

    return {}
