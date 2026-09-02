import json
from datetime import datetime, timedelta, timezone
from llama_cpp import LlamaGrammar


# Ollama has no "does not expire" state on the wire. A negative keep_alive
# becomes time.Duration(math.MaxInt64) nanoseconds and expires_at is simply
# that far ahead, which its CLI renders as "Forever" for anything more than
# twenty years out. A JSON null decodes into Go's zero time, which is in the
# past, and the same CLI renders anything past as "Stopping...". So a model
# pinned in memory has to be reported with this horizon, not with null.
NEVER_EXPIRES_SECONDS = (2 ** 63 - 1) // 1_000_000_000


def never_expires_at() -> str:
    """Expiry timestamp an Ollama client renders as "Forever"."""
    return (
        datetime.now(timezone.utc) + timedelta(seconds=NEVER_EXPIRES_SECONDS)
    ).isoformat()


def get_iso_time():
    """Ollama wants specific time format."""
    return datetime.now(timezone.utc).isoformat()


def strftime_now(fmt="%Y-%m-%d %H:%M:%S", *args, **kwargs):
    return datetime.now(timezone.utc).strftime(fmt)


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


# TLlama's own fixed baseline, used only once neither the request nor a
# virtual model's [sampling] section says otherwise. Deliberately not "leave
# it unset and let the library decide": create_completion() and
# create_chat_completion() do not even agree with each other in
# llama-cpp-python 0.3.35 (temperature defaults to 0.8 for one and 0.2 for
# the other, confirmed directly in its source -- and 0.2 has no basis in
# real llama.cpp's own default of 0.8, nor in Ollama's documented default of
# 0.8), so relying on either would reintroduce exactly the cross-endpoint
# inconsistency this function exists to remove, just hidden a layer deeper.
# top_p is 0.9 to match Ollama's own documented default, a deliberate
# choice over llama.cpp's raw 0.95.
_SAMPLING_DEFAULTS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
    "tfs_z": 1.0,
    "mirostat_mode": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "seed": None,
}

# Ollama's client-facing "options" dict spells this "mirostat"; a .toml's
# [sampling] section and this function's own kwargs use the
# llama-cpp-python/[runtime]-style "mirostat_mode" instead, so the two need
# a small translation only on the client-options side.
_CLIENT_OPTION_ALIASES = {
    "mirostat_mode": "mirostat",
}


def sampling_parameter_names() -> frozenset:
    """Names build_sampling_kwargs will actually act on.

    Derived from what the function reads rather than written out beside
    it, so a name added to the baseline becomes settable in a .toml
    without a second edit somewhere else. max_tokens and stop are read
    directly rather than through _SAMPLING_DEFAULTS, so they are added
    here.
    """
    return frozenset(_SAMPLING_DEFAULTS) | {"max_tokens", "stop"}


def build_sampling_kwargs(opts: dict, metadata_info: dict | None = None) -> dict:
    """Build create_completion()/create_chat_completion() sampling kwargs.

    Priority for every parameter: the request's own options (opts) win if
    present, then a virtual model's [sampling] default (via metadata_info,
    see backend.ModelManager.get_model_metadata), then TLlama's own fixed
    baseline in _SAMPLING_DEFAULTS. The same rule applies to max_tokens and
    stop, layered on top of the existing Ollama-specific normalization
    (num_predict's semantics, stop accepting a bare string or a list).

    Used identically by /api/generate, /api/chat, and /v1/chat/completions
    so a virtual model's [sampling] section behaves the same regardless of
    which endpoint a client happens to use.
    """
    opts = opts or {}
    toml_sampling = (metadata_info or {}).get("sampling_defaults") or {}

    kwargs = {}
    for key, default in _SAMPLING_DEFAULTS.items():
        client_key = _CLIENT_OPTION_ALIASES.get(key, key)
        if client_key in opts:
            kwargs[key] = opts[client_key]
        elif key in toml_sampling:
            kwargs[key] = toml_sampling[key]
        else:
            kwargs[key] = default

    if "num_predict" in opts:
        kwargs["max_tokens"] = normalize_max_tokens_from_options(opts)
    else:
        kwargs["max_tokens"] = normalize_optional_max_tokens(toml_sampling.get("max_tokens"))

    stop = normalize_stop(opts.get("stop"))
    if not stop:
        stop = list((metadata_info or {}).get("stop_defaults") or [])
    if stop:
        kwargs["stop"] = stop

    return kwargs


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


DEFAULT_KEEP_ALIVE_SECONDS = 300


def normalize_keep_alive(keep_alive: str | int | float | None) -> int | None:
    """Normalize an Ollama-style keep_alive value to seconds.

    Accepts a bare number of seconds or a duration string with an s, m or h
    suffix, matching the Ollama API.

    Returns:
        int:
            Number of seconds for finite keep-alive values.
        0:
            Immediate unload semantics.
        None:
            Infinite keep-alive, for any negative value.

    Raises:
        ValueError: The value is not a recognised keep_alive form.
    """
    if keep_alive is None:
        return DEFAULT_KEEP_ALIVE_SECONDS

    if isinstance(keep_alive, (int, float)):
        if keep_alive < 0:
            return None
        return int(keep_alive)

    value = str(keep_alive).strip().lower()

    if value == "":
        return DEFAULT_KEEP_ALIVE_SECONDS

    try:
        numeric = float(value)
        if numeric < 0:
            return None
        return int(numeric)
    except ValueError:
        pass

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
    }

    suffix = value[-1]
    if suffix in multipliers:
        try:
            numeric = float(value[:-1])
        except ValueError:
            raise ValueError(f"Invalid keep_alive value: {keep_alive}")

        if numeric < 0:
            return None

        return int(numeric * multipliers[suffix])

    raise ValueError(f"Invalid keep_alive value: {keep_alive}")


# Real Ollama pads a parameter name out to this width before its value.
# Matching it matters only because anything printing the field verbatim,
# which is what `ollama show --parameters` does, would otherwise look
# subtly wrong next to the real thing.
_OLLAMA_PARAMETER_NAME_WIDTH = 31


def format_ollama_parameters(metadata_info: dict | None) -> str:
    """The parameters a model's definition pins, in Ollama's own layout.

    What a definition sets, not what is in effect. Ollama's field carries
    the PARAMETER lines from the Modelfile, so a model whose definition
    pins nothing reports nothing -- reporting TLlama's baseline here
    instead would claim the model asked for values it never mentioned, and
    a client could not tell the two apart.

    Replaces a hardcoded 'stop "<|end_of_text|>"' that was returned for
    every model regardless of what it actually used.
    """
    metadata_info = metadata_info or {}

    sampling = metadata_info.get("sampling_defaults") or {}
    stops = metadata_info.get("stop_defaults") or []

    lines = []

    for name in sorted(sampling):
        lines.append(_ollama_parameter_line(name, sampling[name]))

    for stop in stops:
        lines.append(_ollama_parameter_line("stop", stop))

    return "\n".join(lines)


def _ollama_parameter_line(name: str, value) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, str):
        rendered = f'"{value}"'
    else:
        rendered = str(value)

    return f"{name:<{_OLLAMA_PARAMETER_NAME_WIDTH}}{rendered}"
