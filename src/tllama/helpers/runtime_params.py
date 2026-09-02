"""Turn TLLAMA_RUNTIME_* strings into Llama() keyword arguments.

Which parameters exist, and what each one takes, is read off
inspect.signature(Llama.__init__) rather than listed here. A list would be
a shadow copy of somebody else's function: it would need an edit whenever
llama-cpp-python gains a parameter, and until that edit it would be wrong
in the direction of silently refusing something that works.

An environment variable is a string and the signature says what it has to
become. Those two facts are the whole mechanism.
"""

from __future__ import annotations

import inspect

from typing import Any, Dict, Mapping, Optional, Tuple

from tllama.config import ConfigError


ENV_PREFIX = "TLLAMA_RUNTIME_"

# Not settable from the environment, each for its own reason.
#
# model_path is the only real prohibition: it comes from [llm] in a .toml,
# and pointing it elsewhere would mean the model is not the model its
# definition names.
#
# The rest are settable, just not here, because a dedicated variable
# already covers them and a second spelling of the same setting is the
# duplication this module exists to avoid. The message says which one.
_HANDLED_ELSEWHERE = {
    "model_path": "it is set by [llm] in a model's .toml",
    "n_ctx": "use TLLAMA_CONTEXT_LENGTH",
    "type_k": "use TLLAMA_K_CACHE_TYPE or TLLAMA_KV_CACHE_TYPE",
    "type_v": "use TLLAMA_V_CACHE_TYPE or TLLAMA_KV_CACHE_TYPE",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_key_for(name: str) -> str:
    """The variable a parameter is set by, for use in messages."""
    return f"{ENV_PREFIX}{name.upper()}"


def _strip_optional(annotation: str) -> str:
    compact = annotation.replace(" ", "")
    if compact.startswith("Optional[") and compact.endswith("]"):
        return compact[len("Optional[") : -1]
    return compact


def _coerce_bool(variable: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(
        f"{variable}={value!r} is not valid: expected one of "
        "1/true/yes/on or 0/false/no/off."
    )


def _coerce(variable: str, annotation: str, value: str) -> Any:
    kind = _strip_optional(annotation)
    raw = value.strip()

    if kind == "bool":
        return _coerce_bool(variable, value)

    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(
                f"{variable}={value!r} is not valid: expected a whole number."
            ) from None

    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(
                f"{variable}={value!r} is not valid: expected a number."
            ) from None

    if kind == "str":
        return value

    if kind == "Union[bool,int]":
        # numa is spelled this way: False, or one of the numa strategy
        # numbers. A word picks the bool reading, digits the int one, so
        # neither spelling is ambiguous.
        if raw.lower() in _TRUE or raw.lower() in _FALSE:
            return _coerce_bool(variable, value)
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(
                f"{variable}={value!r} is not valid: expected true/false "
                "or a whole number."
            ) from None

    if kind == "List[float]":
        # tensor_split, the one list worth taking: it is how a model is
        # spread over several GPUs and there is no other way to say it.
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            raise ConfigError(
                f"{variable}={value!r} is not valid: expected "
                "comma-separated numbers."
            )
        try:
            return [float(part) for part in parts]
        except ValueError:
            raise ConfigError(
                f"{variable}={value!r} is not valid: expected "
                "comma-separated numbers."
            ) from None

    raise ConfigError(
        f"{variable} cannot be set from the environment: {kind} is not a "
        "value a string can express."
    )


def settable_parameters() -> Dict[str, str]:
    """Parameter name to annotation, for those an environment can set."""
    from llama_cpp import Llama

    parameters = inspect.signature(Llama.__init__).parameters

    return {
        name: parameter.annotation
        for name, parameter in parameters.items()
        if name not in ("self",)
        and parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and parameter.annotation is not parameter.empty
        and name not in _HANDLED_ELSEWHERE
    }


def coerce_runtime_overrides(raw: Mapping[str, str]) -> Dict[str, Any]:
    """Convert collected TLLAMA_RUNTIME_* strings into Llama() kwargs.

    Called once at startup, so an unknown name or an unusable value stops
    the server rather than surfacing on the first request that loads a
    model.
    """
    known = settable_parameters()
    coerced: Dict[str, Any] = {}

    for name, value in raw.items():
        variable = env_key_for(name)

        if name in _HANDLED_ELSEWHERE:
            raise ConfigError(
                f"{variable} is not settable: {_HANDLED_ELSEWHERE[name]}."
            )

        annotation = known.get(name)
        if annotation is None:
            raise ConfigError(
                f"{variable} is not settable: {name} is not a parameter of "
                "Llama() in the installed llama-cpp-python."
            )

        coerced[name] = _coerce(variable, annotation, value)

    return coerced
