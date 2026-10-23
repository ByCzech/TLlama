"""Convert TLLAMA_RUNTIME_* strings using TLlama's native contract."""

from __future__ import annotations


from typing import Any, Dict, Mapping

from tllama.config import ConfigError
from tllama.native_params import RUNTIME_TYPES, runtime_problem


ENV_PREFIX = "TLLAMA_RUNTIME_"

# Not settable from the environment, each for its own reason.
#
# model_path comes from [llm] in a .toml,
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
    "type_kv": "use TLLAMA_KV_CACHE_TYPE",
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


def coerce_from_annotation(
    variable: str,
    annotation: str,
    value: str,
    *,
    unsupported_hint: str = "",
) -> Any:
    """Turn one environment string into what an annotation asks for.

    Shared with the sampling layer: an environment string is converted
    using the types declared by TLlama's pinned native contract.

    unsupported_hint is appended when the annotation names something a
    string cannot express, so the message can say where that setting does
    belong instead of only that it does not belong here.
    """
    return _coerce(variable, annotation, value, unsupported_hint=unsupported_hint)


def _coerce(
    variable: str,
    annotation: str,
    value: str,
    *,
    unsupported_hint: str = "",
) -> Any:
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
        parts = [part.strip() for part in raw.split(",")]
        if any(not part for part in parts):
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
        f"value a string can express.{unsupported_hint}"
    )


def runtime_parameters() -> Dict[str, str]:
    """The pinned TLlama runtime contract, independent of Python bindings."""
    return dict(RUNTIME_TYPES)


def settable_parameters() -> Dict[str, str]:
    return {name: kind for name, kind in RUNTIME_TYPES.items() if name not in _HANDLED_ELSEWHERE}


def coerce_runtime_overrides(raw: Mapping[str, str]) -> Dict[str, Any]:
    """Convert collected TLLAMA_RUNTIME_* strings into typed runtime values.

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
                f"{variable}: {runtime_problem(name, value)}."
            )

        coerced[name] = (_coerce(variable, annotation, value,
                                 unsupported_hint=" Set it in a model's .toml [runtime] instead.")
                         if isinstance(value, str) else value)
        problem = runtime_problem(name, coerced[name])
        if problem:
            raise ConfigError(f"{variable}: {problem}.")

    return coerced
