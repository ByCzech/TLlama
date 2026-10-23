"""Convert TLLAMA_SAMPLING_* using TLlama's pinned native contract."""

from typing import Any, Dict, Mapping

from tllama.config import ConfigError
from tllama.native_params import SAMPLING_TYPES, sampling_problem
from tllama.helpers.runtime_params import coerce_from_annotation


ENV_PREFIX = "TLLAMA_SAMPLING_"
_TOML_HINT = " Set it in a model's .toml [sampling] section instead."


def env_key_for(name: str) -> str:
    return f"{ENV_PREFIX}{name.upper()}"


def completion_parameters() -> Dict[str, str]:
    """TLlama sampling vocabulary, independent of completion signatures."""
    return dict(SAMPLING_TYPES)


def settable_parameters() -> Dict[str, str]:
    return completion_parameters()


def coerce_sampling_overrides(raw: Mapping[str, str]) -> Dict[str, Any]:
    """Keep only explicitly set globals, preserving model/GGUF precedence."""
    result = {}
    for name, value in raw.items():
        variable = env_key_for(name)
        annotation = SAMPLING_TYPES.get(name)
        if annotation is None:
            if name == "stream":
                raise ConfigError(f"{variable}: TLlama does not apply stream as a sampling setting.")
            raise ConfigError(f"{variable}: {sampling_problem(name, value)}.")
        result[name] = (coerce_from_annotation(variable, annotation, value, unsupported_hint=_TOML_HINT)
                        if isinstance(value, str) else value)
        problem = sampling_problem(name, result[name])
        if problem:
            raise ConfigError(f"{variable}: {problem}.")
    return result
