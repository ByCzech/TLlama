"""Turn TLLAMA_SAMPLING_* strings into completion keyword arguments.

The same mechanism helpers/runtime_params.py applies to Llama(), applied
to the two completion calls instead: which names exist and what each one
takes is read off their signatures, not listed here.

Which names TLlama will actually act on is a narrower question than which
ones the library accepts, and helpers/common.py already answers it for a
.toml's [sampling]. This layer uses the same answer, so a name settable in
one place is settable in the other and neither can drift.

A name whose annotation no string can express -- logit_bias is a table,
stop is a list -- is refused with a message naming the .toml, which is
where it can be set. That falls out of the coercion rather than being
listed, so it stays right if a parameter's type changes.
"""

from __future__ import annotations

import functools
import inspect

from typing import Any, Dict, Mapping

from tllama.config import ConfigError
from tllama.helpers.common import sampling_parameter_names
from tllama.helpers.runtime_params import coerce_from_annotation


ENV_PREFIX = "TLLAMA_SAMPLING_"

_TOML_HINT = " Set it in a model's .toml [sampling] section instead."


def env_key_for(name: str) -> str:
    """The variable a parameter is set by, for use in messages."""
    return f"{ENV_PREFIX}{name.upper()}"


@functools.lru_cache(maxsize=1)
def completion_parameters() -> Dict[str, str]:
    """Name to annotation for parameters both completion calls accept.

    The intersection, not either one alone. build_sampling_kwargs() is a
    single function feeding /api/generate, /api/chat and
    /v1/chat/completions, so a name it can pass has to exist on both
    methods -- suffix and echo are only on one, tools and response_format
    only on the other, and a global value for any of those would work on
    some endpoints and be a TypeError on the rest.

    Cached: a signature cannot change while the process runs.
    """
    from llama_cpp import Llama

    def annotations_of(method) -> Dict[str, str]:
        return {
            name: parameter.annotation
            for name, parameter in inspect.signature(method).parameters.items()
            if name != "self"
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            and parameter.annotation is not parameter.empty
        }

    completion = annotations_of(Llama.create_completion)
    chat = annotations_of(Llama.create_chat_completion)

    return {name: annotation for name, annotation in completion.items() if name in chat}


def settable_parameters() -> Dict[str, str]:
    """Name to annotation, for those a TLLAMA_SAMPLING_* may set.

    Narrower than completion_parameters() twice over: only what TLlama
    applies at all, and of that only what the library really has, so a
    name that disappears from a future llama-cpp-python is refused here
    rather than passed to a call that no longer takes it.
    """
    applied = sampling_parameter_names()

    return {
        name: annotation
        for name, annotation in completion_parameters().items()
        if name in applied
    }


def coerce_sampling_overrides(raw: Mapping[str, str]) -> Dict[str, Any]:
    """Convert collected TLLAMA_SAMPLING_* strings into completion kwargs.

    Called once at startup, so an unknown name or an unusable value stops
    the server rather than surfacing on the first request that generates
    anything.

    What comes back holds only what was actually set. That is the whole
    point of this layer: an unset global has to stay unset, or it would
    overrule a model's own recommended values everywhere, for every model,
    without anybody having asked for it.
    """
    known = settable_parameters()
    coerced: Dict[str, Any] = {}

    for name, value in raw.items():
        variable = env_key_for(name)

        annotation = known.get(name)
        if annotation is None:
            if name in completion_parameters():
                raise ConfigError(
                    f"{variable} is not settable: TLlama does not apply "
                    f"{name}."
                )
            raise ConfigError(
                f"{variable} is not settable: {name} is not a sampling "
                "parameter of the installed llama-cpp-python."
            )

        coerced[name] = coerce_from_annotation(
            variable, annotation, value, unsupported_hint=_TOML_HINT
        )

    return coerced
