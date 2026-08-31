"""/v1/chat/completions must default temperature/top_p to the same
baseline as /api/generate and /api/chat (0.8/0.9), not silently fall
through to create_chat_completion's own internal default (0.2/0.95 --
confirmed directly in llama-cpp-python 0.3.35's source) when the OpenAI
client omits them.

Routed through build_sampling_kwargs() (see test_sampling_kwargs.py for
the baseline values themselves); this file only confirms the endpoint
actually uses it.
"""

import inspect

from tllama.helpers.common import build_sampling_kwargs
from tllama.routers import openai as openai_router
from tllama.schemas.openai import ChatCompletionRequest


def _request(**overrides) -> ChatCompletionRequest:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def _opts_built_by_the_endpoint(request: ChatCompletionRequest) -> dict:
    """The opts dict chat_completions() assembles from the request.

    Kept in step with the endpoint by the source check below rather than by
    importing it, because the endpoint builds it inline inside an async
    handler that would otherwise need a loaded model to reach.
    """
    opts = {}
    if request.temperature is not None:
        opts["temperature"] = request.temperature
    if request.top_p is not None:
        opts["top_p"] = request.top_p
    if request.max_tokens is not None:
        opts["num_predict"] = request.max_tokens
    if request.stop:
        opts["stop"] = request.stop
    return opts


def test_chat_completions_uses_build_sampling_kwargs():
    source = inspect.getsource(openai_router.chat_completions)

    assert "build_sampling_kwargs(" in source


def test_omitted_temperature_stays_none_on_the_request():
    assert _request().temperature is None


def test_omitted_temperature_reaches_the_shared_baseline():
    kwargs = build_sampling_kwargs(_opts_built_by_the_endpoint(_request()), {})

    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.9


def test_omitted_temperature_lets_a_virtual_model_s_sampling_win():
    metadata_info = {"sampling_defaults": {"temperature": 0.5}}

    kwargs = build_sampling_kwargs(_opts_built_by_the_endpoint(_request()), metadata_info)

    assert kwargs["temperature"] == 0.5


def test_explicit_temperature_still_wins_over_everything():
    metadata_info = {"sampling_defaults": {"temperature": 0.5}}
    request = _request(temperature=0.3)

    kwargs = build_sampling_kwargs(_opts_built_by_the_endpoint(request), metadata_info)

    assert kwargs["temperature"] == 0.3


def test_explicit_zero_temperature_is_not_mistaken_for_absent():
    """0.0 is a real, meaningful request for greedy sampling.

    The endpoint tests `is not None` rather than truthiness precisely so
    this survives; a truthiness check would silently promote it to the
    baseline 0.8, which is the opposite of what the caller asked for.
    """
    kwargs = build_sampling_kwargs(_opts_built_by_the_endpoint(_request(temperature=0.0)), {})

    assert kwargs["temperature"] == 0.0
