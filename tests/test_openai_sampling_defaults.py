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

from tllama.routers import openai as openai_router


def test_chat_completions_uses_build_sampling_kwargs():
    source = inspect.getsource(openai_router.chat_completions)

    assert "build_sampling_kwargs(" in source
