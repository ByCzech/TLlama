"""/v1/chat/completions must default temperature/top_p to the same
baseline as /api/generate and /api/chat (0.8/0.9), not silently fall
through to create_chat_completion's own internal default (0.2/0.95 --
confirmed directly in llama-cpp-python 0.3.35's source) when the OpenAI
client omits them.
"""

import inspect

from tllama.routers import openai as openai_router


def test_temperature_and_top_p_have_explicit_baseline_defaults():
    source = inspect.getsource(openai_router.chat_completions)

    assert '"temperature": 0.8' in source
    assert '"top_p": 0.9' in source
