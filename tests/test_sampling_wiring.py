"""build_sampling_kwargs() must actually be called by all three inference
handlers, not just imported.

This exists because of a real, confirmed miss earlier in this same series:
render_chat_prompt_with_explicit_think() carried the first version of the
[system] default-prompt logic, had passing tests for the function itself,
and was never actually reachable from /api/chat -- superseded by
create_chat_completion_ex() months earlier, with the dead function and its
import simply never removed. A source-inspection test like this one would
have caught that immediately instead of four months later.

/v1/chat/completions is already covered by
tests/test_openai_sampling_defaults.py; the two Ollama-native handlers are
covered here.
"""

import inspect

from tllama.routers import ollama as ollama_router


def test_generate_handler_calls_build_sampling_kwargs():
    source = inspect.getsource(ollama_router.ollama_generate)

    assert "build_sampling_kwargs(" in source


def test_chat_handler_calls_build_sampling_kwargs():
    source = inspect.getsource(ollama_router.ollama_chat)

    assert "build_sampling_kwargs(" in source
