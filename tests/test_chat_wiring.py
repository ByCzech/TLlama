import inspect

from tllama.routers import ollama as ollama_router


def test_chat_handler_calls_apply_default_system_prompt():
    source = inspect.getsource(ollama_router.ollama_chat)

    assert "apply_default_system_prompt(" in source
