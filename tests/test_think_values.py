"""Model-defined thinking levels must reach the template unchanged in meaning."""

import json

import pytest

from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest


@pytest.mark.parametrize("schema", [OllamaChatRequest, OllamaGenerateRequest])
@pytest.mark.parametrize("value", [None, False, True, "max", "custom-level"])
def test_think_retains_its_type_and_value(schema, value):
    request = schema(model="m", messages=[], think=value)

    assert request.think == value
    assert type(request.think) is type(value)


@pytest.mark.parametrize("schema", [OllamaChatRequest, OllamaGenerateRequest])
def test_unset_think_stays_unset(schema):
    assert schema(model="m", messages=[]).think is None


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize("value", [0, 1, 0.5, [], {}])
def test_invalid_think_is_rejected_before_loading(client, monkeypatch, path, value):
    from tllama.routers import ollama as router

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid think value reached model loading")

    monkeypatch.setattr(router.model_manager, "get_model", forbidden)
    response = client.post(path, json={
        "model": "m", "messages": [], "prompt": "hi", "think": value,
    })

    assert response.status_code == 400
    assert "think" in response.json()["error"]


@pytest.fixture
def answered(monkeypatch):
    from tllama.routers import ollama as router

    calls = {}

    class Llm:
        chat_handler = None

        def token_bos(self):
            return -1

        def token_eos(self):
            return -1

        def tokenize(self, *args, **kwargs):
            return [1]

        def create_completion(self, *, prompt, stream, **kwargs):
            calls["prompt"] = prompt
            choice = {"text": "ok", "finish_reason": "stop"}
            if stream:
                return iter([{"choices": [choice]}])
            return {"choices": [choice], "usage": {"completion_tokens": 1}}

    async def get_model(*args, **kwargs):
        return Llm()

    async def metadata(*args, **kwargs):
        return {"template":
                "{{ ThinkLevel }}|{{ reasoning_effort }}|{{ enable_thinking }}|{{ IsThinkSet }}"}

    def chat(*args, stream, **kwargs):
        calls["chat"] = kwargs
        if stream:
            return iter([{"choices": [{
                "delta": {"content": "ok"}, "finish_reason": "stop",
            }]}])
        return {"choices": [{
            "message": {"content": "ok"}, "finish_reason": "stop",
        }], "usage": {"completion_tokens": 1}}

    monkeypatch.setattr(router.model_manager, "get_model", get_model)
    monkeypatch.setattr(router.model_manager, "get_model_metadata", metadata)
    monkeypatch.setattr(router, "create_chat_completion_ex", chat)
    monkeypatch.setattr(router, "reset_eval_counters", lambda *args: False)
    return calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("level", ["max", "custom-level"])
@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_custom_level_reaches_chat_and_generate_templates(
    client, answered, path, level, stream
):
    response = client.post(path, json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "prompt": "hi", "think": level, "stream": stream,
    })

    assert response.status_code == 200
    if stream:
        assert json.loads(response.text.splitlines()[-1])["done"] is True
    else:
        assert response.json()["done"] is True
    if path == "/api/chat":
        assert answered["chat"]["ThinkLevel"] == level
        assert answered["chat"]["reasoning_effort"] == level
        assert answered["chat"]["enable_thinking"] is True
    else:
        assert answered["prompt"] == f"{level}|{level}|True|True"


@pytest.mark.parametrize("value", [False, "none", "false"])
def test_explicit_disable_stays_consistent_in_generate(client, answered, value):
    response = client.post("/api/generate", json={
        "model": "m", "prompt": "hi", "think": value, "stream": False,
    })

    assert response.status_code == 200
    assert answered["prompt"] == "none|none|False|True"


@pytest.mark.parametrize("value", [False, "none", "false"])
@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize("stream", [False, True])
def test_disabled_qwen_answer_is_not_reclassified_as_thinking(
    client, answered, value, path, stream
):
    response = client.post(path, json={
        "model": "Qwen/test", "messages": [{"role": "user", "content": "hi"}],
        "prompt": "hi", "think": value, "stream": stream,
    })

    assert response.status_code == 200
    replies = ([json.loads(line) for line in response.text.splitlines()]
               if stream else [response.json()])
    outputs = [reply["message"] if path == "/api/chat" else reply for reply in replies]
    assert all(not output.get("thinking") for output in outputs)
    key = "content" if path == "/api/chat" else "response"
    assert "".join(output.get(key, "") for output in outputs) == "ok"
