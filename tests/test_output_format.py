"""Invalid output formats must not silently produce unconstrained text."""


import pytest



def payload(path, stream=False, empty=False, unload=False):
    body = {"model": "m", "stream": stream}
    if path == "/api/chat":
        body["messages"] = [] if empty else [{"role": "user", "content": "hi"}]
    else:
        body["prompt"] = "" if empty else "hi"
    if unload:
        body["keep_alive"] = 0
    return body


@pytest.fixture
def no_model_actions(monkeypatch):
    from tllama.routers import ollama as router

    async def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid format reached model loading")

    def forbidden_unload(*args, **kwargs):
        pytest.fail("Invalid format reached model unloading")

    monkeypatch.setattr(router.model_manager, "acquire_native_model", forbidden_load)
    monkeypatch.setattr(router.model_manager, "unload_native_model", forbidden_unload)


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", ["xml", "JSON", "json ", " ", 1, True, []])
def test_invalid_format_is_a_client_error_before_model_actions(
    client, no_model_actions, path, stream, value
):
    response = client.post(path, json={**payload(path, stream), "format": value})

    assert response.status_code == 400
    assert isinstance(response.json()["error"], str)


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize("unload", [False, True])
def test_invalid_format_does_not_bypass_validation_on_load_only_requests(
    client, no_model_actions, path, unload
):
    response = client.post(path, json={
        **payload(path, empty=True, unload=unload), "format": "xml",
    })

    assert response.status_code == 400
