"""Runner compatibility must not select a backend or permit early side effects."""

from dataclasses import replace

import pytest

from tllama.backend import HF_LOOKUP_FOUND, HfFileLookup
from tllama.config import ConfigError, load_backend_config_from_env


PATHS = ["/api/chat", "/api/generate", "/api/show", "/api/pull"]


def request_body(path):
    if path == "/api/pull":
        return {"model": "ns/repo/m.gguf", "stream": False}
    return {"model": "m", "messages": [], "prompt": "", "stream": False}


@pytest.fixture
def policy(manager, monkeypatch):
    from tllama.routers import ollama as router

    calls = []
    monkeypatch.setattr(router, "model_manager", manager)

    def configure(ignore=False):
        manager.config = replace(manager.config, ignore_unsupported_runner=ignore)
        return calls

    async def load(name, **kwargs):
        calls.append(("load", name))
        return object()

    async def unload(name):
        calls.append(("unload", name))

    async def metadata(name, **kwargs):
        calls.append(("metadata", name))
        return {"arch": "llama", "params": 1000, "template": "{{ messages }}"}

    def resolve(name):
        calls.append(("resolve", name))
        return {"repo_id": "ns/repo", "filename": "m.gguf"}

    async def lookup(**kwargs):
        calls.append(("lookup", kwargs["repo_id"]))
        return HfFileLookup(HF_LOOKUP_FOUND)

    async def download(**kwargs):
        calls.append(("download", kwargs["repo_id"]))
        return "/fake/m.gguf"

    async def digest(*args, **kwargs):
        return {"content_sha256": "digest"}

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(manager, "preload_native_model", load)
    monkeypatch.setattr(manager, "unload_native_model", unload)
    monkeypatch.setattr(manager, "get_model_metadata", metadata)
    monkeypatch.setattr(manager, "get_model_metadata_best_effort", metadata)
    monkeypatch.setattr(manager, "read_model_definition", lambda *args: "[llm]\n")
    monkeypatch.setattr(manager, "resolve_hf_pull_target", resolve)
    monkeypatch.setattr(manager, "fetch_hf_file_info", lookup)
    monkeypatch.setattr(manager, "pull_hf_file", download)
    monkeypatch.setattr(manager, "store_hf_digest", digest)
    monkeypatch.setattr(manager, "get_model_digest", digest)
    monkeypatch.setattr(manager, "ensure_metadata_cache_for_path", noop)
    monkeypatch.setattr(manager, "ensure_virtual_model_toml", noop)
    return configure


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("runner", [None, "", "llamacpp", "ggml"])
def test_supported_runner_reaches_the_existing_operation(client, policy, path, runner):
    calls = policy()
    body = {**request_body(path), "runner": runner}
    response = client.post(path, json=body)

    assert response.status_code == 200
    if path == "/api/pull":
        assert calls == [
            ("resolve", "ns/repo/m.gguf"), ("lookup", "ns/repo"),
            ("download", "ns/repo"),
        ]
    elif path == "/api/show":
        assert response.json()["details"]["runner"] == "llamacpp"
    else:
        assert calls == [("load", "m")]
        assert response.json()["done_reason"] == "load"


@pytest.mark.parametrize("path", PATHS)
def test_omitted_runner_stays_compatible(client, policy, path):
    policy()
    assert client.post(path, json=request_body(path)).status_code == 200


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("runner", ["mlx", "unknown", "cuda"])
def test_unsupported_runner_is_rejected_before_side_effects(client, policy, path, runner):
    calls = policy()
    response = client.post(path, json={**request_body(path), "runner": runner})

    assert response.status_code == 400
    assert "Unsupported runner" in response.json()["error"]
    assert calls == []


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("runner", ["mlx", "unknown"])
def test_ignore_mode_uses_llamacpp_without_a_warning(client, policy, caplog, path, runner):
    calls = policy(ignore=True)
    response = client.post(path, json={**request_body(path), "runner": runner})

    assert response.status_code == 200
    assert calls
    assert not any("runner" in record.getMessage().lower() for record in caplog.records)
    if path == "/api/show":
        assert response.json()["details"]["runner"] == "llamacpp"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("ignore", [False, True])
@pytest.mark.parametrize("runner", [1, True, [], {}])
def test_non_string_runner_is_always_rejected(client, policy, path, ignore, runner):
    calls = policy(ignore)
    response = client.post(path, json={**request_body(path), "runner": runner})

    assert response.status_code == 400
    assert "runner" in response.json()["error"]
    assert calls == []


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_rejected_runner_cannot_unload_a_model(client, policy, path):
    calls = policy()
    response = client.post(path, json={
        **request_body(path), "runner": "mlx", "keep_alive": 0,
    })

    assert response.status_code == 400
    assert calls == []


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize("runner,ignore", [("ggml", False), ("mlx", True)])
def test_accepted_runner_preserves_unload_semantics(client, policy, path, runner, ignore):
    calls = policy(ignore)
    response = client.post(path, json={
        **request_body(path), "runner": runner, "keep_alive": 0,
    })

    assert response.status_code == 200
    assert response.json()["done_reason"] == "unload"
    assert calls == [("unload", "m")]


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_rejected_runner_cannot_start_inference(client, policy, path):
    calls = policy()
    body = {**request_body(path), "runner": "mlx"}
    body.update(messages=[{"role": "user", "content": "hi"}], prompt="hi")

    assert client.post(path, json=body).status_code == 400
    assert calls == []


def test_runner_metadata_covers_loaded_and_loading_models(client, policy, monkeypatch):
    from tllama.routers import ollama as router

    policy()
    manager = router.model_manager

    async def listing():
        return [{"id": "ready", "mtime": 0, "size": 1}]

    monkeypatch.setattr(manager, "list_local_models_with_metadata", listing)
    monkeypatch.setattr(manager, "list_loaded_models", lambda: [{
        "id": "ready", "model": "ready", "path": "/fake/ready.gguf",
        "size": 1, "size_vram": 0, "n_ctx": 4096, "expires_at": None,
    }])
    monkeypatch.setattr(manager, "list_loading_models", lambda: ["loading"])
    monkeypatch.setattr(manager, "build_model_file_info_best_effort", lambda *args: {
        "path": "/fake/loading.gguf", "size": 2,
    })

    tags = client.get("/api/tags").json()["models"]
    assert tags[0]["details"]["runner"] == "llamacpp"
    ps = client.get("/api/ps")
    assert ps.status_code == 503
    assert "block 06" in ps.json()["error"]


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
def test_ignore_mode_is_a_server_configuration(value, expected, monkeypatch):
    monkeypatch.setenv("TLLAMA_IGNORE_UNSUPPORTED_RUNNER", value)
    assert load_backend_config_from_env().ignore_unsupported_runner is expected


def test_ignore_mode_defaults_to_strict(monkeypatch):
    monkeypatch.delenv("TLLAMA_IGNORE_UNSUPPORTED_RUNNER", raising=False)
    assert load_backend_config_from_env().ignore_unsupported_runner is False


def test_invalid_ignore_mode_fails_configuration(monkeypatch):
    monkeypatch.setenv("TLLAMA_IGNORE_UNSUPPORTED_RUNNER", "maybe")
    with pytest.raises(ConfigError, match="TLLAMA_IGNORE_UNSUPPORTED_RUNNER"):
        load_backend_config_from_env()


def test_openai_schema_does_not_gain_runner():
    from tllama.schemas.openai import ChatCompletionRequest
    assert "runner" not in ChatCompletionRequest.model_fields
