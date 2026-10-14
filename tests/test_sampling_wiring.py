"""Public routes actually apply the shared native sampling layers."""

from dataclasses import replace
import pytest


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/v1/chat/completions"])
def test_every_handler_passes_snapshot_and_global_sampling(client, native_reply, path):
    state = native_reply(metadata={"sampling_defaults": {"temperature": 0.4}})
    state.manager.config = replace(state.manager.config, sampling_overrides={"top_p": "0.65", "temperature": "0.7"})
    payload = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "hi"}]}
    if path.endswith("generate"):
        payload.update(prompt="hi", raw=True)
    assert client.post(path, json=payload).status_code == 200
    body = state.worker.calls[-1][1]
    assert body["temperature"] == 0.4 and body["top_p"] == 0.65
