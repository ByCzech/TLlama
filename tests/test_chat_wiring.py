"""The public chat handler applies the leased system default."""


def test_chat_handler_applies_snapshot_default_system_prompt(client, native_reply):
    state = native_reply(metadata={"default_system_prompt": "from TOML"})
    assert client.post('/api/chat', json={"model": "m", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]}).status_code == 200
    assert state.worker.calls[-1][1]["messages"][0] == {"role": "system", "content": "from TOML"}
