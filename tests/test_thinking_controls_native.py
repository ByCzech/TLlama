"""Qualified thinking controls cross the real owned-process native boundary."""

import json

import pytest

from test_native_public_api import public
from test_native_manager import native_manager, records
from test_thinking_controls import GPT


@pytest.mark.parametrize('path', ['/v1/chat/completions', '/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_owned_worker_receives_qualified_alias_and_generated_prompt(public, monkeypatch, path, stream):
    manager, client, name, toml = public
    toml.write_text(toml.read_text().replace('jinja="{{ bos_token }}{{ System }}|{{ Prompt }}"',
                                             'jinja=' + json.dumps(GPT['template'])))
    monkeypatch.setattr(manager, '_get_model_metadata_sync', lambda _: {**GPT})
    body = {'model': name, 'stream': stream}
    if path == '/v1/chat/completions':
        body.update(messages=[{'role': 'user', 'content': 'hi'}], reasoning_effort='low', reasoning={'effort': 'xhigh'})
    elif path == '/api/chat':
        body.update(messages=[{'role': 'user', 'content': 'hi'}], think='xhigh')
    else:
        body.update(prompt='hi', think='xhigh')
    response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    requests = [r['body'] for r in records(manager.test.roots[0], 'requests')
                if r['path'] in {'/completion', '/v1/chat/completions'}]
    assert len(requests) == 1 and len(manager.test.ledger) == 1
    native = requests[0]
    if path == '/api/generate':
        assert 'Reasoning: high\n' in ''.join(chr(t - 10) for t in native['prompt'] if t >= 10)
    else:
        assert native['reasoning_effort'] == 'high'
        assert native['chat_template_kwargs']['enable_thinking'] is True
    assert native['temperature'] == 0.4
    assert manager.native_snapshot()[0]['leases'] == 0
