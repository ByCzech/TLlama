"""Go GenerateRequest zero-value strings must not accidentally select FIM."""

import signal

import pytest

from tllama.schemas.ollama import OllamaGenerateRequest
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public


def go_request(model, **values):
    # api.GenerateRequest has no omitempty on these three string fields.
    return {'model': model, 'prompt': '', 'suffix': '', 'system': '', 'template': '',
            'options': None, **values}


@pytest.mark.parametrize('suffix,expected', [(None, None), ('', None), (' ', ' '), ('tail', 'tail')])
def test_only_empty_suffix_is_normalized(suffix, expected):
    request = OllamaGenerateRequest(model='m', suffix=suffix, system='', template='')
    assert request.suffix == expected
    # An explicit empty system still suppresses a configured default.
    assert request.system == request.template == ''


@pytest.mark.parametrize('model', ['a', 'not-registered'])
async def test_cold_and_repeated_unload_never_start_or_resolve_a_worker(public, monkeypatch, model):
    manager, client, _, _ = public
    def unexpected(*args, **kwargs):
        pytest.fail('Unload must not resolve a generation or start a native worker')
    monkeypatch.setattr(manager, '_resolve_native_definition_sync', unexpected)
    for _ in range(2):
        response = await client.post('/api/generate', json=go_request(model, keep_alive=0))
        assert response.status_code == 200, response.text
        assert response.json()['done_reason'] == 'unload'
    assert not manager.test.ledger and manager.native_snapshot() == []


async def test_go_preload_reuse_and_unload_retire_the_actual_worker(public):
    manager, client, name, _ = public
    for _ in range(2):
        response = await client.post('/api/generate', json=go_request(name, keep_alive=-1))
        assert response.status_code == 200 and response.json()['done_reason'] == 'load'
    assert len(manager.test.ledger) == 1
    response = await client.post('/api/generate', json=go_request(name, keep_alive=0))
    assert response.status_code == 200 and response.json()['done_reason'] == 'unload'
    await wait_until(lambda: not manager.native_snapshot())
    worker = manager.test.ledger[0]
    assert worker.process.returncode in (0, -signal.SIGTERM) and not worker.runtime_directory.exists()
    assert not [row for row in records(manager.test.roots[0], 'requests')
                if row['path'] in {'/completion', '/v1/chat/completions'}]


@pytest.mark.parametrize('stream', [False, True])
async def test_go_empty_suffix_with_text_uses_ordinary_generation(public, stream):
    manager, client, name, _ = public
    response = await client.post('/api/generate', json=go_request(name, prompt='hello', stream=stream))
    assert response.status_code == 200, response.text
    calls = [row['body'] for row in records(manager.test.roots[0], 'requests') if row['path'] == '/completion']
    assert len(calls) == 1
    # The existing template applies, with the caller's explicit empty system.
    assert ''.join(chr(token - 10) for token in calls[0]['prompt'][1:]) == '|hello'


async def test_empty_suffix_does_not_bypass_other_request_validation(public):
    manager, client, name, _ = public
    response = await client.post('/api/generate', json=go_request(name, keep_alive=0, context=[1]))
    assert response.status_code == 400 and 'context' in response.json()['error']
    assert not manager.test.ledger
