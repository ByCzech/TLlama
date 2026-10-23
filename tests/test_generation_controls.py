"""A known unsupported request cannot turn into accidental inference/unload."""

import json

import pytest

from test_native_manager import native_manager
from test_native_public_api import public, request_body


PATHS = ['/api/chat', '/api/generate', '/v1/chat/completions']


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('extra,word', [
    ({'_debug_render_only': True}, '_debug_render_only'),
    ({'logprobs': True}, 'logprobs'), ({'top_logprobs': 2}, 'logprobs'),
    ({'logprobs': 'true'}, 'logprobs'), ({'logprobs': 1}, 'logprobs'),
    ({'_debug_render_only': 'true'}, '_debug_render_only'),
    ({'top_logprobs': True}, 'top_logprobs'), ({'top_logprobs': -1}, 'top_logprobs'),
    ({'top_logprobs': 21}, 'top_logprobs'),
])
async def test_unintegrated_or_invalid_generation_controls_fail_before_admission(public, path, extra, word):
    manager, client, name, _ = public
    response = await client.post(path, json={**request_body(name, path, True), **extra})
    assert response.status_code == 400 and word in response.text
    assert not manager.test.ledger and not manager.native_snapshot()


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('control', ['truncate', 'shift'])
@pytest.mark.parametrize('value', [False, True, 1, 'false'])
async def test_top_level_context_flags_do_not_look_like_supported_policy(public, path, control, value):
    manager, client, name, _ = public
    response = await client.post(path, json={**request_body(name, path), control: value})
    assert response.status_code == 400 and control in response.text
    assert not manager.test.ledger


@pytest.mark.parametrize('path', PATHS)
async def test_disabled_default_controls_remain_compatible_and_do_not_reach_native(public, path):
    manager, client, name, _ = public
    payload = {**request_body(name, path), '_debug_render_only': False,
               'logprobs': False, 'top_logprobs': 0, 'sdk_annotation': 'ignored'}
    if path.startswith('/api/'):
        payload.update(truncate=None, shift=None)
    response = await client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_debug_request_cannot_unload_an_existing_generation(public, path):
    manager, client, name, _ = public
    await manager.preload_native_model(name, keep_alive=-1)
    generation = manager.native_snapshot()[0]['generation']
    payload = {'model': name, 'keep_alive': 0, '_debug_render_only': True,
               **({'messages': []} if path.endswith('chat') else {'prompt': ''})}
    response = await client.post(path, json=payload)
    assert response.status_code == 400
    assert manager.native_snapshot()[0]['generation'] == generation
    assert manager.test.ledger[0].process.returncode is None
