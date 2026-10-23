"""OpenAI request controls reach native sampling, residency and public SSE."""

import json

import pytest

from test_native_public_api import public, request_body
from test_native_manager import native_manager, records, wait_until


@pytest.mark.parametrize('stream', [False, True])
async def test_explicit_sampling_and_zero_keep_alive_reach_the_owned_generation(public, stream):
    manager, client, name, toml = public
    toml.write_text(toml.read_text() + '\nseed=123\nfrequency_penalty=0.7\npresence_penalty=0.8\n')
    payload = {**request_body(name, '/v1/chat/completions', stream),
               'seed': 0, 'frequency_penalty': 0.0, 'presence_penalty': -0.25, 'keep_alive': 0}
    response = await client.post('/v1/chat/completions', json=payload)
    assert response.status_code == 200, response.text
    await wait_until(lambda: not manager.native_snapshot())
    calls = [r['body'] for r in records(manager.test.roots[0], 'requests')
             if r['path'] == '/v1/chat/completions']
    assert len(calls) == 1
    assert calls[0]['seed'] == 0 and calls[0]['frequency_penalty'] == 0.0
    assert calls[0]['presence_penalty'] == -0.25 and 'keep_alive' not in calls[0]
    worker = manager.test.ledger[0]
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


async def test_omitted_sampling_controls_preserve_model_defaults_and_negative_keep_alive(public):
    manager, client, name, toml = public
    toml.write_text(toml.read_text() + '\nseed=123\nfrequency_penalty=0.7\npresence_penalty=0.8\n')
    response = await client.post('/v1/chat/completions', json={
        **request_body(name, '/v1/chat/completions'), 'keep_alive': -1})
    assert response.status_code == 200, response.text
    call = next(r['body'] for r in records(manager.test.roots[0], 'requests')
                if r['path'] == '/v1/chat/completions')
    assert (call['seed'], call['frequency_penalty'], call['presence_penalty']) == (123, 0.7, 0.8)
    assert manager.native_snapshot()[0]['keep_alive'] is None


@pytest.mark.parametrize('options,include', [(None, False), ({}, False),
    ({'include_usage': False}, False), ({'include_usage': True}, True)])
async def test_public_usage_frame_is_opt_in_and_precedes_done(public, options, include):
    manager, client, name, _ = public
    payload = request_body(name, '/v1/chat/completions', True)
    if options is not None:
        payload['stream_options'] = options
    response = await client.post('/v1/chat/completions', json=payload)
    assert response.status_code == 200
    frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    assert frames[-1] == '[DONE]'
    bodies = [json.loads(frame) for frame in frames[:-1]]
    usage = [b for b in bodies if not b['choices']]
    assert len(usage) == int(include)
    if include:
        assert usage[0] is bodies[-1] and bodies[-2]['choices'][0]['finish_reason'] == 'stop'
        assert usage[0]['usage']['prompt_tokens'] == 25
        assert usage[0]['usage']['prompt_tokens_details']['cached_tokens'] == 19
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('extra', [
    {'seed': True}, {'seed': 1.5}, {'seed': '42'}, {'seed': -2}, {'seed': 2**32},
    {'frequency_penalty': True}, {'frequency_penalty': '0.5'}, {'frequency_penalty': float('inf')},
    {'presence_penalty': float('nan')}, {'keep_alive': True}, {'keep_alive': 'bad-duration'},
    {'keep_alive': float('inf')}, {'stream_options': {'include_usage': 1}},
    {'stream_options': {'include_usage': 'true'}}, {'stream_options': {'unknown': True}},
])
async def test_invalid_controls_fail_before_load_or_retirement(public, extra):
    manager, client, name, _ = public
    # Serialize nonfinite JSON explicitly: the transport's convenience encoder
    # otherwise refuses it before the application can validate the request.
    payload = {**request_body(name, '/v1/chat/completions'), **extra}
    response = await client.post('/v1/chat/completions', content=json.dumps(payload),
                                 headers={'Content-Type': 'application/json'})
    assert response.status_code == 400, response.text
    assert not manager.test.ledger and not manager.native_snapshot()
