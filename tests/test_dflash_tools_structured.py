"""DFlash adapter combinations over subprocess sockets, not native GPU quality."""

import asyncio
import json
from pathlib import Path
import socket

import httpx
import pytest
import uvicorn

from tllama.main import app
from test_helper_definitions import publish, reference
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, request_body
from test_native_public_tools import TOOLS, KEYS, messages, decode
from test_repository_worker_pins import protected_groups
from test_speculative_stats import EXTENSION


@pytest.fixture
async def dflash(public, monkeypatch, tmp_path):
    manager, client, name, toml = public
    _, relative = publish(manager, tmp_path, 'dflash', count=2)
    toml.write_text(toml.read_text() + '\n' + reference('draft', relative)
                    + '[runtime]\nspec_type="draft-dflash"\n')
    monkeypatch.setenv('STUB_PUBLIC_DFLASH', '1')
    original = toml.read_bytes()
    yield manager, client, name, toml
    assert toml.read_bytes() == original


def bodies(response, path, stream):
    assert response.status_code == 200, response.text
    if not stream:
        return [response.json()]
    if path.startswith('/v1/'):
        frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
        assert frames[-1] == '[DONE]'
        return [json.loads(frame) for frame in frames[:-1]]
    values = [json.loads(line) for line in response.text.splitlines()]
    assert values[-1]['done'] and not any('error' in value for value in values)
    return values


def measured(response, path, stream, active=True):
    values = bodies(response, path, stream)
    exported = [value['tllama'] for value in values if value.get('tllama', {}).get('speculative')]
    assert exported == ([EXTENSION['tllama']] if active else [])
    return values


def launch(manager, active=True):
    worker = manager.test.ledger[-1]
    args = worker.command
    assert args[args.index('--spec-type') + 1] == ('draft-dflash' if active else 'none')
    assert ('--spec-draft-model' in args) == active
    if active:
        path = Path(args[args.index('--spec-draft-model') + 1])
        assert len(list(path.parent.glob('*.gguf'))) == 2
        assert path.parent.name in protected_groups(manager)
    assert manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_dflash_tool_cycle_preserves_escaped_calls_history_and_request_counters(dflash, monkeypatch, path, stream):
    manager, client, name, _ = dflash
    monkeypatch.setenv('STUB_TOOLS', 'growing-name')
    api = 'openai' if path.startswith('/v1/') else 'ollama'
    body = {'model': name, 'stream': stream, 'messages': messages(), 'tools': TOOLS}
    if api == 'openai':
        body.update(tool_choice='required', parallel_tool_calls=True,
                    stream_options={'include_usage': True} if stream else None)
    response = await client.post(path, json=body)
    values = measured(response, path, stream)
    first = decode(response, api, stream)
    assert [c['function']['name'] for c in first['tool_calls']] == ['lookup_record'] * 2
    arguments = [c['function']['arguments'] for c in first['tool_calls']]
    assert (arguments if api == 'ollama' else list(map(json.loads, arguments))) == [{'key': k} for k in KEYS]
    assert len({c['id'] for c in first['tool_calls']}) == 2
    if api == 'ollama':
        assert values[-1]['done']
        if stream:
            assert 'tool_calls' not in values[-1]['message']
            assert all('tllama' not in value for value in values[:-1])
    else:
        assert [v['choices'][0]['finish_reason'] for v in values if v.get('tllama')] == ['tool_calls']
        if stream:
            assert values[-1]['choices'] == [] and values[-1]['usage']['completion_tokens'] == 10
    first['thinking' if api == 'ollama' else 'reasoning_content'] = 'tool reasoning'
    body['messages'] = messages() + [first] + [
        {'role': 'tool', 'tool_call_id': c['id'], 'name': c['function']['name'],
         'content': json.dumps({'marker': f'MARKER_{i}'})} for i, c in enumerate(first['tool_calls'])]
    if api == 'openai':
        body['tool_choice'] = 'none'
    response = await client.post(path, json=body)
    measured(response, path, stream)
    answer = decode(response, api, stream)
    assert answer['content'] == 'MARKER_0 MARKER_1' and not answer.get('tool_calls')
    assert len(manager.test.ledger) == 1
    sent = [r['body'] for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/chat/completions']
    assert len(sent) == 2 and sent[-1]['tools'] == TOOLS
    replay = next(m for m in sent[-1]['messages'] if m.get('tool_calls'))
    assert replay['tool_calls'] == first['tool_calls'] and replay['reasoning_content'] == 'tool reasoning'
    launch(manager)


@pytest.mark.parametrize('stream', [False, True])
async def test_tool_result_survives_off_reload_without_stale_draft_counts(dflash, monkeypatch, stream):
    manager, client, name, _ = dflash
    monkeypatch.setenv('STUB_TOOLS', 'normal')
    path = '/api/chat'
    body = {'model': name, 'stream': stream, 'messages': messages(), 'tools': TOOLS}
    first_response = await client.post(path, json=body)
    measured(first_response, path, stream)
    first = decode(first_response, 'ollama', stream)
    launch(manager)
    body['messages'] = messages() + [first] + [
        {'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps({'marker': f'MARKER_{i}'})}
        for i, c in enumerate(first['tool_calls'])]
    body['options'] = {'spec_type': 'none', 'draft_num_predict': 3}
    response = await client.post(path, json=body)
    measured(response, path, stream, active=False)
    assert decode(response, 'ollama', stream)['content'] == 'MARKER_0 MARKER_1'
    assert len(manager.test.ledger) == 2 and manager.test.ledger[0].process.returncode is not None
    assert not protected_groups(manager)
    sent = records(manager.test.roots[-1], 'requests')[-1]['body']
    assert next(m for m in sent['messages'] if m.get('tool_calls'))['tool_calls'] == first['tool_calls']
    launch(manager, active=False)


SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'integer'}},
          'required': ['answer'], 'additionalProperties': False}


@pytest.mark.parametrize('path,raw', [('/api/chat', False), ('/api/generate', False),
                                    ('/api/generate', True), ('/v1/chat/completions', False)])
@pytest.mark.parametrize('stream', [False, True])
async def test_schema_payload_and_result_preserve_active_dflash(dflash, path, raw, stream):
    manager, client, name, _ = dflash
    body = request_body(name, path, stream)
    if path.startswith('/v1/'):
        expected = {'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True, 'schema': SCHEMA}}
        body['response_format'] = expected
    else:
        body['format'] = SCHEMA
        expected = {'type': 'json_object', 'schema': SCHEMA}
        if raw:
            body['raw'] = True
    response = await client.post(path, json=body)
    values = measured(response, path, stream)
    if path == '/api/generate':
        text = ''.join(v.get('response', '') for v in values)
    elif path == '/api/chat':
        text = ''.join(v['message'].get('content', '') for v in values)
    else:
        text = ''.join(c.get('delta' if stream else 'message', {}).get('content', '')
                       for v in values for c in v['choices'])
    assert json.loads(text) == {'answer': 42}
    sent = [r['body'] for r in records(manager.test.roots[0], 'requests')
            if r['path'] in {'/completion', '/v1/chat/completions'}]
    assert len(sent) == 1
    assert sent[0].get('json_schema' if path == '/api/generate' else 'response_format') == (
        SCHEMA if path == '/api/generate' else expected)
    assert not {'spec_type', 'draft_num_predict', 'draft_n_max', 'grammar'} & sent[0].keys()
    assert len(manager.test.ledger) == 1
    launch(manager)


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('mode', ['malformed-json', 'no-done', 'truncated'])
async def test_incomplete_tools_never_publish_success_or_speculative_stats(dflash, monkeypatch, path, mode):
    manager, client, name, _ = dflash
    monkeypatch.setenv('STUB_TOOLS', mode)
    body = {'model': name, 'messages': messages(), 'tools': TOOLS, 'stream': True}
    response = await client.post(path, json=body)
    assert response.status_code == 200 and 'error' in response.text
    values = [json.loads(line[6:] if path.startswith('/v1/') else line)
              for line in response.text.splitlines() if line and line != 'data: [DONE]']
    assert not any(v.get('tllama', {}).get('speculative') for v in values)
    if path == '/api/chat':
        assert not any(v.get('done') or v.get('message', {}).get('tool_calls') for v in values)
    else:
        assert '[DONE]' not in response.text and not any(
            c.get('finish_reason') for v in values for c in v.get('choices', []))
    launch(manager)
    body['stream'] = False  # The same stub only corrupts streamed tool fragments.
    measured(await client.post(path, json=body), path, False)
    assert len(manager.test.ledger) == 1


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_tcp_disconnect_cancels_dflash_tool_stream_without_retiring_worker(dflash, monkeypatch, path):
    manager, _, name, _ = dflash
    monkeypatch.setenv('STUB_TOOLS', 'normal')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, http='h11', loop='asyncio', lifespan='off',
        access_log=False, log_level='warning', timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await wait_until(lambda: server.started)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}',
                                     trust_env=False, timeout=3) as client:
            body = {'model': name, 'stream': True, 'tools': TOOLS, 'messages': [
                {'role': 'user', 'content': 'Retrieve keys: ["cancel_' + 'x' * 1000 + '"]'}]}
            async with client.stream('POST', path, json=body) as response:
                assert response.status_code == 200
                await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
                assert manager.native_snapshot()[0]['leases'] == 1
            await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
            assert records(manager.test.roots[0], 'cancelled') == ['tools']
            launch(manager)
            body.update(stream=False, messages=messages())
            measured(await client.post(path, json=body), path, False)
            assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode is None
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=3)
        finally:
            sock.close()
