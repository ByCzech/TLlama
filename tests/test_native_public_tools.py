"""Caller-owned Tools cycle through public APIs and a real subprocess socket."""

import asyncio
import json
import socket

import httpx
import pytest
import uvicorn

from test_native_public_api import public, native_manager
from test_native_manager import records, wait_until
from tllama.main import app


TOOLS = [{'type': 'function', 'function': {'name': 'lookup_record', 'parameters': {
    'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}}]
KEYS = ['alpha', 'žluťoučký "kůň"' + chr(92) + 'folder' + chr(10) + 'line']


def messages():
    return [{'role': 'user', 'content': 'Retrieve record keys: ' + json.dumps(KEYS, ensure_ascii=False)}]


def decode(response, api, stream):
    assert response.status_code == 200, response.text
    if not stream:
        value = response.json()
        return value['message'] if api == 'ollama' else value['choices'][0]['message']
    if api == 'ollama':
        values = [json.loads(line) for line in response.text.splitlines()]
        assert values[-1]['done'] and not any('error' in v for v in values)
        assert all(not v['done'] for v in values[:-1])
        calls = [c for v in values for c in v['message'].get('tool_calls', [])]
        return {'role': 'assistant', 'content': ''.join(v['message']['content'] for v in values), 'tool_calls': calls}
    frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    assert frames[-1] == '[DONE]'
    result, calls = {'role': 'assistant', 'content': ''}, {}
    for frame in frames[:-1]:
        value = json.loads(frame)
        assert 'error' not in value
        for choice in value['choices']:
            delta = choice['delta']
            result['content'] += delta.get('content', '')
            for fragment in delta.get('tool_calls', []):
                call = calls.setdefault(fragment['index'], {'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if 'id' in fragment:
                    assert 'id' not in call  # Repeated native IDs are normalized away.
                    call['id'] = fragment['id']
                for key, text in fragment.get('function', {}).items():
                    call['function'][key] += text
    result['tool_calls'] = [calls[i] for i in sorted(calls)]
    return result


@pytest.mark.parametrize('api,path', [('ollama', '/api/chat'), ('openai', '/v1/chat/completions')])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mode', ['normal', 'growing-name'])
async def test_parallel_same_function_cycle_preserves_objects_strings_and_history(public, monkeypatch, api, path, stream, mode):
    manager, client, name, _ = public
    monkeypatch.setenv('STUB_TOOLS', mode)
    body = {'model': name, 'stream': stream, 'messages': messages(), 'tools': TOOLS}
    if api == 'openai':
        body.update(tool_choice='required', parallel_tool_calls=True)
    first = decode(await client.post(path, json=body), api, stream)
    assert len(first['tool_calls']) == 2
    assert [c['function']['name'] for c in first['tool_calls']] == ['lookup_record'] * 2
    args = [c['function']['arguments'] for c in first['tool_calls']]
    decoded = args if api == 'ollama' else [json.loads(a) for a in args]
    assert decoded == [{'key': key} for key in KEYS]
    assert len({c['id'] for c in first['tool_calls']}) == 2
    first['thinking' if api == 'ollama' else 'reasoning_content'] = 'keep tool reasoning'
    history = messages() + [first] + [
        {'role': 'tool', 'tool_call_id': call['id'], 'name': call['function']['name'],
         'content': json.dumps({'marker': f'MARKER_{i}'})}
        for i, call in enumerate(first['tool_calls'])]
    body['messages'] = history
    if api == 'openai':
        body['tool_choice'] = 'none'
    answer = decode(await client.post(path, json=body), api, stream)
    assert answer['content'] == 'MARKER_0 MARKER_1' and not answer.get('tool_calls')
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    upstream = [r['body'] for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/chat/completions']
    assert len(upstream) == 2
    replay = next(m for m in upstream[-1]['messages'] if m.get('tool_calls'))
    assert replay['tool_calls'] == first['tool_calls']
    assert replay['reasoning_content'] == 'keep tool reasoning'


@pytest.mark.parametrize('api,path', [('ollama', '/api/chat'), ('openai', '/v1/chat/completions')])
@pytest.mark.parametrize('mode', ['malformed-json', 'no-done', 'truncated'])
async def test_public_tool_stream_error_has_no_success_or_ollama_call(public, monkeypatch, api, path, mode):
    manager, client, name, _ = public
    monkeypatch.setenv('STUB_TOOLS', mode)
    response = await client.post(path, json={'model': name, 'messages': messages(), 'tools': TOOLS, 'stream': True})
    assert response.status_code == 200 and 'error' in response.text
    if api == 'ollama':
        values = [json.loads(line) for line in response.text.splitlines()]
        assert not any(v.get('done') or v.get('message', {}).get('tool_calls') for v in values)
    else:
        assert '[DONE]' not in response.text and '"finish_reason": "tool_calls"' not in response.text
    assert manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_tcp_disconnect_cancels_pending_arguments_even_without_ollama_output(public, monkeypatch, path):
    manager, _, name, _ = public
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
                # The synthetic native task stops after its first argument
                # fragment. Ollama has nothing to emit yet; do not wait for it.
                await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
            await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
            assert records(manager.test.roots[0], 'cancelled') == ['tools']
            assert manager.test.ledger[0].process.returncode is None
            body.update(stream=False, messages=messages())
            next_response = await client.post(path, json=body)
            assert next_response.status_code == 200
            assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=3)
        finally:
            sock.close()
