"""Optional real Python SDK consumers over public TCP and an owned worker."""

import asyncio
from contextlib import aclosing
import json
import socket

import pytest
import uvicorn

from tllama.main import app
from test_native_public_api import public, native_manager
from test_native_manager import records, wait_until
from test_native_public_tools import TOOLS, messages, KEYS


@pytest.fixture
def sdk_import(monkeypatch):
    # Ollama constructs default clients at import time. Keep this local-only
    # integration independent of workstation proxy credentials/extras.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(key, raising=False)
    return lambda name: pytest.importorskip(name)


@pytest.fixture
async def sdk_public(public):
    manager, _, name, _ = public
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, http='h11', loop='asyncio', lifespan='off',
        access_log=False, log_level='warning', timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await wait_until(lambda: server.started)
        yield manager, name, f'http://127.0.0.1:{sock.getsockname()[1]}'
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=3)
        finally:
            sock.close()


@pytest.mark.parametrize('stream', [False, True])
async def test_ollama_sdk_objects_replay_without_ids_or_type(sdk_import, sdk_public, monkeypatch, stream):
    ollama = sdk_import('ollama')
    manager, name, address = sdk_public
    monkeypatch.setenv('STUB_TOOLS', 'normal')
    client = ollama.AsyncClient(host=address, trust_env=False, timeout=3)
    try:
        response = await client.chat(model=name, messages=messages(), tools=TOOLS, stream=stream, think=False)
        if stream:
            calls = []
            async with aclosing(response):
                async for part in response:
                    calls.extend(part.message.tool_calls or [])
            assert part.done
        else:
            calls = response.message.tool_calls
        assert [dict(c.function.arguments) for c in calls] == [{'key': key} for key in KEYS]
        # Replay the SDK objects directly, using the documented tool_name form.
        history = messages() + [ollama.Message(role='assistant', tool_calls=calls)] + [
            ollama.Message(role='tool', tool_name=c.function.name, content=json.dumps({'marker': f'MARKER_{i}'}))
            for i, c in enumerate(calls)]
        answer = await client.chat(model=name, messages=history, tools=TOOLS, stream=stream, think=False)
        if stream:
            text = ''
            async with aclosing(answer):
                async for part in answer:
                    text += part.message.content or ''
                    assert not part.message.tool_calls
            assert part.done
        else:
            text = answer.message.content
            assert answer.done and not answer.message.tool_calls
        assert text == 'MARKER_0 MARKER_1'
        upstream = [r['body'] for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/chat/completions']
        replay = next(m for m in upstream[-1]['messages'] if m.get('tool_calls'))
        assert len(replay['tool_calls']) == 2
        assert all(c['type'] == 'function' for c in replay['tool_calls'])
        assert manager.native_snapshot()[0]['leases'] == 0 and len(manager.test.ledger) == 1
    finally:
        await client.close()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mode', ['normal', 'growing-name'])
async def test_openai_sdk_string_deltas_and_id_history(sdk_import, sdk_public, monkeypatch, stream, mode):
    openai = sdk_import('openai')
    manager, name, address = sdk_public
    monkeypatch.setenv('STUB_TOOLS', mode)
    async with openai.AsyncOpenAI(base_url=address + '/v1', api_key='local-test', max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(trust_env=False), timeout=3) as client:
        response = await client.chat.completions.create(model=name, messages=messages(), tools=TOOLS,
            stream=stream, tool_choice='required', parallel_tool_calls=True)
        if stream:
            by_index = {}
            async with response:
                async for part in response:
                    for choice in part.choices:
                        for c in choice.delta.tool_calls or []:
                            current = by_index.setdefault(c.index, {'type': 'function', 'function': {'name': '', 'arguments': ''}})
                            if c.id is not None:
                                current['id'] = c.id
                            if c.function:
                                current['function']['name'] += c.function.name or ''
                                current['function']['arguments'] += c.function.arguments or ''
            calls = [by_index[i] for i in sorted(by_index)]
            assistant = {'role': 'assistant', 'content': '', 'tool_calls': calls}
        else:
            assistant = response.choices[0].message.model_dump(exclude_none=True)
            calls = assistant['tool_calls']
        assert [c['function']['name'] for c in calls] == ['lookup_record'] * 2
        assert [json.loads(c['function']['arguments']) for c in calls] == [{'key': key} for key in KEYS]
        history = messages() + [assistant] + [{'role': 'tool', 'tool_call_id': c['id'],
            'content': json.dumps({'marker': f'MARKER_{i}'})} for i, c in enumerate(calls)]
        answer = await client.chat.completions.create(model=name, messages=history, tools=TOOLS, tool_choice='none')
        assert answer.choices[0].message.content == 'MARKER_0 MARKER_1'
        assert answer.choices[0].finish_reason == 'stop' and not answer.choices[0].message.tool_calls
        assert manager.native_snapshot()[0]['leases'] == 0 and len(manager.test.ledger) == 1
