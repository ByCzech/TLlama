"""Public routes, real native subprocess TCP, and generation ownership."""

import asyncio
from contextlib import aclosing
import json
import os
from pathlib import Path
import socket
import sys

import httpx
import pytest
import uvicorn
from starlette.requests import ClientDisconnect

from tllama.main import app
from tllama.native_routes import serve_native
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.routers import ollama, openai

from test_native_manager import native_manager, records, wait_until


@pytest.fixture
async def public(native_manager, monkeypatch):
    manager = native_manager()
    name, toml = manager.test.define(extra='[template]\njinja="{{ bos_token }}{{ System }}|{{ Prompt }}"\n'
                                           '[system]\nprompt="SYSTEM"\n[sampling]\ntemperature=0.4')
    monkeypatch.setattr(ollama, 'model_manager', manager)
    monkeypatch.setattr(openai, 'model_manager', manager)
    await manager.start()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://public') as client:
        yield manager, client, name, toml


def request_body(name, path, stream=False):
    body = {"model": name, "stream": stream}
    if path == '/api/generate':
        body.update(prompt='hello', think=False)
    else:
        body['messages'] = [{"role": "user", "content": "hello"}]
        body['think' if path == '/api/chat' else 'reasoning_effort'] = False if path == '/api/chat' else 'none'
    return body


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_public_text_result_and_stream_use_one_native_worker(public, path, stream):
    manager, client, name, _ = public
    response = await client.post(path, json=request_body(name, path, stream))
    assert response.status_code == 200
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    logs = records(manager.test.roots[0], 'requests')
    inference = [r for r in logs if r['path'] in {'/completion', '/v1/chat/completions'}]
    assert len(inference) == 1 and inference[0]['body']['temperature'] == 0.4
    if path == '/api/generate':
        tokens = inference[0]['body']['prompt']
        assert tokens[0] == 1 and tokens[1] != 1  # Exactly one BOS from the rendered template.
        assert ''.join(chr(t - 10) for t in tokens[1:]) == 'SYSTEM|hello'
    if path == '/v1/chat/completions':
        if stream:
            frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
            assert frames[-1] == '[DONE]'
            bodies = [json.loads(frame) for frame in frames[:-1]]
            assert all(body['model'] == name for body in bodies)
            assert ''.join(choice.get('delta', {}).get('reasoning_content', '')
                           for body in bodies for choice in body['choices']) == 'myšlenka'
        else:
            body = response.json()
            assert body['model'] == name and body['usage']['prompt_tokens'] == 25
            assert body['choices'][0]['message']['content'] == 'Ahoj světe'
            assert body['choices'][0]['message']['reasoning_content'] == 'myšlenka'
    else:
        bodies = [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]
        assert bodies[-1]['done'] and bodies[-1]['model'] == name
        assert bodies[-1]['prompt_eval_count'] == 6 and bodies[-1]['prompt_eval_cached_count'] == 19
        assert bodies[-1]['eval_duration'] == 200_100_000


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_public_preload_reuse_unload_zero_does_not_generate(public, path):
    manager, client, name, _ = public
    body = {"model": name, "stream": True, "keep_alive": -1,
            **({"messages": []} if path.endswith('chat') else {"prompt": ''})}
    for _ in range(2):
        result = await client.post(path, json=body)
        assert result.status_code == 200 and result.json()['done_reason'] == 'load'
    assert len(manager.test.ledger) == 1
    body['keep_alive'] = 0
    assert (await client.post(path, json=body)).json()['done_reason'] == 'unload'
    await wait_until(lambda: not manager.native_snapshot())
    body['model'] = 'missing'
    assert (await client.post(path, json=body)).json()['done_reason'] == 'unload'
    assert len(manager.test.ledger) == 1
    assert not [r for r in records(manager.test.roots[0], 'requests') if r['path'] in {'/completion', '/v1/chat/completions'}]


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_zero_keep_alive_inference_closes_after_public_result(public, path):
    manager, client, name, _ = public
    result = await client.post(path, json={**request_body(name, path, True), 'keep_alive': 0})
    assert result.status_code == 200 and json.loads(result.text.splitlines()[-1])['done']
    await wait_until(lambda: not manager.native_snapshot())
    worker = manager.test.ledger[0]
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


@pytest.mark.parametrize('path,extra,status', [('/api/generate', {'suffix': 'suffix'}, 501),
    ('/api/generate', {'context': [1, 2]}, 400), ('/api/generate', {'raw': True, 'system': ''}, 400),
    ('/api/generate', {'images': ['not decoded']}, 501), ('/v1/chat/completions', {'functions': []}, 400),
    ('/v1/chat/completions', {'function_call': 'auto'}, 400), ('/api/chat', {'format': 'bad'}, 400),
    ('/v1/chat/completions', {'response_format': {'type': 'json_schema'}}, 400)])
async def test_checkpoint_limits_are_explicit_before_any_model_load(public, path, extra, status):
    manager, client, name, _ = public
    response = await client.post(path, json={**request_body(name, path), **extra})
    assert response.status_code == status and 'error' in response.json()
    assert not manager.test.ledger


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_media_is_501_and_not_fetched_or_advertised(public, path):
    manager, client, name, _ = public
    body = request_body(name, path)
    body['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': 'file:///etc/passwd'}}]
    assert (await client.post(path, json=body)).status_code == 501
    assert not manager.test.ledger
    assert 'vision' not in (await client.post('/api/show', json={'model': name})).json()['capabilities']


async def test_ps_is_explicit_503_and_listing_does_not_spawn_worker(public):
    manager, client, name, _ = public
    for path in ['/api/tags', '/v1/models']:
        assert (await client.get(path)).status_code == 200
    response = await client.get('/api/ps')
    assert response.status_code == 503 and 'models' not in response.json()
    assert not manager.test.ledger


@pytest.mark.parametrize('path,status', [('/api/chat', 400), ('/api/generate', 400), ('/v1/chat/completions', 404)])
@pytest.mark.parametrize('stream', [False, True])
async def test_missing_model_status_remains_surface_specific(public, path, status, stream):
    manager, client, _, _ = public
    response = await client.post(path, json=request_body('missing', path, stream))
    assert response.status_code == status and not manager.test.ledger


async def test_definition_deleted_during_active_request_retires_exact_generation(public):
    manager, client, name, toml = public
    await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    async with manager.acquire_native_model(name, keep_alive=-1) as lease:
        response = await client.request('DELETE', '/api/delete', json={'model': name})
        assert response.status_code == 200 and not toml.exists()
        assert manager.native_snapshot()[0]['state'] == 'draining' and worker.process.returncode is None
        assert (await lease.worker.chat({'messages': []}))['choices']
    await manager.wait_native_closed(lease.generation)
    assert not manager.native_snapshot() and worker.process.returncode is not None


async def test_metadata_and_toml_edits_cannot_change_an_acquired_snapshot(public, monkeypatch):
    manager, _, name, toml = public
    metadata = {'recommended_sampling': {'top_k': 17}}
    monkeypatch.setattr(manager, '_get_model_metadata_sync', lambda _: metadata)
    async with manager.acquire_native_model(name) as lease:
        metadata['recommended_sampling']['top_k'] = 99
        toml.write_text(toml.read_text().replace('SYSTEM', 'CHANGED').replace('temperature=0.4', 'temperature=0.8'))
        from tllama.native_routes import lease_metadata
        from tllama.native_requests import prepare_ollama_chat
        body = prepare_ollama_chat(OllamaChatRequest(model=name, messages=[]), resolved=lease.resolved,
                                   metadata_info=lease_metadata(lease)).body()
        assert body['top_k'] == 17 and body['temperature'] == 0.4
        assert body['messages'][0]['content'] == 'SYSTEM'


async def test_weight_change_during_header_scan_retries_before_spawn(public, monkeypatch):
    manager, _, name, _ = public
    scans = []
    weights = manager.local_models_dir / 'a.gguf'
    def scan(_):
        scans.append(1)
        if len(scans) == 1:
            weights.write_bytes(b'replaced weights')
        return {'recommended_sampling': {'top_k': len(scans)}}
    monkeypatch.setattr(manager, '_get_model_metadata_sync', scan)
    async with manager.acquire_native_model(name) as lease:
        assert len(scans) == 2 and lease.definition.metadata['recommended_sampling']['top_k'] == 2
    assert len(manager.test.ledger) == 1


@pytest.mark.parametrize('point', ['headers', 'body'])
async def test_send_failure_closes_lease_even_when_body_never_started(public, point):
    manager, _, name, _ = public
    response = await serve_native(manager, OllamaChatRequest(model=name, messages=[{'role':'user','content':'hi'}],
                                  stream=True, keep_alive=0), api='ollama')
    assert manager.native_snapshot()[0]['leases'] == 1
    async def send(message):
        if message['type'] == ('http.response.start' if point == 'headers' else 'http.response.body'):
            raise OSError('client disconnected')
    async def receive():
        await asyncio.Event().wait()
    with pytest.raises((OSError, ClientDisconnect)):
        await asyncio.wait_for(response({'type':'http','asgi':{'spec_version':'2.4'}}, receive, send), timeout=2)
    await wait_until(lambda: not manager.native_snapshot())


async def test_public_non_stream_disconnect_cancels_upstream_without_unloading_other_users(public):
    manager, client, name, _ = public
    # The stub holds ignore_eos requests until their TCP connection is closed.
    prepared = request_body(name, '/api/chat')
    # Inject only at the private fixture boundary, not through a public API option.
    original = manager._native._factory
    def factory(spec):
        worker = original(spec)
        chat = worker.chat
        async def held(body):
            return await chat({**body, 'stub': 'hold'})
        worker.chat = held
        return worker
    manager._native._factory = factory
    task = asyncio.create_task(client.post('/api/chat', json=prepared))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0
    assert manager.test.ledger[0].process.returncode is None


def stream_mode(manager, mode, method='chat_stream'):
    original = manager._native._factory
    def factory(spec):
        worker = original(spec)
        stream = getattr(worker, method)
        setattr(worker, method, lambda body: stream({**body, 'stub': mode}))
        return worker
    manager._native._factory = factory


@pytest.mark.parametrize('cancel_queued', [False, True])
async def test_public_unload_preserves_active_and_admitted_queued_streams(public, cancel_queued):
    manager, client, name, _ = public
    stream_mode(manager, 'controlled')
    first = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
    second = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    try:
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 2)
        if cancel_queued:
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 1)
        unloaded = await client.post('/api/chat', json={'model': name, 'messages': [], 'keep_alive': 0})
        assert unloaded.json()['done_reason'] == 'unload'
        snapshot = manager.native_snapshot()[0]
        assert snapshot['state'] == 'draining' and manager.test.ledger[0].process.returncode is None
    finally:
        (manager.test.roots[0] / 'release').touch()
        results = await asyncio.gather(first, second, return_exceptions=True)
    assert results[0].status_code == 200 and json.loads(results[0].text.splitlines()[-1])['done']
    if not cancel_queued:
        assert results[1].status_code == 200 and json.loads(results[1].text.splitlines()[-1])['done']
    await wait_until(lambda: not manager.native_snapshot())
    assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode is not None


async def test_public_stream_client_cancellation_closes_upstream_response(public):
    manager, client, name, _ = public
    stream_mode(manager, 'hold')
    task = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0
    assert manager.test.ledger[0].process.returncode is None


@pytest.mark.parametrize('spec_version', ['2.0', '2.3', '2.4'])
@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
async def test_stream_disconnect_interrupts_wait_for_next_native_token(public, spec_version, path):
    manager, client, name, _ = public
    stream_mode(manager, 'hold', 'completion_stream' if path == '/api/generate' else 'chat_stream')
    request = (OllamaGenerateRequest(**request_body(name, path, True)) if path == '/api/generate'
               else OllamaChatRequest(**request_body(name, path, True)))
    if path.startswith('/v1/'):
        from tllama.schemas.openai import ChatCompletionRequest
        request = ChatCompletionRequest(**request_body(name, path, True))
    response = await serve_native(manager, request, api='openai' if path.startswith('/v1/') else 'ollama',
                                  kind='completion' if path == '/api/generate' else 'chat')
    output = asyncio.Event()
    sent = []
    async def send(message):
        # Some servers silently discard sends after a closed connection.
        sent.append(message)
        if message['type'] == 'http.response.body' and message.get('body'):
            output.set()
    async def receive():
        await output.wait()
        return {'type': 'http.disconnect'}
    await asyncio.wait_for(response({'type':'http', 'asgi':{'spec_version':spec_version}}, receive, send), timeout=2)
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0
    assert not any(b'"done": true' in frame.get('body', b'') or b'[DONE]' in frame.get('body', b'')
                   for frame in sent)
    worker = manager.test.ledger[0]
    assert worker.process.returncode is None
    followup = await client.post(path, json=request_body(name, path))
    assert followup.status_code == 200 and len(manager.test.ledger) == 1


@pytest.mark.parametrize('spec_version', ['2.3', '2.4'])
@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
async def test_tcp_stream_disconnect_releases_native_slot_before_next_token(public, path, spec_version):
    manager, _, name, _ = public
    stream_mode(manager, 'hold', 'completion_stream' if path == '/api/generate' else 'chat_stream')
    async def public_app(scope, receive, send):
        # Exercise both Starlette branches regardless of installed Uvicorn.
        scope = {**scope, 'asgi':{**scope.get('asgi', {}), 'spec_version':spec_version}}
        await app(scope, receive, send)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(public_app, http='h11', loop='asyncio', lifespan='off',
                                          access_log=False, log_level='warning', timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await wait_until(lambda: server.started)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}',
                                     trust_env=False, timeout=3) as client:
            async with client.stream('POST', path, json=request_body(name, path, True)) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if 'Ahoj světe' in line:
                        break
                else:
                    pytest.fail('Missing first native token')
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
            await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
            worker = manager.test.ledger[0]
            assert worker.process.returncode is None
            result = await client.post(path, json=request_body(name, path))
            assert result.status_code == 200 and len(manager.test.ledger) == 1
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=3)
        finally:
            sock.close()


async def test_cancelled_loading_client_does_not_cancel_other_public_requests(public):
    manager, client, name, _ = public
    manager.test.options['mode'] = 'loading'
    tasks = [asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat'))) for _ in range(4)]
    await wait_until(lambda: bool(manager.test.ledger))
    tasks[0].cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert all(result.status_code == 200 for result in results[1:])
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('mode', ['error', 'error-stream', 'finish-without-done'])
@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_public_stream_errors_include_context_open_failure_and_no_success(public, mode, path):
    manager, client, name, _ = public
    stream_mode(manager, mode)
    response = await client.post(path, json=request_body(name, path, True))
    assert response.status_code == 200 and '"error"' in response.text
    assert '[DONE]' not in response.text and '"done": true' not in response.text
    assert manager.native_snapshot()[0]['leases'] == 0


async def test_public_crash_is_not_replayed_and_next_independent_request_can_reload(public):
    manager, client, name, _ = public
    original = manager._native._factory
    def factory(spec):
        worker = original(spec)
        if len(manager.test.ledger) == 1:
            chat = worker.chat
            async def crash(body):
                return await chat({**body, 'stub': 'crash'})
            worker.chat = crash
        return worker
    manager._native._factory = factory
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 500
    await wait_until(lambda: not manager.native_snapshot())
    requests = [r for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/chat/completions']
    assert len(requests) == 1
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    assert len(manager.test.ledger) == 2


@pytest.mark.parametrize('valid', [False, True])
async def test_public_hardware_smoke_entry_point(public, valid):
    manager, _, name, _ = public
    root = manager.models_dir.parent / 'cli-worker'
    root.mkdir()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'api-smoke.py'
    env = {**os.environ, 'PYTHONPATH':str(script.parent.parent / 'src'),
           'STUB_ROOT':str(root), 'TLLAMA_RUNTIME_DEVICE':'none'}
    process = await asyncio.create_subprocess_exec(sys.executable, str(script),
        '--model', name if valid else 'missing', '--models-dir', str(manager.models_dir),
        '--executable', str(manager._native_launch_options['executable']), '--load-timeout', '5', '--request-timeout','5',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    try:
        output, error = await asyncio.wait_for(process.communicate(), timeout=20)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    stages = [json.loads(line) for line in output.decode().splitlines() if line.startswith('{')]
    assert process.returncode == (0 if valid else 1), (output.decode(), error.decode())
    assert stages[-1]['stage'] == ('manager-closed' if valid else 'error')
    if valid:
        assert len(records(root, 'launch')) == 1
        assert any(stage['stage'] == 'after-stream-cancel' for stage in stages)
        closed = next(stage for stage in stages if stage['stage'] == 'worker-closed')
        assert closed['directory_removed'] and not closed['residency']
    else:
        assert not records(root, 'launch')
