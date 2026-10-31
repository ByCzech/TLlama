"""Public text Decision through ASGI, TCP disconnects and owned native workers."""

import asyncio
import anyio
import json
import socket

import httpx
import pytest
import uvicorn
from starlette.requests import ClientDisconnect

from tllama.decision_contract import DECISION_RUNTIME, MAX_REQUEST_BYTES
from tllama.main import app

from test_decision_smoke import header
from test_native_decision import BODY, LLM
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public


PATH = '/v1/systemone'


@pytest.fixture
async def decisions(public, monkeypatch, request):
    manager, client, name, toml = public
    kind = getattr(request, 'param', 'laya')
    monkeypatch.setenv('STUB_DECISION', 'fast')
    monkeypatch.setenv('STUB_DECISION_KIND', kind)
    monkeypatch.setenv('STUB_BUILD', DECISION_RUNTIME)
    header(manager.local_models_dir / 'a.gguf', kind=kind)
    toml.write_text(LLM)
    return manager, client, name, toml, kind


def body(name, **changes):
    return {**BODY, 'model': name, **changes}


def posts(manager):
    return [row['body'] for root in manager.test.roots for row in records(root, 'requests')
            if row['path'] == PATH]


@pytest.mark.parametrize('decisions', ['laya', 'nimble', 'clef'], indirect=True)
async def test_public_typed_result_equals_managed_native_and_keeps_literal_json(decisions):
    manager, client, name, toml, kind = decisions
    original = toml.read_bytes()
    response = await client.post(PATH, json=body(name))
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {'model', 'answers', 'usage'} and result['model'] == name
    async with manager.acquire_native_model(name, operation='decision') as lease:
        native = await lease.worker.decision(BODY)
    assert result == {**native, 'model': name}
    keys = list(result['answers'][next(iter(BODY['questions']))]['probabilities'])
    expected = list(next(iter(BODY['questions'].values()))['criteria'])
    assert keys == (sorted(expected) if kind == 'clef' else expected)
    assert list(result['answers']) == list(BODY['questions'])
    assert posts(manager) == [{**BODY, 'model': lease.worker.spec.alias}] * 2
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    assert toml.read_bytes() == original
    await manager.unload_native_model(name)
    await manager.wait_native_closed(lease.generation)
    assert not lease.worker.runtime_directory.exists()


@pytest.mark.parametrize('count', [1, 27, 255])
async def test_choice_boundaries_and_singleton_confidence(decisions, count):
    _, client, name, _, _ = decisions
    questions = {'choice': {'type': 'choice', 'instructions': '',
                            'criteria': {f'key-{i}': None for i in reversed(range(count))}}}
    response = await client.post(PATH, json=body(name, questions=questions))
    assert response.status_code == 200, response.text
    answer = response.json()['answers']['choice']
    assert len(answer['probabilities']) == count
    if count == 1:
        assert answer['probabilities'] == {'key-0': 1} and answer['confidence'] == 1


async def test_32_questions_work_without_generation_options_or_fabricated_output_tokens(decisions):
    _, client, name, _, _ = decisions
    questions = {f'q{i}': {'type': 'noul', 'instructions': False} for i in range(32)}
    response = await client.post(PATH, json=body(name, questions=questions))
    assert response.status_code == 200 and list(response.json()['answers']) == list(questions)
    assert response.json()['usage']['input_tokens'] > 0 and response.json()['usage']['output_tokens'] == 0


@pytest.mark.parametrize('state', [False, 0, '', [], {}])
async def test_non_null_json_state_is_not_confused_with_missing_input(decisions, state):
    manager, client, name, _, _ = decisions
    response = await client.post(PATH, json=body(name, state=state))
    assert response.status_code == 200, response.text
    assert posts(manager)[0]['state'] == state and type(posts(manager)[0]['state']) is type(state)


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('part', [
    {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/a.png'}},
    {'type': 'image_url', 'image_url': 'data:image/png;base64,aW1hZ2U='},
    {'type': 'input_audio', 'input_audio': {'data': 'YXVkaW8=', 'format': 'wav'}},
])
async def test_chat_state_media_cannot_bypass_the_text_contract(decisions, wrapped, part):
    manager, client, name, _, _ = decisions
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'literal'}, part]}]
    state = {'messages': messages} if wrapped else messages
    response = await client.post(PATH, json=body(name, state=state))
    assert response.status_code == 400 and 'Media parts' in response.text
    assert not manager.test.ledger and not posts(manager)


async def test_media_shaped_data_outside_native_chat_layout_stays_literal(decisions):
    manager, client, name, _, _ = decisions
    state = {'ticket': {'messages': [{'content': [{'type': 'image_url', 'image_url': 'https://example.invalid/a.png'}]}]}}
    response = await client.post(PATH, json=body(name, state=state))
    assert response.status_code == 200 and posts(manager)[0]['state'] == state


@pytest.mark.parametrize('change', [
    {'model': None}, {'model': 1}, {'model': ''}, {'model': ' '}, {'state': None},
    {'questions': {}}, {'questions': []}, {'questions': {'q': {'type': 'noul'}}},
    {'questions': {'q': {'type': 'unknown', 'instructions': ''}}},
    {'questions': {'q': {'type': 'noul', 'instructions': '', 'extra': 1}}},
    {'questions': {'q': {'type': 'choice', 'instructions': '', 'criteria': {}}}},
    {'questions': {'q': {'type': 'choice', 'instructions': '', 'criteria': {str(i): None for i in range(256)}}}},
    {'questions': {'q': {'type': 'score', 'instructions': '', 'criteria': ['one']}}},
    {'questions': {'q': {'type': 'score', 'instructions': '', 'criteria': list(range(11))}}},
    {'questions': {'q': {'type': 'noul', 'instructions': '', 'criteria': []}}},
    {'questions': {str(i): {'type': 'noul', 'instructions': ''} for i in range(33)}},
    {'images': []}, {'images': ['https://example.invalid/a.png']}, {'audio': 'x'},
    {'stream': False}, {'options': {}}, {'keep_alive': 0}, {'raw': True},
    {'system': ''}, {'template': ''}, {'temperature': 0},
])
async def test_invalid_contract_fails_before_model_admission(decisions, change):
    manager, client, name, _, _ = decisions
    response = await client.post(PATH, json=body(name, **change))
    assert response.status_code == 400, response.text
    assert set(response.json()) == {'error'}
    assert response.json()['error']['type'] == 'invalid_request_error'
    assert not manager.test.ledger and not manager.native_snapshot()


@pytest.mark.parametrize('data', [
    b'', b'{', b'null', b'[]', b'false', b'{"model":"a"}',
    b'{"model":"a","model":"other"}',
    b'{"model":"a","state":1,"questions":{"q":{},"\\u0071":{}}}',
    b'{"model":"a","state":NaN,"questions":{}}',
    b'{"model":"a","state":Infinity,"questions":{}}', b'{"state":"\xff"}',
    b'[' * 1100 + b'0' + b']' * 1100,
])
async def test_invalid_or_ambiguous_json_never_loads_a_model(decisions, data):
    manager, client, _, _, _ = decisions
    response = await client.post(PATH, content=data, headers={'Content-Type': 'application/json'})
    assert response.status_code == 400 and isinstance(response.json()['error'], dict)
    assert not manager.test.ledger


async def test_raw_body_limit_stops_streaming_before_decode_and_admission(decisions):
    manager, client, _, _, _ = decisions
    consumed = []
    async def data():
        for i in range(4):
            consumed.append(i)
            yield b' ' * (MAX_REQUEST_BYTES // 2 + 1)
    response = await client.post(PATH, content=data(), headers={'Content-Type': 'application/json'})
    assert response.status_code == 413 and consumed == [0, 1]
    assert not manager.test.ledger


@pytest.mark.parametrize('content_type,status', [('application/json', 200), ('application/problem+json', 200),
                                               ('text/plain', 200), ('multipart/form-data', 415)])
async def test_content_types_follow_existing_json_middleware(decisions, content_type, status):
    manager, client, name, _, _ = decisions
    response = await client.post(PATH, content=json.dumps(body(name)), headers={'Content-Type': content_type})
    assert response.status_code == status, response.text
    assert bool(manager.test.ledger) is (status == 200)


@pytest.mark.parametrize('mode,status', [('bad-sum', 500), ('bad-confidence', 500), ('wrong-model', 500),
    ('wrong-choice', 500), ('wrong-score', 500), ('wrong-legend', 500), ('wrong-order', 500),
    ('nan', 500), ('bad-usage', 500), ('no-usage', 500), ('http400', 400), ('http500', 500), ('http501', 501)])
async def test_native_errors_and_corruption_do_not_return_partial_success_and_recover(decisions, mode, status):
    manager, client, name, _, _ = decisions
    await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    behavior = manager.test.roots[0] / 'decision-mode'
    behavior.write_text(mode)
    response = await client.post(PATH, json=body(name))
    assert response.status_code == status and set(response.json()) == {'error'}
    assert set(response.json()['error']) == {'message', 'type', 'param', 'code'}
    assert manager.native_snapshot()[0]['leases'] == 0 and worker.process.returncode is None
    behavior.write_text('fast')
    recovered = await client.post(PATH, json=body(name))
    assert recovered.status_code == 200 and len(manager.test.ledger) == 1


@pytest.mark.parametrize('kind,template', [(None, True), ('pplx-decider', True), ('laya', False)])
async def test_wrong_or_unintegrated_metadata_fails_before_load(decisions, kind, template):
    manager, client, name, _, _ = decisions
    header(manager.local_models_dir / 'a.gguf', kind=kind, template=template)
    response = await client.post(PATH, json=body(name))
    assert response.status_code == 400 and not manager.test.ledger
    shown = await client.post('/api/show', json={'model': name})
    assert 'decision' not in shown.json()['capabilities'] and not manager.test.ledger


async def test_missing_model_bad_toml_startup_and_shutdown_have_distinct_statuses(decisions, monkeypatch):
    manager, client, name, toml, _ = decisions
    missing = await client.post(PATH, json=body('missing'))
    assert missing.status_code == 404
    toml.write_text('[llm\n')
    broken = await client.post(PATH, json=body(name))
    assert broken.status_code == 500
    toml.write_text(LLM)
    monkeypatch.setenv('STUB_DECISION', 'no-decisions')
    startup = await client.post(PATH, json=body(name))
    assert startup.status_code == 500 and manager.native_snapshot() == []
    await manager.shutdown()
    closed = await client.post(PATH, json=body(name))
    assert closed.status_code == 503


@pytest.mark.parametrize('embedding', [False, True])
async def test_wrong_mode_does_not_evict_a_resident_worker(decisions, embedding):
    manager, client, _, _, _ = decisions
    name, _ = manager.test.define(name='other', extra='[runtime]\nembedding=true' if embedding else '')
    await manager.preload_native_model(name, keep_alive=-1)
    before = manager.native_snapshot()
    response = await client.post(PATH, json=body(name))
    assert response.status_code == 400 and manager.native_snapshot() == before
    assert len(manager.test.ledger) == 1 and not posts(manager)


async def test_public_worker_crash_is_not_replayed_and_next_request_reloads(decisions):
    manager, client, name, _, _ = decisions
    await manager.preload_native_model(name, keep_alive=-1)
    (manager.test.roots[0] / 'decision-mode').write_text('crash')
    assert (await client.post(PATH, json=body(name))).status_code == 500
    await wait_until(lambda: not manager.native_snapshot())
    assert len(posts(manager)) == 1
    assert (await client.post(PATH, json=body(name))).status_code == 200
    assert len(manager.test.ledger) == 2


async def test_existing_public_unload_retires_decision_without_generation_inference(decisions):
    manager, client, name, _, _ = decisions
    assert (await client.post(PATH, json=body(name))).status_code == 200
    worker = manager.test.ledger[0]
    unloaded = await client.post('/api/generate', json={'model': name, 'keep_alive': 0})
    assert unloaded.status_code == 200 and unloaded.json()['done_reason'] == 'unload'
    await manager.wait_native_closed(1)
    assert manager.native_snapshot() == [] and not worker.runtime_directory.exists()
    assert len(posts(manager)) == 1
    assert not any(row['path'] in {'/completion', '/v1/chat/completions'}
                   for row in records(manager.test.roots[0], 'requests'))


async def test_public_queued_cancellation_preserves_other_request_and_owned_unload(decisions):
    manager, client, name, _, _ = decisions
    await manager.preload_native_model(name, keep_alive=-1)
    root, worker = manager.test.roots[0], manager.test.ledger[0]
    behavior = root / 'decision-mode'
    behavior.write_text('hold')
    first = asyncio.create_task(client.post(PATH, json=body(name)))
    second = None
    try:
        await wait_until(lambda: records(root, 'started'))
        second = asyncio.create_task(client.post(PATH, json=body(name)))
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 2)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert manager.native_snapshot()[0]['leases'] == 1 and len(posts(manager)) == 1
        await manager.unload_native_model(name)
        assert worker.process.returncode is None
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        tasks = [x for x in (first, second) if x is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    await manager.wait_native_closed(1)
    assert not worker.runtime_directory.exists() and manager.native_snapshot() == []


@pytest.mark.parametrize('abort', ['disconnect', 'cancel-handler'])
async def test_public_cancellation_survives_a_consumed_transport_cancel(decisions, monkeypatch, abort):
    """A transient transport boundary must not consume the owner's stop intent."""
    from starlette.requests import Request
    from tllama.native_decision import decision_response
    manager, _, name, _, _ = decisions
    await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    entered, consumed, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
    disconnect = asyncio.Event()
    transport_task = None
    async def transport_boundary(payload):
        nonlocal transport_task
        transport_task = asyncio.current_task()
        try:
            entered.set()
            try:
                await anyio.sleep_forever()
            except asyncio.CancelledError:
                # Reproduce an inner transport scope absorbing one cancel.
                consumed.set()
            await anyio.sleep_forever()
        finally:
            released.set()
    monkeypatch.setattr(worker, 'decision', transport_boundary)
    wire = json.dumps(body(name)).encode()
    delivered = False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type':'http.request', 'body':wire, 'more_body':False}
        await disconnect.wait()
        return {'type':'http.disconnect'}
    request = Request({'type':'http', 'method':'POST', 'path':PATH,
                       'headers':[(b'content-type', b'application/json')]}, receive)
    task = asyncio.create_task(decision_response(request, manager))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert manager.native_snapshot()[0]['leases'] == 1
        if abort == 'disconnect':
            disconnect.set()
        else:
            task.cancel()
        await asyncio.wait_for(consumed.wait(), 2)
        # asyncio.wait observes completion without injecting a second cancel,
        # which would hide the exact regression being exercised.
        done, _ = await asyncio.wait({task}, timeout=1)
        assert done, 'Public cleanup stalled after one transport cancellation was consumed'
        with pytest.raises(ClientDisconnect if abort == 'disconnect' else asyncio.CancelledError):
            await task
        assert released.is_set() and manager.native_snapshot()[0]['leases'] == 0
    finally:
        # A second cancellation is only test cleanup after a failed assertion.
        task.cancel()
        if transport_task is not None:
            transport_task.cancel()
        await asyncio.gather(*(item for item in (task, transport_task) if item is not None),
                             return_exceptions=True)


async def test_real_tcp_disconnect_releases_native_response_and_public_lease(decisions):
    manager, client, name, _, _ = decisions
    await manager.preload_native_model(name, keep_alive=-1)
    root = manager.test.roots[0]
    behavior = root / 'decision-mode'
    behavior.write_text('hold')
    async def public_app(scope, receive, send):
        try:
            await app(scope, receive, send)
        except ClientDisconnect:
            pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(public_app, http='h11', loop='asyncio', lifespan='off',
        access_log=False, log_level='warning', timeout_graceful_shutdown=1))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    writer = None
    try:
        await wait_until(lambda: server.started)
        _, writer = await asyncio.open_connection('127.0.0.1', sock.getsockname()[1])
        payload = json.dumps(body(name)).encode()
        writer.write(f'POST {PATH} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n'.encode() + payload)
        await writer.drain()
        await wait_until(lambda: records(root, 'started'))
        writer.close()
        await writer.wait_closed()
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
        await wait_until(lambda: records(root, 'cancelled'))
        behavior.write_text('fast')
        assert (await client.post(PATH, json=body(name))).status_code == 200
        assert len(manager.test.ledger) == 1
    finally:
        if writer is not None:
            writer.close()
        server.should_exit = True
        await asyncio.wait_for(serving, 3)
        sock.close()


@pytest.mark.parametrize('decisions', ['laya', 'nimble', 'clef'], indirect=True)
async def test_show_advertises_only_text_decision_without_starting_a_worker(decisions):
    manager, client, name, _, _ = decisions
    shown = await client.post('/api/show', json={'model': name})
    assert shown.status_code == 200 and shown.json()['capabilities'] == ['decision']
    assert not manager.test.ledger


def test_openapi_describes_the_nonstandard_extension_request():
    spec = app.openapi()['paths'][PATH]['post']
    assert 'Decision API extension' in spec['tags']
    schema = spec['requestBody']['content']['application/json']['schema']
    assert schema['required'] == ['model', 'state', 'questions'] and schema['additionalProperties'] is False
