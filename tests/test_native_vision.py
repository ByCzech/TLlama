"""Image validation, native payloads and public ownership over real TCP workers."""

import asyncio
import json
import os
from pathlib import Path
import sys

import gguf
import pytest

from tllama.native_requests import prepare_chat
from tllama.routers import ollama
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, stream_mode
from test_native_requests import resolved
from vision_fixtures import RED_BLUE, BLUE_RED, RED_BLUE_URL, BLUE_RED_URL


@pytest.fixture
async def vision(public):
    manager, client, name, toml = public
    for filename, arch in [('a.gguf','qwen35moe'), ('projector.gguf','clip')]:
        writer = gguf.GGUFWriter(str(manager.local_models_dir / filename), arch)
        if arch == 'clip':
            writer.add_bool('clip.has_vision_encoder', True)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
    toml.write_text(toml.read_text().replace('temperature=0.4', 'temperature=0.4\nstop=["TOML STOP"]') +
                   '\n[mmproj]\nmodel="Local/projector.gguf"\n')
    return manager, client, name, toml


def body(name, path, stream=False, image=RED_BLUE):
    message = {'role':'user', 'content':'Describe the two colored panels.'}
    if path == '/api/chat':
        message['images'] = [image]
        return {'model':name, 'messages':[message], 'stream':stream, 'think':False}
    message['content'] = [{'type':'text','text':message['content']},
                          {'type':'image_url','image_url':{'url':'data:image/png;base64,' + image}}]
    return {'model':name, 'messages':[message], 'stream':stream, 'reasoning_effort':'none'}


def test_mixed_parts_and_ollama_attachments_preserve_order_and_history(resolved):
    messages = [{'role':'user', 'content':[{'type':'text','text':'first'},
                {'type':'image_url','image_url':{'url':RED_BLUE_URL}}, {'type':'text','text':'last'}], 'images':[BLUE_RED]},
                {'role':'assistant','content':'previous answer', 'thinking':'previous reasoning'},
                {'role':'user','content':None,'images':[RED_BLUE]}]
    prepared = prepare_chat(messages, {}, resolved=resolved)
    mapped = prepared.body()['messages']
    assert mapped[0]['role'] == 'system'
    parts = mapped[1]['content']
    assert [p['type'] for p in parts] == ['text','image_url','text','image_url']
    assert parts[1]['image_url']['url'] == RED_BLUE_URL and parts[3]['image_url']['url'] == BLUE_RED_URL
    assert mapped[2]['reasoning_content'] == 'previous reasoning'
    assert mapped[3]['content'][1]['image_url']['url'] == RED_BLUE_URL
    messages[0]['images'][0] = 'changed'
    assert prepared.body()['messages'] == mapped
    assert 'images' not in mapped[1]


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_public_vision_reaches_one_worker_without_stripping_template_or_images(vision, path, stream):
    manager, client, name, _ = vision
    payload = body(name, path, stream)
    if path == '/api/chat':
        payload['options'] = {'stop':['REQUEST STOP']}
    else:
        payload['stop'] = ['REQUEST STOP']
    response = await client.post(path, json=payload)
    assert response.status_code == 200 and '"error"' not in response.text
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    worker = manager.test.ledger[0]
    assert worker.props['modalities']['vision'] is True
    calls = [r['body'] for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/chat/completions']
    assert not any(r['path'] == '/tokenize' for r in records(manager.test.roots[0], 'requests'))
    [request] = calls
    assert request['messages'][0] == {'role':'system', 'content':'SYSTEM'}
    assert request['messages'][-1]['content'][1]['image_url']['url'] == RED_BLUE_URL
    assert request['stop'] == ['REQUEST STOP'] and request['reasoning_effort'] == 'none'
    assert request['temperature'] == 0.4
    assert '--mmproj' in worker.command and worker.props['chat_template'] == '{{ bos_token }}{{ System }}|{{ Prompt }}'
    if stream:
        assert '"done": true' in response.text if path == '/api/chat' else '[DONE]' in response.text
    elif path == '/api/chat':
        assert json.loads(response.json()['message']['content']) == {'left':'red','right':'blue'}
        assert response.json()['prompt_eval_count'] == 6 and response.json()['prompt_eval_cached_count'] == 19
    else:
        assert response.json()['usage']['prompt_tokens'] == 25


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('image', ['http://127.0.0.1:9/x', 'file:///etc/passwd', 'nonsense!!',
    'data:image/png;base64,nonsense!!', 'data:text/plain;base64,' + RED_BLUE])
async def test_bad_image_is_refused_before_model_load_or_remote_fetch(public, path, image):
    manager, client, name, _ = public
    payload = body(name, path)
    if path == '/api/chat':
        payload['messages'][0]['images'] = [image]
    else:
        payload['messages'][0]['content'][1]['image_url']['url'] = image
    response = await client.post(path, json=payload)
    assert response.status_code == 400 and not manager.test.ledger


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_valid_image_to_text_worker_is_preheader_400_without_inference(public, path, stream):
    manager, client, name, _ = public
    response = await client.post(path, json=body(name, path, stream))
    assert response.status_code == 400 and 'does not support multimodal' in response.text
    assert manager.native_snapshot()[0]['leases'] == 0
    assert not any(r['path'] == '/v1/chat/completions' for r in records(manager.test.roots[0], 'requests'))


@pytest.mark.parametrize('capability', [None, False, 1, 'true'])
async def test_projector_configuration_cannot_override_actual_worker_capability(vision, capability):
    manager, client, name, _ = vision
    await manager.preload_native_model(name)
    worker = manager.test.ledger[0]
    worker.props['modalities'] = None if capability is None else {'vision':capability}
    response = await client.post('/api/chat', json=body(name, '/api/chat', True))
    assert response.status_code == 400
    assert not any(r['path'] == '/v1/chat/completions' for r in records(manager.test.roots[0], 'requests'))


async def test_show_advertises_configured_vision_without_starting_a_worker(vision):
    manager, client, name, _ = vision
    response = await client.post('/api/show', json={'model':name})
    assert response.status_code == 200 and 'vision' in response.json()['capabilities']
    assert not manager.test.ledger


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_different_images_share_generation_but_not_image_payload_or_answer(vision, path):
    manager, client, name, _ = vision
    answers = []
    for image in (RED_BLUE, BLUE_RED):
        response = (await client.post(path, json=body(name, path, image=image))).json()
        text = response['message']['content'] if path == '/api/chat' else response['choices'][0]['message']['content']
        answers.append(json.loads(text))
    assert answers == [{'left':'red','right':'blue'}, {'left':'blue','right':'red'}]
    assert len(manager.test.ledger) == 1


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_upstream_image_decode_failure_has_public_error_and_no_success(vision, path, stream):
    manager, client, name, _ = vision
    manager.test.options['mode'] = 'bad-image'
    response = await client.post(path, json=body(name, path, stream))
    assert response.status_code == (200 if stream else 400)
    assert 'Image decoder rejected' in response.text and '"error"' in response.text
    assert '[DONE]' not in response.text and '"done": true' not in response.text
    assert manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_cancelled_image_stream_closes_upstream_but_worker_is_reusable(vision, path):
    manager, client, name, _ = vision
    stream_mode(manager, 'hold')
    task = asyncio.create_task(client.post(path, json=body(name, path, True)))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0
    assert (await client.post(path, json=body(name, path))).status_code == 200
    assert len(manager.test.ledger) == 1


async def test_image_stream_unload_drains_protected_generation(vision):
    manager, client, name, _ = vision
    stream_mode(manager, 'controlled')
    task = asyncio.create_task(client.post('/api/chat', json=body(name, '/api/chat', True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
        generation = manager.native_snapshot()[0]['generation']
        await manager.unload_native_model(name)
        [process] = (await client.get('/api/ps')).json()['models']
        assert process['state'] == 'draining' and process['leases'] == 1
        assert process['size'] is None and process['memory']['reason'] == 'projector_allocation_unreported'
    finally:
        (manager.test.roots[0] / 'release').touch()
        response = await task
    assert '"done": true' in response.text
    await manager.wait_native_closed(generation)
    assert (await client.get('/api/ps')).json() == {'models':[]}


@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_cancelled_nonstream_image_request_closes_upstream(vision, path):
    manager, client, name, _ = vision
    original = manager._native._factory
    def factory(spec):
        worker = original(spec)
        chat = worker.chat
        worker.chat = lambda payload: chat({**payload, 'stub':'hold'})
        return worker
    manager._native._factory = factory
    task = asyncio.create_task(client.post(path, json=body(name, path)))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0 and manager.test.ledger[0].process.returncode is None


@pytest.mark.parametrize('part', [{'type':'media_marker','text':'<injected>'},
    {'type':'input_audio','input_audio':{'url':'file:///etc/passwd'}},
    {'type':'image_url','image_url':{}}, {'type':'text','text':False}])
async def test_unsupported_or_malformed_parts_do_not_reach_native_media_loader(public, part):
    manager, client, name, _ = public
    payload = {'model':name,'messages':[{'role':'user','content':[part]}]}
    assert (await client.post('/v1/chat/completions', json=payload)).status_code == 400
    assert not manager.test.ledger


@pytest.mark.parametrize('mode', ['normal', 'no-vision'])
async def test_vision_hardware_smoke_runs_over_public_tcp_and_checks_image_changes(vision, mode):
    manager, _, name, _ = vision
    root = manager.models_dir.parent / 'vision-cli-worker'
    root.mkdir()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'vision-smoke.py'
    env = {**os.environ, 'PYTHONPATH':str(script.parent.parent / 'src'), 'STUB_ROOT':str(root),
           'STUB_MODE':mode, 'TLLAMA_RUNTIME_DEVICE':'none', 'TLLAMA_CONTEXT_LENGTH':'4096'}
    process = await asyncio.create_subprocess_exec(sys.executable, str(script), '--model', name,
        '--models-dir', str(manager.models_dir), '--executable', str(manager._native_launch_options['executable']),
        '--load-timeout','5','--request-timeout','5', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    try:
        output, error = await asyncio.wait_for(process.communicate(), timeout=20)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    stages = [json.loads(line) for line in output.decode().splitlines() if line.startswith('{')]
    assert process.returncode == (0 if mode == 'normal' else 1), (output.decode(), error.decode())
    assert len(records(root, 'launch')) == 1
    if mode == 'normal':
        assert stages[-1] == {'stage':'manager-closed','residency':[]}
        assert next(s for s in stages if s['stage'] == 'ready')['requested_context'] == 4096
        for api in ('ollama','openai'):
            assert next(s for s in stages if s['stage'] == api + '-image-0')['colors'] == {'left':'red','right':'blue'}
            assert next(s for s in stages if s['stage'] == api + '-image-1')['colors'] == {'left':'blue','right':'red'}
            assert len([s for s in stages if s['stage'].startswith(api + '-image-')]) == 4
        assert next(s for s in stages if s['stage'] == 'after-vision-cancel')['colors'] == {'left':'blue','right':'red'}
        assert next(s for s in stages if s['stage'] == 'worker-closed')['directory_removed']
        assert (root / 'cancelled').exists()
    else:
        assert stages[-1]['stage'] == 'error' and 'no vision capability' in stages[-1]['error']
