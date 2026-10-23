"""Public TCP/SDK portion of embeddings-multimodal-smoke.py --public.

The independent native controls and vector inspector are supplied by the
reference harness, never imported from the production embedding adapter.
"""

import asyncio
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys

import httpx
import uvicorn
from pydantic import ValidationError
from starlette.requests import ClientDisconnect


async def exercise(args, manager, worker, entries, controls, dimension, emit, notes, reference):
    from tllama import main
    from tllama.routers import ollama, openai

    spec = importlib.util.spec_from_file_location('embedding_public_sdk_helpers',
                                                Path(__file__).with_name('embeddings-api-smoke.py'))
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    ol_sdk, oa_sdk = helpers.load_sdks()
    require = reference.require
    previous = main.model_manager, ollama.model_manager, openai.model_manager
    main.model_manager = ollama.model_manager = openai.model_manager = manager

    async def app(scope, receive, send):
        try:
            await main.app(scope, receive, send)
        except ClientDisconnect:
            pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, http='h11', loop='asyncio', lifespan='off',
        log_level='warning', access_log=False, timeout_graceful_shutdown=5))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    # Ollama's public object has a single image. Preserve that contract; the
    # native content controls with multiple images stay on the /v1 extension.
    ol_entries = []
    for entry in entries[:5]:
        text = ''.join(p['text'] for p in entry['content'] if p['type'] == 'text')
        images = [p['image_url']['url'].split(',', 1)[1] for p in entry['content'] if p['type'] == 'image_url']
        ol_entries.append({'text': text, 'image': images[0]} if images else text)

    async def native(item, label):
        raw = reference.raw_entry(item, worker.props['media_marker'])
        emit('request', source='native', probe=label, path='/v1/embeddings', body={'input': raw})
        payload = await worker.embeddings({'input': raw})
        emit('native-response', probe=label, payload=payload)
        actual = reference.vectors(payload, 1, dimension, 'float', args, emit, notes, label)
        usage = payload.get('usage')
        require(isinstance(usage, dict) and type(usage.get('prompt_tokens')) is int
                and usage['prompt_tokens'] > 0 and usage.get('total_tokens') == usage['prompt_tokens'],
                'Public comparison requires authoritative native media usage')
        return actual[0], usage['prompt_tokens']

    # Each public representation gets an independent raw-native reference.
    # Ollama puts all text before its one image; it cannot express interleaving.
    native_usages, ol_controls, ol_usages = [], [], []

    def inspect(payload, path, count, label, encoding='float'):
        if path == '/api/embed':
            usage = payload.get('prompt_eval_count')
            payload = {'object': 'list', 'data': [
                {'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(payload.get('embeddings', []))],
                'usage': {'prompt_tokens': usage, 'total_tokens': usage}}
        return reference.vectors(payload, count, dimension, encoding, args, emit, notes, label)

    async def public(client, path, value, expected, expected_tokens, label, **extra):
        body = {'model': args.model, 'input': value, **({'truncate': False} if path == '/api/embed' else {}), **extra}
        emit('request', source='public', probe=label, path=path, body=body)
        result = await client.post(path, json=body)
        emit('public-response', probe=label, path=path, status=result.status_code, payload=result.json())
        require(result.status_code == 200, label + ': public embedding request failed')
        usage = result.json().get('prompt_eval_count') if path == '/api/embed' else result.json().get('usage', {}).get('prompt_tokens')
        total = usage if path == '/api/embed' else result.json().get('usage', {}).get('total_tokens')
        require(type(usage) is int and usage == total == expected_tokens, label + ': public/native usage differs')
        actual = inspect(result.json(), path, len(expected), label, extra.get('encoding_format', 'float'))
        reference.compare(expected, actual, args, emit, notes, label)

    async def error(client, path, value, label, **extra):
        body = {'model': args.model, 'input': value, **({'truncate': False} if path == '/api/embed' else {}), **extra}
        emit('request', source='public', probe=label, path=path, body=body)
        response = await client.post(path, json=body)
        emit('public-error', probe=label, path=path, status=response.status_code, payload=response.json())
        require(response.status_code == 400 and isinstance(response.json().get('error'),
                str if path == '/api/embed' else dict), label + ': wrong public client error')

    try:
        await helpers.wait_until(lambda: server.started or serving.done(), 15)
        require(server.started, 'Public diagnostic listener failed to start')
        for i, item in enumerate(entries):
            actual, usage = await native(item, f'public-reference-{i}')
            reference.compare(controls[i:i + 1], [actual], args, emit, notes, f'public-reference-{i}')
            native_usages.append(usage)
        for i, item in enumerate(ol_entries):
            typed = {'content': [reference.text(item)]} if isinstance(item, str) else {
                'content': [reference.text(item['text']), reference.image(item['image'])]}
            actual, usage = await native(typed, f'ollama-reference-{i}')
            ol_controls.append(actual)
            ol_usages.append(usage)
        address = f'http://127.0.0.1:{sock.getsockname()[1]}'
        async with httpx.AsyncClient(base_url=address, trust_env=False, timeout=args.request_timeout) as client:
            for path, items, expected, usages in (('/api/embed', ol_entries, ol_controls, ol_usages),
                                                  ('/v1/embeddings', entries, controls, native_usages)):
                for i, item in enumerate(items):
                    await public(client, path, item, expected[i:i + 1], usages[i], f'public:{path}:single-{i}')
                for label, order in (('batch', list(range(len(items)))), ('repeat', list(range(len(items)))),
                                     ('reordered-duplicates', [2, 0, 4, 1, 0, 3])):
                    await public(client, path, [items[i] for i in order], [expected[i] for i in order], sum(usages[i] for i in order),
                                 f'public:{path}:{label}')
                await public(client, path, items, expected, sum(usages), f'public:{path}:full-width', dimensions=dimension)
                await error(client, path, items[1], path + ':wrong-dimensions', dimensions=dimension - 1)
            await public(client, '/v1/embeddings', entries, controls, sum(native_usages), 'public:base64', encoding_format='base64')
            await error(client, '/api/embed', {'image': 'https://example.invalid/image.png'}, 'remote-image')
            await error(client, '/api/embed', {'image': '%%%'}, 'bad-base64')
            await error(client, '/api/embed', ol_entries[1], 'media-truncate', truncate=True)
            await error(client, '/api/embed', [ol_entries[1]] * 33, 'media-item-bound')
            await error(client, '/v1/embeddings', {'content': [{'type': 'input_audio', 'data': 'x'}]}, 'unsupported-audio')
            # Magic is recognizable; native must actually decode the image.
            import base64
            corrupt = base64.b64encode(b'\xff\xd8\xff\xe0not a decodable JPEG').decode()
            await error(client, '/api/embed', {'image': corrupt}, 'native-decoder-error')
            for path, item, expected, usage in (('/api/embed', ol_entries[3], ol_controls[3:4], ol_usages[3]),
                                               ('/v1/embeddings', entries[3], controls[3:4], native_usages[3])):
                await public(client, path, item, expected, usage, path + ':after-errors')
                body = {'model': args.model, 'input': [item] * 8, **({'truncate': False} if path == '/api/embed' else {})}
                emit('request', source='public', probe='cancel', path=path, body=body)
                task = asyncio.create_task(client.post(path, json=body))
                try:
                    done, _ = await asyncio.wait({task}, timeout=args.cancel_after_ms / 1000)
                    if done:
                        response = await task
                        require(response.status_code == 200, 'Public request failed before cancellation')
                    else:
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    # The reference lease is deliberately retained throughout.
                    await helpers.wait_until(lambda: manager.native_snapshot()[0]['leases'] == 1, args.request_timeout)
                    emit('public-transport-cancel', path=path, pending_at_cancel=not bool(done),
                         native_early_stop_observed=None, inference_started_observed=None)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await public(client, path, item, expected, usage, path + ':after-public-cancel')

            emit('sdk-versions', ollama=version('ollama'), openai=version('openai'))
            ol_hooks = helpers.sdk_hooks(emit, 'ollama')
            sent = []
            async def sent_request(req):
                sent.append(req.url.path)
            ol_hooks['request'].append(sent_request)
            ol = ol_sdk.AsyncClient(host=address, trust_env=False, timeout=args.request_timeout,
                                    event_hooks=ol_hooks)
            try:
                try:
                    result = await ol.embed(model=args.model, input=ol_entries, truncate=False, dimensions=dimension)
                    emit('sdk-capability', api='ollama', high_level_media=True)
                except (ValidationError, TypeError) as exc:
                    # A response-parser failure must fail the diagnostic;
                    # only a local input refusal qualifies for this fallback.
                    if sent:
                        raise
                    emit('sdk-capability', api='ollama', high_level_media=False, error=type(exc).__name__,
                         fallback='SDK HTTP transport and EmbedResponse parsing only')
                    result = await ol._request(ol_sdk.EmbedResponse, 'POST', '/api/embed', json={
                        'model': args.model, 'input': ol_entries, 'truncate': False, 'dimensions': dimension})
                payload = result.model_dump(mode='json', exclude_none=True)
                emit('sdk-response', api='ollama', probe='public-media', payload=payload)
                actual = inspect(payload, '/api/embed', len(ol_entries), 'sdk:ollama-media')
                require(payload['prompt_eval_count'] == sum(ol_usages), 'Ollama SDK/native usage differs')
                reference.compare(ol_controls, actual, args, emit, notes, 'sdk:ollama-media')
                try:
                    await ol._request(ol_sdk.EmbedResponse, 'POST', '/api/embed', json={
                        'model': args.model, 'input': ol_entries[1], 'truncate': False, 'dimensions': dimension - 1})
                except ol_sdk.ResponseError as exc:
                    require(exc.status_code == 400, 'Wrong Ollama SDK media error status')
                    emit('sdk-error', api='ollama', status=exc.status_code)
                else:
                    raise RuntimeError('Ollama SDK accepted unsupported media dimensions')
            finally:
                await ol._client.aclose()
            async with oa_sdk.AsyncOpenAI(base_url=address + '/v1', api_key='local-test', max_retries=0,
                    timeout=args.request_timeout, http_client=oa_sdk.DefaultAsyncHttpxClient(
                        trust_env=False, event_hooks=helpers.sdk_hooks(emit, 'openai'))) as oa:
                for encoding in ('float', 'base64'):
                    # Explicit provider extension; the SDK's standard input
                    # annotation does not promise image embedding support.
                    result = await oa.embeddings.create(model=args.model, input='overridden-by-extra-body',
                        encoding_format=encoding, extra_body={'input': entries})
                    payload = result.model_dump(mode='json', exclude_none=True)
                    emit('sdk-response', api='openai', probe='media-' + encoding, payload=payload)
                    actual = inspect(payload, '/v1/embeddings', len(entries), 'sdk:openai:' + encoding, encoding)
                    require(payload['usage'] == {'prompt_tokens': sum(native_usages), 'total_tokens': sum(native_usages)},
                            'OpenAI SDK/native usage differs')
                    reference.compare(controls, actual, args, emit, notes, 'sdk:openai:' + encoding)
                try:
                    await oa.embeddings.create(model=args.model, input='overridden-by-extra-body',
                        dimensions=dimension - 1, extra_body={'input': entries[1]})
                except oa_sdk.BadRequestError as exc:
                    emit('sdk-error', api='openai', status=exc.status_code)
                else:
                    raise RuntimeError('OpenAI SDK accepted unsupported media dimensions')
            require(len(manager.native_snapshot()) == 1 and manager.native_snapshot()[0]['pid'] == worker.pid,
                    'Public requests replaced the native generation')
            await helpers.wait_until(lambda: manager.native_snapshot()[0]['leases'] == 1, args.request_timeout)
            response = await client.post('/api/embed', json={'model': args.model, 'keep_alive': 0})
            emit('public-unload', status=response.status_code, payload=response.json(), reference_lease_retained=True)
            require(response.status_code == 200 and response.json()['embeddings'] == [], 'Public unload failed')
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, 15)
        finally:
            sock.close()
            main.model_manager, ollama.model_manager, openai.model_manager = previous
