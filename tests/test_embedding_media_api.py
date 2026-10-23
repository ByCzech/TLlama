"""Public image inputs compared with independent native marker/media requests."""

import asyncio
import base64
import math
import struct

import pytest

from tllama.schemas.embeddings import OllamaEmbedRequest, OpenAIEmbeddingRequest
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public
from test_embedding_api import embeddings, calls, ids
from test_native_embedding_projector import PROJECTOR, projector
from vision_fixtures import RED_BLUE_URL, BLUE_RED_URL


PATHS = ['/api/embed', '/v1/embeddings']
TEXT = 'a\nb; literal \\n. 日本語 🐈'


@pytest.fixture
async def media(embeddings, monkeypatch):
    manager, client, name, toml, header = embeddings
    monkeypatch.setenv('STUB_MULTIMODAL_EMBEDDINGS', 'fast')
    monkeypatch.setenv('STUB_PUBLIC_CTX', '1024')
    header(pooling=3, context=1024)
    projector(manager.local_models_dir / 'vision.gguf')
    toml.write_text(toml.read_text().replace('n_ctx=16', 'n_ctx=1024\nn_batch=1024\nn_ubatch=1024') + PROJECTOR)
    return manager, client, name, toml


def entry(url=RED_BLUE_URL, text=TEXT):
    return {'content': [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': url}}]}


def request(name, path, value, **extra):
    return {'model': name, 'input': value, **({'truncate': False} if path == '/api/embed' else {}), **extra}


def vector(payload, path):
    return payload['embeddings'] if path == '/api/embed' else [r['embedding'] for r in payload['data']]


@pytest.mark.parametrize('path', PATHS)
async def test_public_literal_images_order_duplicates_and_usage_match_raw_native(media, path):
    manager, client, name, _ = media
    a, b = entry(), entry(BLUE_RED_URL)
    if path == '/v1/embeddings':
        a['content'].extend([{'type': 'text', 'text': '\nEnd.'}, b['content'][1]])
        b['content'].extend([{'type': 'text', 'text': '\nEnd.'}, entry()['content'][1]])
    public_items = [a, b, a] if path == '/v1/embeddings' else [
        {'text': TEXT, 'image': RED_BLUE_URL.split(',')[1]},
        {'text': TEXT, 'image': BLUE_RED_URL}, {'text': TEXT, 'image': RED_BLUE_URL}]
    response = await client.post(path, json=request(name, path, public_items))
    assert response.status_code == 200, response.text
    vectors = vector(response.json(), path)
    assert vectors[0] == vectors[2] and vectors[0] != vectors[1]
    assert all(math.hypot(*v) == pytest.approx(1) for v in vectors)
    assert (response.json().get('prompt_eval_count') or response.json()['usage']['prompt_tokens']) == 150
    async with manager.acquire_native_model(name, operation='text_embedding') as lease:
        for item, actual in zip([a, b, a], vectors):
            prompt, images = '', []
            for part in item['content']:
                if part['type'] == 'text':
                    prompt += part['text']
                else:
                    prompt += '<stub-media>'
                    images.append(part['image_url']['url'].split(',')[1])
            native = await lease.worker.embeddings({'input': {'prompt_string': prompt, 'multimodal_data': images}})
            assert actual == native['data'][0]['embedding']
    assert not [r for r in records(manager.test.roots[0], 'requests') if r['path'] == '/tokenize']
    assert all(c['model'] == 'tllama-worker' for c in calls(manager))
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


async def test_mixed_text_media_batch_and_base64_keep_authoritative_counts(media):
    manager, client, name, _ = media
    value = [TEXT, entry(), {'content': [{'type': 'image_url', 'image_url': {'url': BLUE_RED_URL}}]}, TEXT]
    floating = await client.post('/v1/embeddings', json=request(name, '/v1/embeddings', value))
    assert floating.status_code == 200, floating.text
    encoded = await client.post('/v1/embeddings', json=request(name, '/v1/embeddings', value, encoding_format='base64'))
    assert encoded.status_code == 200, encoded.text
    for actual, row in zip(vector(floating.json(), '/v1/embeddings'), encoded.json()['data']):
        assert struct.unpack('<8f', base64.b64decode(row['embedding'], validate=True)) == pytest.approx(actual, rel=1e-7)
    assert floating.json()['usage'] == encoded.json()['usage'] == {
        'prompt_tokens': 2 * len(ids(TEXT)) + 100, 'total_tokens': 2 * len(ids(TEXT)) + 100}
    assert calls(manager)[0]['input'] == [ids(TEXT)] and calls(manager)[1]['input'] == [entry()]


@pytest.mark.parametrize('path,value', [
    ('/api/embed', {'image': 'https://example.com/a.png'}),
    ('/api/embed', {'image': 'file:///etc/passwd'}), ('/api/embed', {'image': '%%%'}),
    ('/api/embed', {'image': RED_BLUE_URL, 'audio': 'ignored?'}),
    ('/api/embed', {'text': 1}), ('/api/embed', {'text': ''}),
    ('/api/embed', [{'image': RED_BLUE_URL}] * 33),
    ('/v1/embeddings', {'content': [{'type': 'input_video', 'url': 'x'}]}),
    ('/v1/embeddings', {'content': [], 'prompt_string': 'x'}),
    ('/v1/embeddings', {'content': [{'type': 'image_url', 'image_url': {'url': RED_BLUE_URL, 'detail': 'low'}}]}),
    ('/v1/embeddings', {'content': [{'type': 'image_url', 'image_url': RED_BLUE_URL}]}),
    ('/v1/embeddings', {'content': [entry()['content'][1]] * 33}),
    ('/v1/embeddings', {'content': [{'type': 'text', 'text': '', 'cache_control': {}}]}),
])
async def test_invalid_media_is_refused_before_admission(media, path, value):
    manager, client, name, _ = media
    response = await client.post(path, json=request(name, path, value))
    assert response.status_code == 400, response.text
    assert not manager.test.ledger


async def test_default_image_truncation_is_explicitly_refused(media):
    manager, client, name, _ = media
    response = await client.post('/api/embed', json={'model': name, 'input': {'image': RED_BLUE_URL}})
    assert response.status_code == 400 and 'truncate=false' in response.text and not manager.test.ledger


@pytest.mark.parametrize('reason', ['no-projector', 'no-vision', 'marker', 'missing-marker'])
async def test_media_requires_live_capability_and_unambiguous_marker(media, reason):
    manager, client, name, toml = media
    if reason == 'no-projector':
        toml.write_text(toml.read_text().replace(PROJECTOR, ''))
    await manager.preload_native_model(name)
    worker = manager.test.ledger[0]
    if reason == 'no-vision':
        worker.props['modalities']['vision'] = False
    if reason == 'missing-marker':
        worker.props.pop('media_marker')
    value = entry(text='<stub-media>') if reason == 'marker' else entry()
    response = await client.post('/v1/embeddings', json=request(name, '/v1/embeddings', value))
    assert response.status_code == (500 if reason == 'missing-marker' else 400) and not calls(manager)
    assert manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('pooling,count,status', [(3, 1023, 200), (3, 1024, 400), (1, 1024, 200)])
async def test_media_usage_boundary_is_checked_after_native_work(media, pooling, count, status):
    manager, client, name, toml = media
    toml.write_text(toml.read_text().replace('embedding=true', f'embedding=true\npooling_type={pooling}'))
    await manager.preload_native_model(name)
    (manager.test.roots[0] / 'media-token-count').write_text(str(count))
    response = await client.post('/v1/embeddings', json=request(name, '/v1/embeddings', entry()))
    assert response.status_code == status, response.text
    assert len(calls(manager)) == 1  # Native counted the image, not /tokenize.
    if status == 200:
        assert response.json()['usage'] == {'prompt_tokens': count, 'total_tokens': count}


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('mode,status', [('no-usage', 500), ('boolean-usage', 500), ('bad-usage', 500),
    ('over-limit', 400), ('bad-norm', 500), ('wrong-dimension', 500), ('nan', 500), ('out-of-range', 500)])
async def test_native_faults_do_not_fabricate_usage_or_return_partial_batch(media, path, mode, status):
    manager, client, name, _ = media
    await manager.preload_native_model(name)
    behavior = manager.test.roots[0] / 'media-behavior'
    behavior.write_text(mode)
    value = [entry(), entry(BLUE_RED_URL)] if path == '/v1/embeddings' else [{'image': RED_BLUE_URL}, {'image': BLUE_RED_URL}]
    response = await client.post(path, json=request(name, path, value))
    assert response.status_code == status, response.text
    assert isinstance(response.json()['error'], str if path == '/api/embed' else dict)
    behavior.write_text('fast')
    good = await client.post(path, json=request(name, path, value))
    assert good.status_code == 200 and len(manager.test.ledger) == 1


@pytest.mark.parametrize('mode,status', [('fast', 400), ('decoder-error', 500)])
async def test_exact_pinned_decoder_error_is_client_error_and_other_500_survives(media, mode, status):
    manager, client, name, _ = media
    await manager.preload_native_model(name)
    (manager.test.roots[0] / 'media-behavior').write_text(mode)
    # Recognizable JPEG magic passes media validation, but the PNG fixture decoder cannot decode it.
    invalid = base64.b64encode(b'\xff\xd8\xff\xe0' + b'not a decodable image').decode()
    response = await client.post('/api/embed', json=request(name, '/api/embed', {'image': invalid}))
    assert response.status_code == status, response.text
    recovered = await client.post('/api/embed', json=request(name, '/api/embed', {'image': RED_BLUE_URL}))
    assert recovered.status_code == 200 and len(manager.test.ledger) == 1


async def test_cancelled_public_media_releases_slot_and_unloads(media):
    manager, client, name, _ = media
    await manager.preload_native_model(name)
    worker, root = manager.test.ledger[0], manager.test.roots[0]
    hold = root / 'media-hold'
    hold.touch()
    task = asyncio.create_task(client.post('/api/embed', json=request(name, '/api/embed', {'image': RED_BLUE_URL})))
    try:
        await wait_until(lambda: records(root, 'started'))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: records(root, 'cancelled'))
        hold.unlink()
        recovered = await client.post('/api/embed', json=request(name, '/api/embed', {'image': RED_BLUE_URL}))
        assert recovered.status_code == 200 and len(manager.test.ledger) == 1
        assert manager.native_snapshot()[0]['leases'] == 0
        await client.post('/api/embed', json={'model': name, 'keep_alive': 0})
        await wait_until(lambda: not manager.native_snapshot())
        assert worker.process.returncode is not None and not worker.runtime_directory.exists()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_media_byte_limits_are_independent_of_pixel_and_token_guesses(monkeypatch):
    import tllama.embedding_media as limits
    monkeypatch.setattr(limits, 'MAX_IMAGE_BYTES', 8)
    with pytest.raises(ValueError, match='4 MiB'):
        OllamaEmbedRequest(model='a', input={'image': RED_BLUE_URL}, truncate=False)
    monkeypatch.setattr(limits, 'MAX_IMAGE_BYTES', 1024)
    monkeypatch.setattr(limits, 'MAX_MEDIA_BYTES', 1)
    with pytest.raises(ValueError, match='16 MiB'):
        OpenAIEmbeddingRequest(model='a', input=entry())
