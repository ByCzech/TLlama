"""Embedding public contracts through owned TCP workers and real SDK parsing."""

import asyncio
import base64
from dataclasses import replace
import json
import math
import socket
import struct

import gguf
import httpx
import pytest
import uvicorn
from starlette.requests import ClientDisconnect

from tllama.main import app
from tllama.helpers.gguf_metadata import read_gguf_metadata
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public


PATHS = ['/api/embed', '/api/embeddings', '/v1/embeddings']


def body(name, path, text='hello', **extra):
    return {'model': name, 'prompt' if path == '/api/embeddings' else 'input': text, **extra}


@pytest.fixture
async def embeddings(public, monkeypatch):
    manager, client, name, toml = public
    monkeypatch.setenv('STUB_PUBLIC_EMBEDDINGS', '1')
    monkeypatch.setenv('STUB_PUBLIC_CTX', '16')
    weights = manager.local_models_dir / 'a.gguf'

    def header(*, pooling=1, dimension=8, output_dimension=None, decision=None):
        writer = gguf.GGUFWriter(str(weights), arch='bert')
        writer.add_uint32('bert.pooling_type', pooling)
        writer.add_uint32('bert.embedding_length', dimension)
        if output_dimension is not None:
            writer.add_uint32('bert.embedding_length_out', output_dimension)
        writer.add_uint32('bert.context_length', 16)
        if decision is not None:
            writer.add_string('bert.decision.type', decision)
        writer.add_token_list([f'token{i}' for i in range(512)])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

    header()
    toml.write_text('[llm]\nmodel="Local/a.gguf"\n[runtime]\nembedding=true\nn_ctx=16\n')
    yield manager, client, name, toml, header


def calls(manager):
    return [r['body'] for root in manager.test.roots for r in records(root, 'requests') if r['path'] == '/v1/embeddings']


def ids(text):
    # Independent oracle for this TCP stub's published tokenizer contract.
    return [1, *(value + 10 for value in text.encode('utf-8')), 2]


@pytest.mark.parametrize('path', PATHS)
async def test_public_shapes_usage_aliases_and_raw_legacy_contract(embeddings, path):
    manager, client, name, _, _ = embeddings
    response = await client.post(path, json=body(name, path, 'žluť 🐈'))
    assert response.status_code == 200, response.text
    payload = response.json()
    call = calls(manager)[0]
    assert call['input'] == [ids('žluť 🐈')] and call['model'] == 'tllama-worker'
    assert call['encoding_format'] == 'float'
    if path == '/api/embeddings':
        assert set(payload) == {'embedding'} and call['embd_normalize'] == -1
        assert math.hypot(*payload['embedding']) > 1
    else:
        vector = payload['embeddings'][0] if path == '/api/embed' else payload['data'][0]['embedding']
        assert len(vector) == 8 and math.hypot(*vector) == pytest.approx(1)
        assert call['embd_normalize'] == 2 and payload['model'] == name
        if path == '/api/embed':
            assert payload['prompt_eval_count'] == len(ids('žluť 🐈'))
            assert 0 <= payload['load_duration'] <= payload['total_duration']
        else:
            assert payload['object'] == 'list' and payload['data'][0]['index'] == 0
            assert payload['usage'] == {'prompt_tokens': len(ids('žluť 🐈')), 'total_tokens': len(ids('žluť 🐈'))}
    assert manager.native_snapshot()[0]['leases'] == 0


async def test_modern_openai_and_normalized_legacy_vectors_agree(embeddings):
    _, client, name, _, _ = embeddings
    modern = (await client.post('/api/embed', json=body(name, '/api/embed'))).json()['embeddings'][0]
    oai = (await client.post('/v1/embeddings', json=body(name, '/v1/embeddings'))).json()['data'][0]['embedding']
    legacy = (await client.post('/api/embeddings', json=body(name, '/api/embeddings'))).json()['embedding']
    assert modern == oai
    assert [v / math.hypot(*legacy) for v in legacy] == pytest.approx(modern)


@pytest.mark.parametrize('path', ['/api/embed', '/v1/embeddings'])
async def test_batch_order_duplicates_unicode_newlines_and_literal_escapes_are_preserved(embeddings, path):
    manager, client, name, _, _ = embeddings
    texts = ['a\nb', r'a\nb', '日本語', 'a\nb']
    response = await client.post(path, json=body(name, path, texts))
    assert response.status_code == 200, response.text
    data = response.json()
    vectors = data.get('embeddings') or [r['embedding'] for r in data['data']]
    assert vectors[0] == vectors[3] and vectors[0] != vectors[1]
    assert calls(manager)[0]['input'] == [ids(text) for text in texts]
    assert (data.get('prompt_eval_count') or data['usage']['prompt_tokens']) == sum(len(ids(t)) for t in texts)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('extra', [{'model': ''}, {'model': 42}, {'dimensions': 0}, {'stream': True},
                                  {'options': {'embedding': True}}, {'keep_alive': True}])
async def test_invalid_requests_fail_before_loading_or_unloading(embeddings, path, extra):
    manager, client, name, _, _ = embeddings
    response = await client.post(path, json=body(name, path, **extra))
    assert response.status_code == 400, response.text
    assert not manager.test.ledger


@pytest.mark.parametrize('path,value', [('/api/embed', 1), ('/api/embed', {'image': 'data'}),
    ('/api/embed', [1]), ('/api/embed', ['ok', {}]), ('/v1/embeddings', ''), ('/v1/embeddings', []),
    ('/v1/embeddings', [True]), ('/v1/embeddings', [1.5]), ('/v1/embeddings', [-1]),
    ('/v1/embeddings', [[]]), ('/v1/embeddings', ['ok', '']), ('/v1/embeddings', ['ok', [1]]),
    ('/v1/embeddings', None), ('/api/embeddings', ['text'])])
async def test_invalid_input_shapes_are_400_and_do_not_admit_models(embeddings, path, value):
    manager, client, name, _, _ = embeddings
    response = await client.post(path, json=body(name, path, value))
    assert response.status_code == 400
    assert not manager.test.ledger


@pytest.mark.parametrize('value', ['', [], None])
async def test_empty_embed_is_load_control_and_missing_model_unload_is_idempotent(embeddings, value):
    manager, client, name, _, _ = embeddings
    response = await client.post('/api/embed', json=body(name, '/api/embed', value, keep_alive=-1))
    assert response.status_code == 200 and response.json()['embeddings'] == []
    assert not calls(manager)
    response = await client.post('/api/embed', json=body(name, '/api/embed', value, keep_alive=0))
    assert response.status_code == 200
    await wait_until(lambda: not manager.native_snapshot())
    response = await client.post('/api/embed', json=body('missing', '/api/embed', value, keep_alive=0))
    assert response.status_code == 200 and len(manager.test.ledger) == 1


async def test_empty_string_in_a_batch_is_an_embedding_item(embeddings):
    manager, client, name, _, _ = embeddings
    response = await client.post('/api/embed', json=body(name, '/api/embed', ['', 'hello']))
    assert response.status_code == 200 and len(response.json()['embeddings']) == 2
    assert calls(manager)[0]['input'][0] == [1, 2]


async def test_ollama_truncation_preserves_added_bos_and_eos_tokens(embeddings):
    manager, client, name, _, _ = embeddings
    response = await client.post('/api/embed', json=body(name, '/api/embed', 'x' * 20))
    assert response.status_code == 200, response.text
    assert calls(manager)[0]['input'] == [[1, *([130] * 14), 2]]
    assert response.json()['prompt_eval_count'] == 16


@pytest.mark.parametrize('path', PATHS)
async def test_explicit_limit_error_never_embeds_and_same_worker_recovers(embeddings, path):
    manager, client, name, _, _ = embeddings
    extra = {'truncate': False} if path == '/api/embed' else {}
    bad = await client.post(path, json=body(name, path, 'x' * 15, **extra))
    assert bad.status_code == 400 and '16' in str(bad.json())
    assert not calls(manager)
    good = await client.post(path, json=body(name, path, 'x' * 14, **extra))
    assert good.status_code == 200 and len(manager.test.ledger) == 1


async def test_physical_ubatch_is_a_conservative_limit_without_architecture_guessing(embeddings):
    manager, client, name, toml, _ = embeddings
    toml.write_text(toml.read_text() + 'n_ubatch=8\n')
    response = await client.post('/api/embed', json=body(name, '/api/embed', 'x' * 12))
    assert response.status_code == 200 and response.json()['prompt_eval_count'] == 8
    assert calls(manager)[0]['input'][0] == [1, *([130] * 6), 2]


@pytest.mark.parametrize('pooling,override,limit', [(1, None, 16), (2, None, 16),
                                                (3, None, 15), (1, 3, 15)])
async def test_native_context_boundary_and_default_truncation(embeddings, monkeypatch, pooling, override, limit):
    manager, client, name, toml, header = embeddings
    header(pooling=pooling)
    effective_pooling = pooling if override is None else override
    monkeypatch.setenv('STUB_EMBEDDING_POOLING', str(effective_pooling))
    if override is not None:
        toml.write_text(toml.read_text() + f'pooling_type={override}\n')
    # A native-owned strict LAST boundary, independent of public preflight.
    async with manager.acquire_native_model(name, keep_alive=-1, operation='text_embedding') as lease:
        from tllama.worker import WorkerRequestError
        native_body = {'input': [[1] * 16], 'encoding_format': 'float'}
        if effective_pooling == 3:
            with pytest.raises(WorkerRequestError) as error:
                await lease.worker.embeddings(native_body)
            assert error.value.http_status == 400
        else:
            assert len((await lease.worker.embeddings(native_body))['data']) == 1
    for path in PATHS:
        extra = {'truncate': False} if path == '/api/embed' else {}
        good = await client.post(path, json=body(name, path, 'x' * (limit - 2), **extra))
        assert good.status_code == 200, good.text
        count = len(calls(manager))
        bad = await client.post(path, json=body(name, path, 'x' * (limit - 1), **extra))
        assert bad.status_code == 400 and len(calls(manager)) == count
    for value in ([1] * limit, [[1] * limit, [1] * limit]):
        assert (await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', value))).status_code == 200
    count = len(calls(manager))
    bad = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', [1] * (limit + 1)))
    assert bad.status_code == 400 and len(calls(manager)) == count
    truncated = await client.post('/api/embed', json=body(name, '/api/embed', 'x' * 20))
    assert truncated.status_code == 200 and truncated.json()['prompt_eval_count'] == limit
    assert calls(manager)[-1]['input'] == [[1, *([130] * (limit - 2)), 2]]
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('bound', ['n_batch', 'n_ubatch', 'trained'])
async def test_last_context_reserve_does_not_reduce_other_bounds(embeddings, monkeypatch, bound):
    manager, client, name, toml, header = embeddings
    header(pooling=3)
    monkeypatch.setenv('STUB_EMBEDDING_POOLING', '3')
    if bound == 'trained':
        # Effective native context is larger than the model's trained bound.
        monkeypatch.setenv('STUB_PUBLIC_CTX', '32')
        toml.write_text(toml.read_text().replace('n_ctx=16', 'n_ctx=32'))
        limit = 16
    else:
        toml.write_text(toml.read_text() + f'{bound}=8\n')
        limit = 8
    response = await client.post('/api/embed', json=body(name, '/api/embed', 'x' * 40))
    assert response.status_code == 200 and response.json()['prompt_eval_count'] == limit
    assert len(calls(manager)[-1]['input'][0]) == limit


@pytest.mark.parametrize('value', [[1, 17, 2], [[1, 17, 2], [1, 18, 19, 2]]])
async def test_openai_token_arrays_are_exact_final_tokens_and_usage(embeddings, value):
    manager, client, name, _, _ = embeddings
    response = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', value))
    assert response.status_code == 200, response.text
    rows = [value] if isinstance(value[0], int) else value
    assert calls(manager)[0]['input'] == rows
    assert response.json()['usage']['prompt_tokens'] == sum(map(len, rows))
    assert not [r for r in records(manager.test.roots[0], 'requests') if r['path'] == '/tokenize']


@pytest.mark.parametrize('value', [[512], [1] * 17])
async def test_invalid_token_ids_or_lengths_are_refused_before_decode(embeddings, value):
    manager, client, name, _, _ = embeddings
    response = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', value))
    assert response.status_code == 400 and not calls(manager)


async def test_full_dimensions_and_little_endian_float32_base64(embeddings):
    manager, client, name, _, _ = embeddings
    floating = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', dimensions=8))
    encoded = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', dimensions=8, encoding_format='base64'))
    assert floating.status_code == encoded.status_code == 200
    vector = floating.json()['data'][0]['embedding']
    decoded = struct.unpack('<8f', base64.b64decode(encoded.json()['data'][0]['embedding'], validate=True))
    assert decoded == pytest.approx(vector, rel=1e-7)
    count = len(calls(manager))
    bad = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings', dimensions=4))
    assert bad.status_code == 400 and len(calls(manager)) == count


@pytest.mark.parametrize('mode,status', [('http400', 400), ('http500', 500), ('nan', 500),
    ('bad-index', 500), ('bad-norm', 500), ('wrong-dimension', 500), ('bad-usage', 500)])
@pytest.mark.parametrize('path', ['/api/embed', '/v1/embeddings'])
async def test_upstream_errors_are_atomic_and_same_pid_recovers(embeddings, mode, status, path):
    manager, client, name, _, _ = embeddings
    await manager.preload_native_model(name)
    behavior = manager.test.roots[0] / 'embedding-behavior'
    behavior.write_text(mode)
    response = await client.post(path, json=body(name, path, ['one', 'two']))
    assert response.status_code == status, response.text
    error = response.json()['error']
    assert isinstance(error, str if path == '/api/embed' else dict)
    behavior.write_text('normal')
    good = await client.post(path, json=body(name, path))
    assert good.status_code == 200 and len(manager.test.ledger) == 1
    assert manager.native_snapshot()[0]['leases'] == 0


async def test_missing_native_usage_uses_exact_submitted_tokens_and_indices_can_be_reordered(embeddings):
    manager, client, name, _, _ = embeddings
    first = await client.post('/api/embed', json=body(name, '/api/embed', ['one', 'two']))
    behavior = manager.test.roots[0] / 'embedding-behavior'
    behavior.write_text('reverse')
    reverse = await client.post('/api/embed', json=body(name, '/api/embed', ['one', 'two']))
    assert first.json()['embeddings'] == reverse.json()['embeddings']
    behavior.write_text('no-usage')
    missing = await client.post('/v1/embeddings', json=body(name, '/v1/embeddings'))
    assert missing.status_code == 200 and missing.json()['usage'] == {'prompt_tokens': 7, 'total_tokens': 7}


async def test_private_batch_splitting_preserves_global_order_and_usage(embeddings, monkeypatch):
    from tllama import native_embeddings
    manager, client, name, _, _ = embeddings
    texts = ['one', 'two', 'one']
    baseline = await client.post('/api/embed', json=body(name, '/api/embed', texts))
    monkeypatch.setattr(native_embeddings, '_NATIVE_EMBEDDING_JSON_BUDGET', 1024)
    split = await client.post('/api/embed', json=body(name, '/api/embed', texts))
    assert split.status_code == 200 and split.json()['embeddings'] == baseline.json()['embeddings']
    assert split.json()['prompt_eval_count'] == 15 and len(calls(manager)) == 4


@pytest.mark.parametrize('pooling,decision', [(0, None), (4, None), (1, 'laya')])
async def test_nonsemantic_outputs_are_not_advertised_or_evaluated(embeddings, pooling, decision):
    manager, client, name, _, header = embeddings
    header(pooling=pooling, decision=decision)
    show = await client.post('/api/show', json={'model': name})
    assert show.status_code == 200 and show.json()['capabilities'] == []
    response = await client.post('/api/embed', json=body(name, '/api/embed'))
    assert response.status_code == 400 and not calls(manager) and not manager.test.ledger


async def test_real_gguf_metadata_and_show_support_embedding_without_loading(embeddings):
    manager, client, name, _, _ = embeddings
    meta = read_gguf_metadata(str(manager.local_models_dir / 'a.gguf'))
    assert meta['bert.embedding_length'] == 8 and meta['bert.pooling_type'] == 1
    assert meta['embedding']['vocab_size'] == 512 and 'tokenizer.ggml.tokens' not in meta
    response = await client.post('/api/show', json={'model': name})
    assert response.json()['capabilities'] == ['embedding'] and not manager.test.ledger


@pytest.mark.parametrize('path', PATHS)
async def test_projected_output_width_overrides_hidden_width_and_refreshes_old_cache(embeddings, path):
    from tllama.helpers.metadata_cache import get_metadata_cache_path, save_metadata_cache
    from tllama.helpers.gguf_metadata import build_model_metadata_payload
    manager, client, name, _, header = embeddings
    header(dimension=512, output_dimension=8)
    weights = manager.local_models_dir / 'a.gguf'
    metadata = read_gguf_metadata(str(weights))
    assert metadata['bert.embedding_length'] == 512 and metadata['bert.embedding_length_out'] == 8
    # Schema 5 could have already cached the same GGUF without its output key.
    old = build_model_metadata_payload({k: v for k, v in metadata.items() if k != 'bert.embedding_length_out'})
    save_metadata_cache(manager.metadata_cache_dir, name, weights, old)
    cache = get_metadata_cache_path(manager.metadata_cache_dir, weights)
    document = json.loads(cache.read_text())
    document['schema_version'] = 5
    cache.write_text(json.dumps(document))
    extra = {} if path == '/api/embeddings' else {'dimensions': 8}
    response = await client.post(path, json=body(name, path, 'hello', **extra))
    assert response.status_code == 200, response.text
    payload = response.json()
    if path == '/api/embed':
        vectors = payload['embeddings']
    elif path == '/api/embeddings':
        vectors = [payload['embedding']]
    else:
        vectors = [r['embedding'] for r in payload['data']]
    assert len(vectors[0]) == 8
    assert json.loads(cache.read_text())['schema_version'] == 6
    show = (await client.post('/api/show', json={'model': name})).json()
    assert show['model_info']['bert.embedding_length'] == 512
    assert show['model_info']['bert.embedding_length_out'] == 8
    assert show['capabilities'] == ['embedding']
    if path != '/api/embeddings':
        before = len(calls(manager))
        bad = await client.post(path, json=body(name, path, dimensions=512))
        assert bad.status_code == 400 and len(calls(manager)) == before
    if path == '/v1/embeddings':
        encoded = await client.post(path, json=body(name, path, encoding_format='base64'))
        decoded = struct.unpack('<8f', base64.b64decode(encoded.json()['data'][0]['embedding'], validate=True))
        assert decoded == pytest.approx(vectors[0], rel=1e-7)


@pytest.mark.parametrize('width,expected', [(None, 512), (0, 512), (768, 768)])
def test_output_dimension_contract_uses_native_zero_default(width, expected):
    from tllama.embedding_contract import embedding_contract
    raw = {'general.architecture': 'gemma-embedding2', 'gemma-embedding2.pooling_type': 1,
           'gemma-embedding2.embedding_length': 512}
    if width is not None:
        raw['gemma-embedding2.embedding_length_out'] = width
    assert embedding_contract({'metadata_raw': raw}).dimension == expected


@pytest.mark.parametrize('width', [None, True, -1, 8.0, '768'])
def test_invalid_explicit_output_width_is_not_hidden_by_fallback(width):
    from tllama.embedding_contract import embedding_contract
    raw = {'general.architecture': 'bert', 'bert.pooling_type': 1, 'bert.embedding_length': 512,
           'bert.embedding_length_out': width}
    with pytest.raises(ValueError, match='embedding_length_out'):
        embedding_contract({'metadata_raw': raw})


async def test_explicit_toml_pooling_override_is_plain_integer_in_offline_capabilities(embeddings):
    manager, client, name, toml, header = embeddings
    header(pooling=0)
    toml.write_text(toml.read_text() + 'pooling_type=1\n')
    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200 and response.json()['capabilities'] == ['embedding']
    assert not manager.test.ledger
    assert (await client.post('/api/embed', json=body(name, '/api/embed'))).status_code == 200


async def test_pre_embedding_metadata_cache_is_invalidated_before_capability_advertisement(embeddings):
    from tllama.helpers.metadata_cache import get_metadata_cache_path, load_metadata_cache, save_metadata_cache
    manager, client, name, _, _ = embeddings
    weights = manager.local_models_dir / 'a.gguf'
    save_metadata_cache(manager.metadata_cache_dir, name, weights, {'arch': 'bert'})
    cache = get_metadata_cache_path(manager.metadata_cache_dir, weights)
    document = json.loads(cache.read_text())
    document['schema_version'] = 5
    cache.write_text(json.dumps(document))
    assert load_metadata_cache(manager.metadata_cache_dir, weights) is None
    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200 and response.json()['capabilities'] == ['embedding']


async def test_cancelling_a_queued_embedding_request_preserves_the_active_lease(embeddings):
    manager, client, name, _, _ = embeddings
    await manager.preload_native_model(name)
    behavior = manager.test.roots[0] / 'embedding-behavior'
    behavior.write_text('hold')
    first = asyncio.create_task(client.post('/api/embed', json=body(name, '/api/embed')))
    second = None
    try:
        await wait_until(lambda: records(manager.test.roots[0], 'started'))
        second = asyncio.create_task(client.post('/api/embed', json=body(name, '/api/embed', 'queued')))
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 2)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert manager.native_snapshot()[0]['leases'] == 1 and len(calls(manager)) == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
        await wait_until(lambda: records(manager.test.roots[0], 'cancelled'))
        behavior.write_text('normal')
        assert (await client.post('/api/embed', json=body(name, '/api/embed'))).status_code == 200
        assert len(manager.test.ledger) == 1
    finally:
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)


async def test_generation_mode_is_not_implicitly_converted_and_missing_models_are_404(embeddings):
    manager, client, name, toml, _ = embeddings
    toml.write_text(toml.read_text().replace('embedding=true', 'embedding=false'))
    assert (await client.post('/api/embed', json=body(name, '/api/embed'))).status_code == 400
    assert not manager.test.ledger
    for path in PATHS:
        assert (await client.post(path, json=body('missing', path))).status_code == 404


@pytest.mark.parametrize('path', PATHS)
async def test_zero_keep_alive_finishes_inference_then_cleans_worker(embeddings, path):
    manager, client, name, _, _ = embeddings
    if path == '/v1/embeddings':
        # OpenAI uses server TTL, never an undeclared request keep_alive field.
        manager.config = replace(manager.config, keep_alive=0)
        extra = {}
    else:
        extra = {'keep_alive': 0}
    response = await client.post(path, json=body(name, path, **extra))
    assert response.status_code == 200
    await wait_until(lambda: not manager.native_snapshot())
    assert not manager.test.ledger[0].runtime_directory.exists()


async def test_real_ollama_and_openai_sdks_parse_all_response_variants(embeddings, monkeypatch):
    # The SDK constructs global clients during import; never inherit a proxy
    # for this explicitly local protocol test.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(name, raising=False)
    ollama = pytest.importorskip('ollama')
    openai = pytest.importorskip('openai')
    manager, _, name, _, _ = embeddings
    client = ollama.AsyncClient(host='http://public', transport=httpx.ASGITransport(app=app), trust_env=False)
    try:
        result = await client.embed(model=name, input=['one', 'two'], truncate=False, dimensions=8)
        assert len(result.embeddings) == 2 and result.prompt_eval_count == 10
        assert len((await client.embeddings(model=name, prompt='one')).embedding) == 8
        with pytest.raises(ollama.ResponseError) as error:
            await client.embed(model=name, input='one', dimensions=4)
        assert error.value.status_code == 400
    finally:
        await client._client.aclose()
    async with openai.AsyncOpenAI(base_url='http://public/v1', api_key='local-test', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app))) as client:
        auto = await client.embeddings.create(model=name, input='one')
        assert len(auto.data[0].embedding) == 8 and auto.usage.prompt_tokens == 5
        encoded = await client.embeddings.create(model=name, input=[[1, 20, 2]], encoding_format='base64')
        assert isinstance(encoded.data[0].embedding, str) and encoded.usage.prompt_tokens == 3
        with pytest.raises(openai.BadRequestError):
            await client.embeddings.create(model=name, input='one', dimensions=4)


@pytest.mark.parametrize('path', PATHS)
async def test_real_tcp_nonstream_disconnect_releases_native_transport_and_lease(embeddings, path):
    manager, client, name, _, _ = embeddings
    await manager.preload_native_model(name)
    behavior = manager.test.roots[0] / 'embedding-behavior'
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
        payload = json.dumps(body(name, path)).encode()
        writer.write(f'POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n'.encode() + payload)
        await writer.drain()
        await wait_until(lambda: records(manager.test.roots[0], 'started'))
        writer.close()
        await writer.wait_closed()
        await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
        await wait_until(lambda: records(manager.test.roots[0], 'cancelled'))
        behavior.write_text('normal')
        good = await client.post(path, json=body(name, path))
        assert good.status_code == 200 and len(manager.test.ledger) == 1
    finally:
        if writer is not None:
            writer.close()
        server.should_exit = True
        await asyncio.wait_for(serving, 3)
        sock.close()
