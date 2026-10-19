"""Managed Decision mode with real GGUF headers and authenticated TCP workers."""

import asyncio
import json
from pathlib import Path

import gguf
import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.decision_contract import (DECISION_RUNTIME, MAX_REQUEST_BYTES, decision_type,
                                     prepare_decision_request, validate_decision_response)
from tllama.helpers.gguf_metadata import read_gguf_metadata, build_model_metadata_payload
from tllama.helpers.metadata_cache import (load_metadata_cache, save_metadata_cache,
                                          get_metadata_cache_path, SCHEMA_VERSION)
from tllama.native_config import resolve_worker_config
from tllama.worker import WorkerRequestError, WorkerProtocolError, WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec

from test_decision_smoke import header
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, request_body


LLM = '[llm]\nmodel="Local/a.gguf"\n'
BODY = {'state': {'ticket': 'I was charged twice.\nLiteral \\n 日本語 🐈', 'data': [False, 0, None]},
        'questions': {
            'řádek\nliteral\\n 🐈': {'type': 'choice', 'instructions': {'choose': 'a department'},
                                     'criteria': {'z': None, 'a\nb': False, 'literal\\n': ['billing']}},
            'score': {'type': 'score', 'instructions': 0, 'criteria': [False, {'level': 1}, 'high']},
            'truth': {'type': 'noul', 'instructions': False, 'criteria': {'true': ['angry'], 'false': None}}}}


@pytest.fixture
def decision_model(native_manager, monkeypatch):
    def make(kind='laya', *, mode='normal', extra='', limit=1):
        monkeypatch.setenv('STUB_DECISION', mode)
        monkeypatch.setenv('STUB_DECISION_KIND', kind)
        monkeypatch.setenv('STUB_BUILD', DECISION_RUNTIME)
        manager = native_manager(limit=limit)
        name, toml = manager.test.define(extra=extra)
        weights = manager.local_models_dir / 'a.gguf'
        header(weights, kind=kind)
        if '[mmproj]' in extra:
            writer = gguf.GGUFWriter(manager.local_models_dir / 'p.gguf', 'clip')
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.close()
        return manager, name, toml, weights
    return make


@pytest.mark.parametrize('kind', ['laya', 'nimble', 'clef'])
async def test_metadata_selects_mode_and_owned_transport_preserves_literal_json(decision_model, kind):
    manager, name, toml, weights = decision_model(kind)
    before = toml.read_bytes()
    async with manager.acquire_native_model(name, operation='decision', keep_alive=0) as lease:
        spec = lease.resolved.launch
        assert spec.mode == 'decision' and spec.decision_type == kind
        assert not spec.embedding and spec.pooling_type == -1 and spec.template is None
        argv = lease.worker.command
        assert '--embeddings' not in argv and '--pooling' not in argv and '--chat-template-file' not in argv
        assert argv[argv.index('--spec-type') + 1] == 'none'
        payload = await lease.worker.decision({**BODY, 'model': 'foreign'})
        assert payload['model'] == spec.alias
        assert list(payload['answers']) == list(BODY['questions'])
        keys = list(payload['answers'][next(iter(BODY['questions']))]['probabilities'])
        expected = list(next(iter(BODY['questions'].values()))['criteria'])
        assert keys == (sorted(expected) if kind == 'clef' else expected)
        assert await lease.worker.tokenize('shared slot')
    await manager.wait_native_closed(lease.generation)
    calls = records(manager.test.roots[0], 'requests')
    sent = next(x['body'] for x in calls if x['path'] == '/v1/systemone')
    assert sent == {**BODY, 'model': spec.alias} and BODY['state']['data'] == [False, 0, None]
    assert not lease.worker.runtime_directory.exists() and toml.read_bytes() == before
    cached = load_metadata_cache(manager.metadata_cache_dir, weights)
    assert cached['metadata_raw']['tokenizer.chat_template.systemone']
    assert decision_type(cached) == kind


@pytest.mark.parametrize('extra', [
    '[runtime]\nembedding=true', '[runtime]\nspec_type="draft-mtp"',
    '[runtime]\ndraft_n_max=0', '[runtime]\nspm_infill=true',
    '[runtime.kv_overrides]\n"modern-bert.attention.causal"=true',
    '[runtime]\nlora_path="Local/lora.gguf"', '[template]\njinja=""',
    '[sampling]\ntemperature=0.4', '[system]\nprompt=""', '[fim]\norder="psm"',
    '[mmproj]\nmodel="Local/p.gguf"',
])
async def test_incompatible_settings_fail_before_worker_construction(decision_model, extra):
    manager, name, _, _ = decision_model(extra=extra)
    with pytest.raises(ConfigError):
        async with manager.acquire_native_model(name, operation='decision'):
            pytest.fail('Incompatible Decision config admitted')
    assert not manager.test.ledger and not manager.native_snapshot()


@pytest.mark.parametrize('kind,template', [('laya', False), ('unknown', True), ('pplx-decider', True)])
async def test_unintegrated_or_incomplete_head_never_launches(decision_model, kind, template):
    manager, name, _, weights = decision_model()
    header(weights, kind=kind, template=template)
    with pytest.raises(ConfigError):
        await manager.preload_native_model(name)
    assert not manager.test.ledger


@pytest.mark.parametrize('operation', ['generation', 'embedding', 'text_embedding'])
async def test_wrong_operation_does_not_load_or_retire_decision(decision_model, operation):
    manager, name, _, _ = decision_model()
    for loaded in (False, True):
        if loaded:
            await manager.preload_native_model(name, keep_alive=-1)
        before = manager.native_snapshot()
        with pytest.raises(ValueError, match='not configured'):
            async with manager.acquire_native_model(name, operation=operation, keep_alive=0):
                pytest.fail('Incompatible operation admitted')
        assert manager.native_snapshot() == before and len(manager.test.ledger) == int(loaded)


@pytest.mark.parametrize('method', ['chat', 'completion', 'chat_stream', 'completion_stream', 'embeddings'])
async def test_wrong_transports_fail_before_post(decision_model, method):
    manager, name, _, _ = decision_model()
    async with manager.acquire_native_model(name, operation='decision') as lease:
        with pytest.raises(WorkerRequestError):
            call = getattr(lease.worker, method)({})
            if method.endswith('_stream'):
                async with call:
                    pytest.fail('Wrong stream opened')
            else:
                await call
    assert not any(x['path'] in {'/v1/chat/completions', '/completion', '/v1/embeddings'}
                   for x in records(manager.test.roots[0], 'requests'))


@pytest.mark.parametrize('embedding', [False, True])
async def test_decision_cannot_enter_another_mode(native_manager, embedding):
    manager = native_manager()
    name, _ = manager.test.define(extra='[runtime]\nembedding=true' if embedding else '')
    with pytest.raises(ValueError, match='not configured'):
        async with manager.acquire_native_model(name, operation='decision'):
            pytest.fail('Missing Decision metadata admitted')
    assert not manager.test.ledger
    async with manager.acquire_native_model(name) as lease:
        with pytest.raises(WorkerRequestError):
            await lease.worker.decision(BODY)


@pytest.mark.parametrize('mode', ['no-decisions', 'wrong-model', 'bad-sum', 'bad-confidence', 'wrong-choice',
                                  'wrong-score', 'wrong-legend', 'wrong-order', 'nan', 'no-usage', 'bad-usage'])
async def test_live_and_response_contracts_reject_corruption(decision_model, mode):
    manager, name, _, _ = decision_model(mode=mode)
    if mode == 'no-decisions':
        with pytest.raises(WorkerStartupError, match='Decision alias/output'):
            await manager.preload_native_model(name)
        assert manager.native_snapshot() == []
    else:
        async with manager.acquire_native_model(name, operation='decision') as lease:
            with pytest.raises(WorkerProtocolError):
                await lease.worker.decision(BODY)
            assert await lease.worker.tokenize('slot released')
            (manager.test.roots[0] / 'decision-mode').write_text('normal')
            assert (await lease.worker.decision(BODY))['answers']
            assert len(manager.test.ledger) == 1


async def test_decision_requires_audited_runtime_even_with_version_warn(decision_model, monkeypatch):
    manager, name, _, _ = decision_model()
    monkeypatch.setenv('STUB_BUILD', 'b11552-ec2797836f')
    with pytest.raises(WorkerStartupError, match='audited native runtime'):
        await manager.preload_native_model(name)
    assert manager.native_snapshot() == [] and manager.test.ledger[0].process.returncode is not None


async def test_native_http_error_keeps_status_and_releases_slot(decision_model):
    manager, name, _, _ = decision_model(mode='http500')
    async with manager.acquire_native_model(name, operation='decision') as lease:
        with pytest.raises(WorkerRequestError) as error:
            await lease.worker.decision(BODY)
        assert error.value.http_status == 500
        assert await lease.worker.tokenize('recovery')
        (manager.test.roots[0] / 'decision-mode').write_text('normal')
        assert (await lease.worker.decision(BODY))['answers']


async def test_cancellation_protects_lease_during_unload_and_releases_slot(decision_model):
    manager, name, _, _ = decision_model(mode='hold')
    async with manager.acquire_native_model(name, operation='decision') as lease:
        pending = asyncio.create_task(lease.worker.decision(BODY))
        try:
            await wait_until(lambda: records(manager.test.roots[0], 'started'))
            await manager.unload_native_model(name)
            assert lease.worker.process.returncode is None
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert await lease.worker.tokenize('recovery')
            await wait_until(lambda: records(manager.test.roots[0], 'cancelled'))
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    await manager.wait_native_closed(lease.generation)
    assert not lease.worker.runtime_directory.exists()


async def test_header_mode_change_replaces_identity_without_killing_admitted_generation(decision_model):
    manager, name, _, weights = decision_model(limit=2)
    weights.write_bytes(b'fake generation weights')
    async with manager.acquire_native_model(name, operation='generation') as old:
        header(weights)
        async with manager.acquire_native_model(name, operation='decision', keep_alive=0) as new:
            assert new.generation != old.generation and new.worker.pid != old.worker.pid
            assert old.resolved.launch.mode == 'generation' and new.resolved.launch.mode == 'decision'
            assert (await old.worker.chat({}))['choices']
            assert (await new.worker.decision(BODY))['answers']
    await wait_until(lambda: not manager.native_snapshot())


async def test_pre_decision_cache_is_rebuilt_before_mode_admission(decision_model):
    manager, name, _, weights = decision_model()
    metadata = build_model_metadata_payload(read_gguf_metadata(str(weights)))
    metadata['metadata_raw'].pop('tokenizer.chat_template.systemone')
    save_metadata_cache(manager.metadata_cache_dir, name, weights, metadata)
    cache = get_metadata_cache_path(manager.metadata_cache_dir, weights)
    document = json.loads(cache.read_text())
    document['schema_version'] = 6
    cache.write_text(json.dumps(document))
    async with manager.acquire_native_model(name, operation='decision') as lease:
        assert lease.resolved.launch.decision_type == 'laya'
    assert json.loads(cache.read_text())['schema_version'] == SCHEMA_VERSION
    assert load_metadata_cache(manager.metadata_cache_dir, weights)['metadata_raw']['tokenizer.chat_template.systemone']


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_existing_public_generation_rejects_decision_before_load(public, path, stream):
    manager, client, name, toml = public
    toml.write_text(LLM)
    header(manager.local_models_dir / 'a.gguf')
    response = await client.post(path, json=request_body(name, path, stream))
    assert response.status_code == 400 and 'not configured for generation' in str(response.json())
    assert not manager.test.ledger
    shown = await client.post('/api/show', json={'model': name})
    assert shown.status_code == 200 and shown.json()['capabilities'] == ['decision']


@pytest.mark.parametrize('body', [
    {}, {'state': None, 'questions': BODY['questions']}, {'state': '', 'questions': {}},
    {**BODY, 'images': []}, {**BODY, 'stream': False}, {**BODY, 'options': {}},
    {'state': 'x', 'questions': {'q': {'type': 'noul'}}},
    {'state': 'x', 'questions': {'q': {'type': 'choice', 'instructions': '', 'criteria': {}}}},
    {'state': 'x', 'questions': {'q': {'type': 'score', 'instructions': '', 'criteria': ['one']}}},
    {'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': '', 'criteria': []}}},
    {'state': {1: 'bad key'}, 'questions': BODY['questions']},
    {'state': float('nan'), 'questions': BODY['questions']},
    {'state': 'x', 'questions': {str(i): {'type': 'noul', 'instructions': ''} for i in range(33)}},
])
async def test_bad_input_fails_before_native_post(decision_model, body):
    manager, name, _, _ = decision_model()
    async with manager.acquire_native_model(name, operation='decision') as lease:
        with pytest.raises(ValueError):
            await lease.worker.decision(body)
    assert not any(x['path'] == '/v1/systemone' for x in records(manager.test.roots[0], 'requests'))


def test_request_is_an_owned_snapshot_and_json_resource_limits_are_enforced():
    original = json.loads(json.dumps(BODY))
    snapshot = prepare_decision_request(original)
    original['state']['data'][0] = 'changed'
    assert snapshot == BODY
    for state in ('x' * MAX_REQUEST_BYTES, '\ud800', [None] * 100001):
        with pytest.raises(ValueError):
            prepare_decision_request({**BODY, 'state': state})
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match='depth/node'):
        prepare_decision_request({**BODY, 'state': cyclic})


def test_json_boolean_legend_cannot_be_replaced_by_an_integer():
    question = {'q': {'type': 'score', 'instructions': '', 'criteria': [False, True]}}
    answer = {'model': 'a', 'usage': {'input_tokens': 1, 'output_tokens': 0}, 'answers': {
        'q': {'type': 'score', 'legend': {'0': 0, '1': 1}, 'probabilities': {'0': 0.5, '1': 0.5},
              'score': 0.5, 'confidence': 0}}}
    with pytest.raises(ValueError, match='legend'):
        validate_decision_response(answer, question, 'laya', 'a')


@pytest.mark.parametrize('changes', [
    {'modern-bert.decision.max_head_tokens': 0}, {'modern-bert.decision.block_count': True},
    {'tokenizer.ggml.mask_token_id': 999}, {'modern-bert.decision.temperature.choice': float('inf')},
    {'modern-bert.decision.temperature.choice': -1},
])
def test_bad_head_metadata_is_not_a_valid_decision_config(tmp_path, changes):
    path = tmp_path / 'a.gguf'
    header(path)
    metadata = build_model_metadata_payload(read_gguf_metadata(str(path)))
    metadata['metadata_raw'].update(changes)
    with pytest.raises(ConfigError):
        resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}), path, metadata=metadata)


@pytest.mark.parametrize('values', [{'decision_type': True}, {'decision_type': 'pplx-decider'},
                                  {'decision_type': 'laya', 'embedding': True},
                                  {'decision_type': 'laya', 'template': ''},
                                  {'decision_type': 'clef', 'runtime_options': {'spm_infill': True}}])
def test_direct_worker_spec_cannot_bypass_mode_validation(values):
    with pytest.raises(ValueError):
        WorkerLaunchSpec(model=Path('/models/a.gguf'), device='none', **values)
