"""Explicit embedding load identity and owned native transport, without public adapters."""

import asyncio
from dataclasses import replace
from pathlib import Path

import gguf
import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.helpers.model_toml import TomlModelError, parse_model_toml
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.native_config import resolve_worker_config
from tllama.worker import WorkerRequestError
from tllama.worker_config import WorkerLaunchSpec

from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, request_body


LLM = '[llm]\nmodel="Local/a.gguf"\n'
EMBEDDING = '[runtime]\nembedding=true\nn_ctx=256\n'


@pytest.mark.parametrize('pooling,flag', [(-1, None), (1, 'mean'), (2, 'cls'), (3, 'last')])
def test_explicit_mode_and_pooling_reach_the_launch_without_attention_override(pooling, flag):
    virtual = parse_model_toml(LLM + EMBEDDING + f'pooling_type={pooling}\n')
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}),
                                     Path('/models/a.gguf'), virtual)
    args = resolved.launch.command(Path('/private/key'))
    assert resolved.launch.embedding is True and resolved.launch.pooling_type == pooling
    assert args.count('--embeddings') == 1 and args.count('--spec-type') == 1
    assert args[args.index('--spec-type') + 1] == 'none' and '--attention' not in args
    assert ('--pooling' in args) == (flag is not None)
    if flag is not None:
        assert args[args.index('--pooling') + 1] == flag
    assert resolved.sources['embedding'] == resolved.sources['pooling_type'] == 'TOML'


def test_toml_mode_can_override_the_explicit_global_mode_without_mutating_it():
    values = {'device': 'none', 'embedding': 'true', 'pooling_type': '-1'}
    assert coerce_runtime_overrides(values)['embedding'] is True
    config = BackendConfig(runtime_overrides=values)
    one = resolve_worker_config(config, Path('/models/a.gguf'))
    two = resolve_worker_config(config, Path('/models/a.gguf'), parse_model_toml(LLM + '[runtime]\nembedding=false'))
    assert one.launch.embedding is True and two.launch.embedding is False
    assert '--embeddings' not in two.launch.command(Path('/private/key'))
    assert values['embedding'] == 'true' and one != two


@pytest.mark.parametrize('line', ['embedding=1', 'embedding="true"', 'pooling_type=true',
                                'pooling_type=0', 'pooling_type=4', 'pooling_type=99', 'attention_type=1'])
def test_invalid_or_unintegrated_mode_values_are_refused(line):
    with pytest.raises(TomlModelError):
        parse_model_toml(LLM + '[runtime]\n' + line)


@pytest.mark.parametrize('extra', [
    'pooling_type=1',
    'embedding=true\nspec_type="draft-mtp"',
    'embedding=true\ndraft_n_max=0',
    'embedding=true\nspm_infill=true',
    'embedding=true\n[template]\njinja=""',
    'embedding=true\n[sampling]\ntemperature=0.4',
    'embedding=true\n[system]\nprompt=""',
    'embedding=true\n[fim]\norder="psm"',
])
def test_generation_only_settings_cannot_enter_an_embedding_launch(extra):
    with pytest.raises(ConfigError):
        resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}), Path('/models/a.gguf'),
                              parse_model_toml(LLM + '[runtime]\n' + extra))


@pytest.mark.parametrize('values', [{'embedding': 1}, {'embedding': 'true'},
                                  {'embedding': True, 'pooling_type': True},
                                  {'embedding': True, 'pooling_type': 0},
                                  {'embedding': True, 'mmproj': Path('/models/projector.gguf')},
                                  {'embedding': True, 'template': '{{ Prompt }}'},
                                  {'pooling_type': 3}])
def test_direct_launch_cannot_bypass_mode_validation(values):
    with pytest.raises(ValueError):
        WorkerLaunchSpec(model=Path('/models/a.gguf'), device='none', **values)


@pytest.mark.parametrize('embedding', [False, True])
async def test_wrong_operation_never_spawns_or_retires_a_worker(native_manager, embedding):
    manager = native_manager()
    name, _ = manager.test.define(extra=EMBEDDING if embedding else '')
    wrong = 'generation' if embedding else 'embedding'
    for loaded in (False, True):
        if loaded:
            generation = await manager.preload_native_model(name, keep_alive=-1)
        before = manager.native_snapshot()
        with pytest.raises(ValueError, match='not configured'):
            async with manager.acquire_native_model(name, operation=wrong, keep_alive=0):
                pytest.fail('Incompatible lease admitted')
        assert manager.native_snapshot() == before
        assert len(manager.test.ledger) == int(loaded)
    assert manager.native_snapshot()[0]['generation'] == generation


async def test_mode_change_replaces_identity_and_preserves_an_admitted_generation(native_manager):
    manager = native_manager(limit=2)
    name, toml = manager.test.define()
    async with manager.acquire_native_model(name, operation='generation') as old:
        toml.write_text(LLM + EMBEDDING)
        async with manager.acquire_native_model(name, operation='embedding', keep_alive=0) as new:
            assert old.generation != new.generation and old.worker.pid != new.worker.pid
            assert old.resolved.launch.embedding is False and new.resolved.launch.embedding is True
            assert (await old.worker.chat({'messages': []}))['choices']
            assert (await new.worker.embeddings({'input': 'new mode'}))['data']
            assert len(manager.native_snapshot()) == 2
    await wait_until(lambda: not manager.native_snapshot())
    assert all(not worker.runtime_directory.exists() for worker in manager.test.ledger)


async def test_embedding_transport_preserves_inputs_and_owns_alias_and_cleanup(native_manager):
    manager = native_manager()
    name, _ = manager.test.define(extra=EMBEDDING)
    body = {'input': ['První řádek\n日本語 🐈', 'second', 'second'], 'model': 'foreign', 'encoding_format': 'float'}
    async with manager.acquire_native_model(name, operation='embedding', keep_alive=0) as lease:
        payload = await lease.worker.embeddings(body)
        assert [row['index'] for row in payload['data']] == [0, 1, 2]
        assert payload['data'][1]['embedding'] == payload['data'][2]['embedding']
        assert await lease.worker.tokenize('hello')
        assert len(manager.native_snapshot()) == 1
    await manager.wait_native_closed(lease.generation)
    call = next(row['body'] for row in records(manager.test.roots[0], 'requests') if row['path'] == '/v1/embeddings')
    assert call == {**body, 'model': lease.resolved.launch.alias} and body['model'] == 'foreign'
    assert lease.worker.process.returncode is not None and not lease.worker.runtime_directory.exists()


@pytest.mark.parametrize('method', ['chat', 'completion', 'chat_stream', 'completion_stream'])
async def test_embedding_worker_rejects_every_generation_transport_before_post(native_manager, method):
    manager = native_manager()
    name, _ = manager.test.define(extra=EMBEDDING)
    async with manager.acquire_native_model(name, operation='embedding') as lease:
        with pytest.raises(WorkerRequestError) as error:
            call = getattr(lease.worker, method)({})
            if method.endswith('_stream'):
                async with call:
                    pytest.fail('Generation stream opened')
            else:
                await call
        assert error.value.status_code == 400
    assert not [row for row in records(manager.test.roots[0], 'requests') if row['path'].startswith(('/completion', '/v1/chat'))]


async def test_generation_worker_rejects_embedding_transport_before_post(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name) as lease:
        with pytest.raises(WorkerRequestError) as error:
            await lease.worker.embeddings({'input': 'text'})
        assert error.value.status_code == 400
    assert not [row for row in records(manager.test.roots[0], 'requests') if row['path'] == '/v1/embeddings']


@pytest.mark.parametrize('mode,status', [('normal', 400), ('error500', 500)])
async def test_embedding_http_error_releases_slot_and_same_worker_recovers(native_manager, monkeypatch, mode, status):
    monkeypatch.setenv('STUB_EMBEDDINGS', mode)
    manager = native_manager()
    name, _ = manager.test.define(extra=EMBEDDING)
    async with manager.acquire_native_model(name, operation='embedding') as lease:
        with pytest.raises(WorkerRequestError) as error:
            await lease.worker.embeddings({'input': 'text', 'encoding_format': 'invalid'})
        assert error.value.status_code == status
        assert (await lease.worker.embeddings({'input': 'recovery'}))['data']
    assert len(manager.test.ledger) == 1


async def test_embedding_transport_cancellation_releases_the_shared_slot(native_manager):
    manager = native_manager()
    name, _ = manager.test.define(extra=EMBEDDING)
    async with manager.acquire_native_model(name, operation='embedding') as lease:
        pending = asyncio.create_task(lease.worker.embeddings({'input': ['hold'] * 32}))
        queued = None
        try:
            await wait_until(lambda: records(manager.test.roots[0], 'started'))
            queued = asyncio.create_task(lease.worker.tokenize('after cancel'))
            await asyncio.sleep(0)
            assert not queued.done()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert await asyncio.wait_for(queued, 2)
            await wait_until(lambda: records(manager.test.roots[0], 'cancelled'))
            assert (await lease.worker.embeddings({'input': 'recovery'}))['data']
        finally:
            for task in (pending, queued):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (pending, queued) if task is not None), return_exceptions=True)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_public_generation_rejects_embedding_mode_before_model_admission(public, path, stream):
    manager, client, name, toml = public
    toml.write_text(LLM + EMBEDDING)
    response = await client.post(path, json=request_body(name, path, stream))
    assert response.status_code == 400 and 'not configured for generation' in str(response.json())
    assert not manager.test.ledger and not manager.native_snapshot()


async def test_public_empty_generate_can_preload_and_unload_embedding_mode(public):
    manager, client, name, toml = public
    toml.write_text(LLM + EMBEDDING)
    for keep_alive, reason in [(-1, 'load'), (0, 'unload')]:
        response = await client.post('/api/generate', json={'model': name, 'keep_alive': keep_alive})
        assert response.status_code == 200 and response.json()['done_reason'] == reason
    await wait_until(lambda: not manager.native_snapshot())
    assert len(manager.test.ledger) == 1
    assert not [row for row in records(manager.test.roots[0], 'requests')
                if row['path'] in {'/v1/embeddings', '/completion', '/v1/chat/completions'}]


@pytest.mark.parametrize('global_mode,local_mode,expected', [('true', None, True), ('true', False, False),
                                                         ('false', True, True)])
async def test_show_requires_known_pooled_output_metadata(
        public, monkeypatch, global_mode, local_mode, expected):
    manager, client, name, toml = public
    writer = gguf.GGUFWriter(str(manager.local_models_dir / 'a.gguf'), arch='bert')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    toml.write_text(LLM + ('' if local_mode is None else f'[runtime]\nembedding={str(local_mode).lower()}'))
    monkeypatch.setattr(manager, 'config', replace(manager.config,
                        runtime_overrides={'device': 'none', 'embedding': global_mode}))
    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200, response.text
    assert response.json()['capabilities'] == ([] if expected else ['completion'])
    assert not manager.test.ledger
