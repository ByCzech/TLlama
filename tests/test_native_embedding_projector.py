"""Production embedding/projector ownership and unchanged public text contracts."""

import asyncio
from pathlib import Path

import gguf
import pytest

from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import resolve_worker_config
from tllama.worker import WorkerRequestError
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public
from test_embedding_api import embeddings, body, calls, ids, PATHS
from vision_fixtures import RED_BLUE_URL, BLUE_RED_URL


PROJECTOR = '[mmproj]\nmodel="Local/vision.gguf"\n'


def projector(path):
    writer = gguf.GGUFWriter(path, 'clip')
    writer.add_bool('clip.has_vision_encoder', True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


@pytest.mark.parametrize('device', ['none', 'Vulkan0'])
def test_explicit_embedding_projector_uses_normal_launch_options(device):
    source = parse_model_toml('[llm]\nmodel="Local/a.gguf"\n[runtime]\n'
                              'embedding=true\npooling_type=3\nspec_type="none"\n'
                              + PROJECTOR + 'warmup=false\nimage_max_tokens=256\n')
    config = BackendConfig(runtime_overrides={'device': device})
    resolved = resolve_worker_config(config, Path('/models/Local/a.gguf'), source,
                                     mmproj=Path('/models/Local/vision.gguf'))
    argv = resolved.launch.command(Path('/private/key'))
    assert argv.count('--embeddings') == argv.count('--mmproj') == argv.count('--pooling') == 1
    assert argv[argv.index('--pooling') + 1] == 'last'
    assert argv[argv.index('--mmproj') + 1] == '/models/Local/vision.gguf'
    assert '--no-warmup' in argv and argv[argv.index('--image-max-tokens') + 1] == '256'
    assert ('--no-mmproj-offload' in argv) is (device == 'none')


@pytest.mark.parametrize('path', PATHS)
async def test_projector_worker_keeps_public_text_and_generation_contracts(embeddings, path):
    manager, client, name, toml, _ = embeddings
    projector(manager.local_models_dir / 'vision.gguf')
    toml.write_text(toml.read_text() + PROJECTOR)
    show = await client.post('/api/show', json={'model': name})
    assert show.json()['capabilities'] == ['embedding'] and not manager.test.ledger
    result = await client.post(path, json=body(name, path, 'hello'))
    assert result.status_code == 200, result.text
    assert calls(manager)[0]['input'] == [ids('hello')]
    worker = manager.test.ledger[0]
    assert worker.spec.embedding and worker.spec.mmproj == manager.local_models_dir / 'vision.gguf'
    before = {k: v for k, v in manager.native_snapshot()[0].items() if k != 'ttl_remaining'}
    generation = await client.post('/api/generate', json={'model': name, 'prompt': 'hello', 'stream': False})
    assert generation.status_code == 400 and 'not configured for generation' in generation.text
    assert {k: v for k, v in manager.native_snapshot()[0].items() if k != 'ttl_remaining'} == before
    assert len(manager.test.ledger) == 1
    media = await client.post(path, json=body(name, path, {'content': [
        {'type': 'image_url', 'image_url': {'url': RED_BLUE_URL}}]}))
    assert media.status_code == 400 and len(calls(manager)) == 1
    unload = await client.post('/api/embed', json={'model': name, 'input': '', 'keep_alive': 0})
    assert unload.status_code == 200
    await wait_until(lambda: not manager.native_snapshot())
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


@pytest.mark.parametrize('pooling,decision', [(0, None), (4, None), (1, 'laya')])
async def test_projector_does_not_make_nonsemantic_output_embedding_capable(embeddings, pooling, decision):
    manager, client, name, toml, header = embeddings
    projector(manager.local_models_dir / 'vision.gguf')
    toml.write_text(toml.read_text() + PROJECTOR)
    header(pooling=pooling, decision=decision)
    show = await client.post('/api/show', json={'model': name})
    assert show.status_code == 200 and show.json()['capabilities'] == []
    request = await client.post('/api/embed', json=body(name, '/api/embed'))
    assert request.status_code == 400 and not manager.test.ledger


@pytest.mark.parametrize('change', ['path', 'weights', 'options'])
async def test_projector_replacement_preserves_admitted_embedding_generation(native_manager, change):
    manager = native_manager(limit=2)
    name, toml = manager.test.define(extra='[runtime]\nembedding=true\n' + PROJECTOR)
    path = manager.local_models_dir / 'vision.gguf'
    projector(path)
    async with manager.acquire_native_model(name, operation='embedding', keep_alive=-1) as old:
        if change == 'path':
            projector(manager.local_models_dir / 'other.gguf')
            toml.write_text(toml.read_text().replace('Local/vision.gguf', 'Local/other.gguf'))
        elif change == 'weights':
            path.write_bytes(path.read_bytes() + b'changed file fingerprint')
        else:
            toml.write_text(toml.read_text() + 'warmup=false\n')
        async with manager.acquire_native_model(name, operation='embedding', keep_alive=-1) as new:
            assert old.generation != new.generation and old.worker.pid != new.worker.pid
            assert old.resolved.launch.mmproj_options == {}
            if change == 'options':
                assert new.resolved.launch.mmproj_options['warmup'] is False
            assert (await old.worker.embeddings({'input': 'old lease'}))['data']
            assert (await new.worker.embeddings({'input': 'new lease'}))['data']
            await manager.unload_native_model(name)
            assert old.worker.process.returncode is new.worker.process.returncode is None
    await wait_until(lambda: not manager.native_snapshot())
    assert all(not w.runtime_directory.exists() for w in manager.test.ledger)


async def test_native_image_transport_cancels_and_recovers_on_same_projector_worker(native_manager, monkeypatch):
    monkeypatch.setenv('STUB_MULTIMODAL_EMBEDDINGS', 'normal')
    manager = native_manager()
    name, _ = manager.test.define(extra='[runtime]\nembedding=true\n' + PROJECTOR)
    projector(manager.local_models_dir / 'vision.gguf')
    entry = {'content': [{'type': 'image_url', 'image_url': {'url': RED_BLUE_URL}}]}
    async with manager.acquire_native_model(name, operation='embedding', keep_alive=0) as lease:
        first = await lease.worker.embeddings({'input': entry, 'model': 'foreign'})
        other = await lease.worker.embeddings({'input': {'content': [
            {'type': 'image_url', 'image_url': {'url': BLUE_RED_URL}}]}})
        assert first['data'] != other['data']
        with pytest.raises(WorkerRequestError):
            await lease.worker.embeddings({'input': [], 'encoding_format': 'invalid'})
        task = asyncio.create_task(lease.worker.embeddings({'input': [entry] * 8}))
        try:
            await wait_until(lambda: records(manager.test.roots[0], 'started'))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            recovered = await lease.worker.embeddings({'input': entry})
            assert recovered['data'] == first['data'] and len(manager.test.ledger) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await manager.wait_native_closed(lease.generation)
    requests = [r['body'] for r in records(manager.test.roots[0], 'requests') if r['path'] == '/v1/embeddings']
    assert requests[0] == {'input': entry, 'model': 'tllama-worker'}
    assert not manager.native_snapshot() and not lease.worker.runtime_directory.exists()
