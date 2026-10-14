"""Metadata reads and native inference do not queue behind unrelated locks."""

import asyncio

from test_native_manager import native_manager


async def test_held_residency_lock_does_not_block_metadata(manager, gguf_file, toml_file):
    gguf_file('Local/model.gguf')
    toml_file('Local/model.toml', '[llm]\nmodel="Local/model.gguf"\n')
    fingerprint = manager._build_model_file_info('model')['sha256']
    manager._set_cached_metadata_entry('model', fingerprint, {'arch':'qwen3'})
    async with manager._native._condition:
        metadata = await asyncio.wait_for(manager.get_model_metadata('model'), 1)
    assert metadata['arch'] == 'qwen3'


async def test_held_metadata_lock_does_not_block_native_load_or_inference(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager._metadata_lock:
        async with asyncio.timeout(3):
            async with manager.acquire_native_model(name) as lease:
                assert (await lease.worker.chat({}))['choices']


async def test_pending_native_load_does_not_block_metadata_or_listing(native_manager):
    manager = native_manager(mode='hang-load')
    name, _ = manager.test.define()
    fingerprint = manager._build_model_file_info(name)['sha256']
    manager._set_cached_metadata_entry(name, fingerprint, {'arch':'qwen3'})
    loading = asyncio.create_task(manager.preload_native_model(name))
    try:
        async with asyncio.timeout(3):
            while not manager.test.roots or not (manager.test.roots[0] / 'launch').exists():
                await asyncio.sleep(0.01)
            assert (await manager.get_model_metadata(name))['arch'] == 'qwen3'
            assert [entry['id'] for entry in await manager.list_local_models()] == [name]
            assert not loading.done()
    finally:
        await manager.shutdown()
        await asyncio.gather(loading, return_exceptions=True)
