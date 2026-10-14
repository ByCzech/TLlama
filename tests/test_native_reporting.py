"""Public /api/ps observes ownership without definition I/O or leases."""

import asyncio
from datetime import datetime, timezone
import hashlib
import signal

import pytest

from tllama.helpers.metadata_cache import save_digest_cache
from test_native_public_api import public, request_body, stream_mode
from test_native_manager import native_manager, records, wait_until


MIB = 1024**2


async def ps(client):
    response = await client.get('/api/ps')
    assert response.status_code == 200
    return response.json()['models']


async def test_ps_reports_native_allocations_and_effective_context_without_extra_work(public):
    manager, client, name, _ = public
    generation = await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    await wait_until(lambda: worker.memory.snapshot()['status'] == 'known')
    before = records(manager.test.roots[0], 'requests')
    for _ in range(3):
        [model] = await ps(client)
        assert model['name'] == name and model['generation'] == generation
        assert model['pid'] == worker.pid and model['state'] == 'ready' and model['leases'] == 0
        assert model['size'] == int(16.5 * MIB) and model['size_vram'] == 0
        assert model['context_length'] == 2048 and model['runner'] == 'llamacpp'
        assert model['memory']['physical_residency_bytes'] is None
        assert datetime.fromisoformat(model['expires_at']).year > datetime.now(timezone.utc).year + 20
    assert records(manager.test.roots[0], 'requests') == before
    assert len(manager.test.ledger) == 1


async def test_missing_memory_is_null_while_inference_remains_usable(public):
    manager, client, name, _ = public
    manager.test.options['mode'] = 'memory-missing'
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    [model] = await ps(client)
    assert model['size'] is None and model['size_vram'] is None
    assert model['memory']['reason'] == 'missing_buffer_reports'
    assert model['memory']['buffers']['CPU']['model'] == 10 * MIB


async def test_loading_generation_has_unknown_context_and_ps_does_not_wait_for_load(public):
    manager, client, name, _ = public
    manager.test.options['mode'] = 'loading'
    task = asyncio.create_task(manager.preload_native_model(name))
    try:
        await wait_until(lambda: manager.test.ledger and manager.test.ledger[0].process is not None)
        [model] = await ps(client)
        assert model['state'] == 'loading' and model['context_length'] is None
        await task
        assert (await ps(client))[0]['state'] == 'ready'
    finally:
        await asyncio.gather(task, return_exceptions=True)


async def test_toml_and_weight_changes_cannot_rewrite_live_reporting_snapshot(public, monkeypatch):
    manager, client, name, toml = public
    monkeypatch.setattr(manager, '_get_model_metadata_sync', lambda _: {'arch':'qwen3', 'params':35_000_000_000, 'bits':'IQ3_S'})
    weights = manager.local_models_dir / 'a.gguf'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    save_digest_cache(manager.metadata_cache_dir, name, weights, {'content_sha256':digest, 'source':'local'})
    await manager.preload_native_model(name, keep_alive=-1)
    [before] = await ps(client)
    assert before['digest'] == digest and before['details']['parameter_size'] == '35b'
    toml.write_text('invalid [toml')
    weights.write_bytes(b'new different weights')
    def forbidden(*args, **kwargs):
        pytest.fail('Reporting reread mutable model definitions or hashed changed weights')
    for method in ('get_model_metadata', 'get_model_metadata_best_effort', 'get_model_digest',
                   '_build_model_file_info', '_resolve_virtual_model_spec', 'list_loaded_models'):
        monkeypatch.setattr(manager, method, forbidden)
    [after] = await ps(client)
    assert after['details'] == before['details'] and after['digest'] == digest
    assert after['size'] == before['size'] and after['generation'] == before['generation']
    weights.unlink()
    toml.unlink()
    assert (await ps(client))[0]['digest'] == digest


async def test_ps_reads_cannot_extend_idle_ttl(public):
    manager, client, name, _ = public
    now = [1000.0]
    manager._native._clock = lambda: now[0]
    generation = await manager.preload_native_model(name, keep_alive=5)
    for _ in range(3):
        now[0] += 1
        [model] = await ps(client)
        assert model['leases'] == 0
        remaining = (datetime.fromisoformat(model['expires_at']) - datetime.now(timezone.utc)).total_seconds()
        assert 5 - (now[0] - 1000) - 0.2 < remaining <= 5 - (now[0] - 1000) + 0.2
    now[0] = 1006
    await manager._native.expire()
    await manager.wait_native_closed(generation)
    assert await ps(client) == []


async def test_draining_stream_stays_listed_until_its_lease_is_released(public):
    manager, client, name, _ = public
    stream_mode(manager, 'controlled')
    task = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
        [active] = await ps(client)
        assert active['leases'] == 1 and active['state'] == 'ready'
        await manager.unload_native_model(name)
        [draining] = await ps(client)
        assert draining['generation'] == active['generation'] and draining['size'] == active['size']
        assert draining['state'] == 'draining' and draining['leases'] == 1
        assert draining['retirement_reason'] == 'explicit unload'
        assert datetime.fromisoformat(draining['expires_at']) <= datetime.now(timezone.utc)
    finally:
        (manager.test.roots[0] / 'release').touch()
        await task
    await manager.wait_native_closed(active['generation'])
    assert await ps(client) == []


async def test_two_generations_of_one_name_are_reported_separately(public):
    manager, client, name, toml = public
    manager._native.max_loaded_models = 2
    async with manager.acquire_native_model(name, keep_alive=-1) as old:
        toml.write_text(toml.read_text() + '\n[runtime]\nn_ctx=8192\n')
        async with manager.acquire_native_model(name, keep_alive=-1) as new:
            models = await ps(client)
            assert len(models) == 2 and {m['generation'] for m in models} == {old.generation, new.generation}
            assert {m['state'] for m in models} == {'ready', 'draining'}
            assert len({m['pid'] for m in models}) == 2
    await manager.wait_native_closed(old.generation)
    assert len(await ps(client)) == 1


async def test_dead_child_is_not_reported_as_still_allocating_memory(public):
    manager, client, name, _ = public
    await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    worker.process.send_signal(signal.SIGKILL)
    await worker.wait_for_exit()
    assert await ps(client) == []


async def test_returned_reporting_data_cannot_mutate_a_generation(public):
    manager, _, name, _ = public
    await manager.preload_native_model(name)
    report = manager.native_reporting_snapshot()[0]
    report['metadata']['arch'] = 'wrong'
    report['memory']['buffers']['CPU']['model'] = 0
    again = manager.native_reporting_snapshot()[0]
    assert again['metadata'].get('arch') != 'wrong'
    assert again['memory']['buffers']['CPU']['model'] == 10 * MIB


async def test_large_accepted_keep_alive_cannot_overflow_reporting_timestamp(public):
    manager, client, name, _ = public
    await manager.preload_native_model(name, keep_alive=10**20)
    [model] = await ps(client)
    assert datetime.fromisoformat(model['expires_at']).year > datetime.now(timezone.utc).year + 20


async def test_memory_parser_failure_cannot_break_startup_or_inference(public, monkeypatch):
    from tllama.native_memory import NativeMemory
    manager, client, name, _ = public
    def broken(*args):
        raise ValueError('diagnostic parser failed')
    monkeypatch.setattr(NativeMemory, 'observe', broken)
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    [model] = await ps(client)
    assert model['size'] is None and model['memory']['reason'] == 'parser_error'
