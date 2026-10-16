"""Selected memory telemetry crosses public /api/ps without extra native work."""

import asyncio
from dataclasses import replace
import signal

import pytest

from test_native_memory_api import payload
from test_native_public_api import public, request_body, stream_mode
from test_native_reporting import ps
from test_native_manager import native_manager, records, wait_until
from test_worker_telemetry import prepare


@pytest.fixture
async def native_public(public, payload, monkeypatch):
    manager, client, name, toml = public
    payload['placement']['layer_groups'][-1].update(location='accelerator', device_ids=['gpu0'])
    options = {'status':200}
    original = manager._native._factory

    def factory(spec):
        worker = original(replace(spec, memory_compare=True))
        prepare(manager.test.roots[-1], payload, options['status'])
        return worker

    monkeypatch.setattr(manager._native, '_factory', factory)
    yield manager, client, name, toml, payload, options


async def loaded(manager, name):
    generation = await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[-1]
    await worker.refresh_memory()
    return generation, worker


async def test_native_ps_reports_complete_and_display_totals_without_io_or_leases(native_public):
    manager, client, name, _, _, _ = native_public
    generation, worker = await loaded(manager, name)
    # Freeze automatic sampling; a public read must consume only cached data.
    worker._memory_task.cancel()
    await asyncio.gather(worker._memory_task, return_exceptions=True)
    before = records(manager.test.roots[0], 'requests')
    for _ in range(3):
        [model] = await ps(client)
        assert model['generation'] == generation and model['leases'] == 0
        assert model['size'] == model['size_vram'] == 1300
        memory = model['memory']
        assert memory['source'] == 'native_memory_api' and memory['allocated_bytes'] == 1350
        assert memory['host_bytes'] == 50 and memory['physical_residency_bytes'] is None
        assert memory['native']['http_status'] == 200 and memory['fallback_reason'] is None
        assert memory['ollama_compatibility']['excluded_host_bytes'] == 50
        assert memory['comparison']['log']['report']['allocated_bytes'] == int(16.5 * 1024**2)
    assert records(manager.test.roots[0], 'requests') == before
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


async def test_public_ps_does_not_wait_for_a_hung_memory_refresh(native_public):
    manager, client, name, _, _, _ = native_public
    _, worker = await loaded(manager, name)
    worker._memory_task.cancel()
    await asyncio.gather(worker._memory_task, return_exceptions=True)
    root = manager.test.roots[0]
    (root / 'memory-behavior').write_text('hang')
    before = len(records(root, 'requests'))
    refresh = asyncio.create_task(worker.refresh_memory())
    try:
        await wait_until(lambda: len(records(root, 'requests')) > before)
        during = records(root, 'requests')
        [model] = await asyncio.wait_for(ps(client), 0.25)
        assert model['memory']['source'] == 'native_memory_api' and model['size'] == 1300
        assert records(root, 'requests') == during
    finally:
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)


@pytest.mark.parametrize('status', ['partial', 'unavailable'])
async def test_native_incomplete_inventory_is_public_without_filling_it_from_log(native_public, status):
    manager, client, name, _, payload, _ = native_public
    payload['status'] = status
    payload['buffers'][0]['size_bytes'] = None
    payload['missing_components'] = [{'owner_id':'model', 'component':'model_weights', 'reason':'opaque_composite'}]
    await loaded(manager, name)
    [model] = await ps(client)
    assert model['size'] is model['size_vram'] is None
    memory = model['memory']
    assert memory['source'] == 'native_memory_api' and memory['status'] == status
    assert memory['known_bytes'] == 350 and memory['unsized_buffers'] == 1
    assert memory['fallback_reason'] is None
    assert memory['ollama_compatibility']['reason'] == 'incomplete_inventory'
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200


async def test_public_known_inventory_with_unknown_placement_keeps_allocations_only(native_public):
    manager, client, name, _, payload, _ = native_public
    payload['placement']['status'] = 'partial'
    payload['placement']['layer_groups'][-1]['location'] = 'unknown'
    await loaded(manager, name)
    [model] = await ps(client)
    assert model['memory']['status'] == 'known' and model['memory']['allocated_bytes'] == 1350
    assert model['size'] is model['size_vram'] is None
    assert model['memory']['ollama_compatibility']['reason'] == 'incomplete_placement'


@pytest.mark.parametrize('status,reason', [(404, 'endpoint_not_supported'), (401, 'authentication_failed'),
                                         (503, 'not_ready'), (500, 'http_error')])
@pytest.mark.parametrize('source', ['auto', 'native'])
async def test_public_failure_modes_preserve_source_policy_and_reason(native_public, status, reason, source):
    manager, client, name, _, _, options = native_public
    manager.config = replace(manager.config, memory_source=source)
    options['status'] = status
    await loaded(manager, name)
    [model] = await ps(client)
    memory = model['memory']
    assert memory['native']['reason'] == reason and memory['native']['http_status'] == status
    if source == 'auto':
        assert memory['source'] == 'native_load_log' and memory['fallback_reason'] == reason
        assert model['size'] == int(16.5 * 1024**2) and model['size_vram'] == 0
        assert memory['scope'] == 'initial_load_allocations'
    else:
        assert memory['source'] is None and memory['status'] == 'unavailable'
        assert memory['runtime_fingerprint'] == 'b11552-ec2797836f'
        assert model['size'] is model['size_vram'] is None
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200


async def test_public_explicit_log_mode_ignores_available_native_metrics_but_can_compare(native_public):
    manager, client, name, _, _, _ = native_public
    manager.config = replace(manager.config, memory_source='log')
    await loaded(manager, name)
    [model] = await ps(client)
    memory = model['memory']
    assert memory['mode'] == 'log' and memory['source'] == 'native_load_log'
    assert model['size'] == int(16.5 * 1024**2)
    assert memory['comparison']['native']['totals']['allocated_bytes'] == 1350


async def test_public_snapshot_copies_cannot_mutate_native_inventory_or_comparison(native_public):
    manager, client, name, _, _, _ = native_public
    await loaded(manager, name)
    entry = manager.native_reporting_snapshot()[0]
    entry['memory']['measurement']['report']['buffers'][0]['size_bytes'] = 999
    [model] = await ps(client)
    model['memory']['buffers'].clear()
    model['memory']['comparison']['native']['report']['owners'].clear()
    [again] = await ps(client)
    assert again['memory']['buffers'][0]['size_bytes'] == 1000
    assert again['memory']['comparison']['native']['report']['owners']


async def test_native_public_reads_leave_idle_expiry_unchanged(native_public):
    manager, client, name, _, _, _ = native_public
    now = [1000.0]
    manager._native._clock = lambda: now[0]
    generation = await manager.preload_native_model(name, keep_alive=5)
    await manager.test.ledger[0].refresh_memory()
    for _ in range(3):
        now[0] += 1
        assert (await ps(client))[0]['leases'] == 0
    now[0] = 1006
    await manager._native.expire()
    await manager.wait_native_closed(generation)
    assert await ps(client) == []


async def test_draining_native_generation_keeps_metrics_until_its_live_stream_ends(native_public):
    manager, client, name, _, _, _ = native_public
    stream_mode(manager, 'controlled')
    generation, worker = await loaded(manager, name)
    task = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    try:
        root = manager.test.roots[0]
        await wait_until(lambda: (root / 'started').exists())
        await manager.unload_native_model(name)
        [model] = await ps(client)
        assert model['state'] == 'draining' and model['leases'] == 1
        assert model['memory']['source'] == 'native_memory_api' and model['size'] == 1300
        (root / 'release').touch()
        await task
    finally:
        (root / 'release').touch()
        await asyncio.gather(task, return_exceptions=True)
    await manager.wait_native_closed(generation)
    assert worker.process.returncode is not None and await ps(client) == []


async def test_dead_worker_is_not_kept_in_public_ps_by_its_known_cached_inventory(native_public):
    manager, client, name, _, _, _ = native_public
    _, worker = await loaded(manager, name)
    worker.process.send_signal(signal.SIGKILL)
    await worker.wait_for_exit()
    assert await ps(client) == []


async def test_concurrent_generations_do_not_join_snapshot_local_ids(native_public):
    manager, client, name, toml, payload, _ = native_public
    manager._native.max_loaded_models = 2
    async with manager.acquire_native_model(name, keep_alive=-1) as old:
        await old.worker.refresh_memory()
        toml.write_text(toml.read_text() + '\n[runtime]\nn_ctx=8192\n')
        payload['buffers'][0]['size_bytes'] = 2000
        async with manager.acquire_native_model(name, keep_alive=-1) as new:
            await new.worker.refresh_memory()
            models = {model['generation']:model for model in await ps(client)}
            assert models[old.generation]['memory']['allocated_bytes'] == 1350
            assert models[new.generation]['memory']['allocated_bytes'] == 2350
            assert models[old.generation]['memory']['buffers'][0]['id'] == models[new.generation]['memory']['buffers'][0]['id']
