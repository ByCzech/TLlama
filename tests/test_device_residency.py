"""Device admission races use controlled workers, not simulated GPU evidence."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tllama.config import BackendConfig
from tllama.native_config import resolve_worker_config
from tllama.residency import NativeResidency, ModelDefinition, ModelCleanupError, ModelUnloaded
from tllama.worker import WorkerStartupError
from test_native_residency import ControlledWorker, wait_until


@pytest.fixture
async def devices(monkeypatch):
    pools = []
    async def discover(spec):
        return ('CUDA0', 'Vulkan0', 'Vulkan1', 'ROCm0')
    monkeypatch.setattr('tllama.automatic_worker.discover_devices', discover)
    def make(rows=(), **kwargs):
        definitions, ledger = {}, []
        now = [0.0]
        options = {'error': {}, 'close_error': {}}
        async def resolve(name, ctx):
            definition = definitions.get(name, 'Vulkan1')
            if isinstance(definition, str) or definition is None:
                config = BackendConfig(runtime_overrides={} if definition is None else {'device': definition},
                                       backend_priority=('CUDA', 'Vulkan'))
                resolved = resolve_worker_config(config, Path('/models')/(name+'.gguf'), num_ctx=ctx)
            else:
                resolved = definition
            return ModelDefinition(name, resolved, ctx)
        def factory(spec):
            worker = ControlledWorker(spec, ledger)
            worker.start_error = options['error'].get(spec.device)
            worker.close_error = options['close_error'].get(spec.device)
            return worker
        pool = NativeResidency(resolve, max_loaded_models=0, device_limits=rows,
                               worker_factory=factory, clock=lambda: now[0], **kwargs)
        pool.test = SimpleNamespace(definitions=definitions, ledger=ledger, now=now, options=options)
        pools.append(pool)
        return pool
    yield make
    for pool in pools:
        for worker in pool.test.ledger:
            worker.allow_start.set()
            worker.allow_close.set()
            worker.close_error = None
        # Tests can deliberately quarantine a generation. Retry only in test cleanup.
        for entry in pool._entries.values():
            if entry.state == 'cleanup_failed':
                entry.close_task = None
                entry.state = 'draining'
                # Controlled children have no OS resources. Reset the proxy's
                # deliberately failed close task solely for fixture teardown.
                if hasattr(entry.worker, '_close_task'):
                    entry.worker._close_task = None
        await pool.shutdown()


async def test_dgpu_replacement_keeps_igpu_and_cpu_generations_and_waits_for_reap(devices):
    pool = devices([{'name': 'Vulkan1', 'max_loaded_models': 1}, {'name': 'Vulkan0', 'max_loaded_models': 2}])
    pool.test.definitions.update(E='Vulkan0', R='Vulkan0', C='none')
    preserved = {name: await pool.preload(name, keep_alive_seconds=None) for name in ['E','R','C']}
    async with pool.lease('A') as old:
        old.worker.allow_close.clear()
        incoming = asyncio.create_task(pool.preload('B'))
        await wait_until(lambda: pool._entries[old.generation].retired)
        assert not incoming.done() and len(pool.test.ledger) == 4
    await wait_until(lambda: old.worker.state == 'closing')
    assert not incoming.done() and pool.occupied == 4
    old.worker.allow_close.set()
    generation = await incoming
    assert old.worker.process.returncode == 0 and generation != old.generation
    assert pool.occupied == 4
    assert {name: await pool.preload(name) for name in preserved} == preserved


async def test_idle_lru_on_same_device_and_forever_is_not_eviction_protection(devices):
    pool = devices([{'name':'Vulkan1','max_loaded_models':2}])
    a = await pool.preload('a', keep_alive_seconds=None)
    pool.test.now[0] += 1
    b = await pool.preload('b', keep_alive_seconds=None)
    pool.test.now[0] += 1
    assert await pool.preload('a', keep_alive_seconds=None) == a
    c = await pool.preload('c')
    assert b not in pool._entries and {a,c} == set(pool._entries)


@pytest.mark.parametrize('rows', [[], [{'max_loaded_models':1}]])
async def test_implicit_default_one_is_per_device_not_global(devices, rows):
    pool = devices(rows)
    pool.test.definitions.update(a='Vulkan0', b='Vulkan1', c='none')
    for name in ['a','b','c']:
        await pool.preload(name)
    assert pool.occupied == 3


async def test_zero_named_and_zero_default_are_unlimited(devices):
    pool = devices([{'name':'Vulkan1','max_loaded_models':0}, {'max_loaded_models':0}])
    pool.test.definitions.update(c='none', d='none')
    for name in ['a','b','c','d']:
        await pool.preload(name)
    assert pool.occupied == 4 and len(pool.test.ledger) == 4


async def test_loading_and_closing_reservations_prevent_parallel_allocation(devices):
    pool = devices()
    original = pool._factory
    def factory(spec):
        worker = original(spec)
        worker.allow_start.clear()
        return worker
    pool._factory = factory
    one = asyncio.create_task(pool.preload('a'))
    await wait_until(lambda: len(pool.test.ledger) == 1)
    two = asyncio.create_task(pool.preload('b'))
    await asyncio.sleep(0)
    assert len(pool.test.ledger) == 1
    pool.test.ledger[0].allow_start.set()
    await one
    await wait_until(lambda: len(pool.test.ledger) == 2)
    assert pool.test.ledger[0].process.returncode == 0
    pool.test.ledger[1].allow_start.set()
    await two


async def test_same_generation_shared_leases_use_one_slot_and_cancelled_waiter_leaks_none(devices):
    pool = devices()
    async with pool.lease('a') as one:
        async with pool.lease('a') as two:
            assert one.generation == two.generation and pool.occupied == 1
            waiting = asyncio.create_task(pool.preload('b'))
            await wait_until(lambda: pool._entries[one.generation].retired)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert len(pool.test.ledger) == 1
    await pool.wait_closed(one.generation)
    assert pool.occupied == 0
    await pool.preload('c')


async def test_reload_waits_for_old_lease_and_close_on_same_device(devices):
    pool = devices()
    async with pool.lease('a', num_ctx=1024) as old:
        new = asyncio.create_task(pool.preload('a', num_ctx=2048))
        await wait_until(lambda: pool._entries[old.generation].retired)
        assert len(pool.test.ledger) == 1
    number = await new
    assert number != old.generation and pool.test.ledger[-1].spec.n_ctx == 2048
    assert old.worker.process.returncode == 0


async def test_failed_cleanup_blocks_only_its_device(devices):
    pool = devices()
    pool.test.options['close_error']['Vulkan1'] = RuntimeError('still owns memory')
    await pool.preload('a')
    with pytest.raises(ModelCleanupError, match='failed cleanup'):
        await pool.preload('b')
    pool.test.definitions['cpu'] = 'none'
    await pool.preload('cpu')
    assert pool.occupied == 2 and len(pool.test.ledger) == 2


async def test_helper_and_csv_reserve_each_touched_device_once(devices):
    pool = devices()
    base = resolve_worker_config(BackendConfig(runtime_overrides={'device':'Vulkan1'}), Path('/main'))
    launch = replace(base.launch, device='Vulkan1,Vulkan1', draft=Path('/draft'),
        runtime_options={**base.launch.runtime_options, 'spec_type':'draft-mtp', 'draft_device':'none'})
    pool.test.definitions['composite'] = replace(base, launch=launch)
    number = await pool.preload('composite')
    assert pool._entries[number].devices == {'Vulkan1','CPU'} and pool.occupied == 1
    pool.test.definitions['cpu'] = 'none'
    await pool.preload('cpu')
    assert number not in pool._entries


async def test_automatic_fallback_transfers_reservation_after_failed_child_reaped(devices):
    pool = devices()
    pool.test.definitions['auto'] = None
    pool.test.options['error']['CUDA0'] = WorkerStartupError('CUDA driver version is insufficient')
    number = await pool.preload('auto')
    assert [worker.spec.device for worker in pool.test.ledger] == ['CUDA0','Vulkan0,Vulkan1']
    assert pool.test.ledger[0].process.returncode == 0
    assert pool._entries[number].devices == {'Vulkan0','Vulkan1'}
    assert pool.occupied == 1


async def test_full_candidate_quota_waits_without_falling_back(devices):
    pool = devices()
    pool.test.definitions.update(existing='CUDA0', auto=None)
    async with pool.lease('existing') as old:
        incoming = asyncio.create_task(pool.preload('auto'))
        await wait_until(lambda: pool._entries[old.generation].retired)
        assert len(pool.test.ledger) == 1 and not incoming.done()
    number = await incoming
    assert pool.test.ledger[-1].spec.device == 'CUDA0'
    assert pool._entries[number].devices == {'CUDA0'}


async def test_cancel_and_unload_automatic_capacity_wait_preserves_other_devices(devices):
    pool = devices()
    pool.test.definitions.update(existing='CUDA0', auto=None, other='none')
    saved = await pool.preload('other')
    async with pool.lease('existing') as old:
        incoming = asyncio.create_task(pool.preload('auto'))
        await wait_until(lambda: pool._entries[old.generation].retired)
        assert await pool.unload('auto')
        with pytest.raises(ModelUnloaded):
            await incoming
    await pool.wait_closed(old.generation)
    assert await pool.preload('other') == saved
    assert all(entry.devices == {'CPU'} for entry in pool._entries.values())


async def test_unavailable_warning_is_once_and_only_after_complete_discovery(devices, caplog):
    pool = devices([{'name':'Absent42','max_loaded_models':2}])
    pool.test.definitions.update(a=None, b=None)
    await pool.preload('a')
    await pool.unload('a')
    await pool.preload('b')
    assert sum('Absent42 is unavailable' in r.message for r in caplog.records) == 1
    assert pool._device_limits['Absent42'] == 2


async def test_discovery_failure_is_not_absence_evidence_and_cpu_still_has_quota(devices, monkeypatch, caplog):
    async def broken(spec):
        raise OSError('discovery unavailable')
    monkeypatch.setattr('tllama.automatic_worker.discover_devices', broken)
    pool = devices([{'name':'Absent42','max_loaded_models':1}])
    pool.test.definitions['a'] = None
    number = await pool.preload('a')
    assert pool._entries[number].devices == {'CPU'}
    assert not any('Absent42 is unavailable' in r.message for r in caplog.records)


async def test_failed_start_releases_reservation_for_independent_retry(devices):
    pool = devices()
    pool.test.options['error']['Vulkan1'] = WorkerStartupError('bad model, not a backend failure')
    with pytest.raises(WorkerStartupError, match='bad model'):
        await pool.preload('a')
    assert pool.occupied == 0 and pool.test.ledger[0].process.returncode == 0
    pool.test.options['error'].clear()
    await pool.preload('a')
    assert pool.occupied == 1


async def test_failed_automatic_cleanup_cannot_release_quota_or_try_next_backend(devices):
    pool = devices()
    pool.test.definitions['auto'] = None
    pool.test.options['error']['CUDA0'] = WorkerStartupError('CUDA driver version is insufficient')
    pool.test.options['close_error']['CUDA0'] = RuntimeError('cannot reap CUDA child')
    with pytest.raises(ModelCleanupError, match='clean failed generation'):
        await pool.preload('auto')
    assert len(pool.test.ledger) == 1
    [entry] = pool._entries.values()
    assert entry.devices == {'CUDA0'} and entry.state == 'cleanup_failed'
    pool.test.definitions['cuda'] = 'CUDA0'
    with pytest.raises(ModelCleanupError, match='failed cleanup'):
        await pool.preload('cuda')


async def test_automatic_wait_deadline_reaps_generation_but_does_not_cancel_busy_inference(devices):
    pool = devices()
    automatic = resolve_worker_config(BackendConfig(backend_priority=('CUDA',)), Path('/auto'))
    pool.test.definitions.update(existing='CUDA0', auto=replace(automatic,
        launch=replace(automatic.launch, load_timeout=0.02)))
    async with pool.lease('existing') as old:
        with pytest.raises(WorkerStartupError, match='deadline'):
            await pool.preload('auto')
        assert old.worker.state == 'ready' and old.worker.process.returncode is None
        assert len(pool.test.ledger) == 1 and pool.occupied == 1
    await pool.wait_closed(old.generation)
    assert pool.occupied == 0


async def test_shutdown_cancels_automatic_capacity_wait_and_drains_all_slots(devices):
    pool = devices()
    pool.test.definitions.update(existing='CUDA0', auto=None)
    async with pool.lease('existing') as old:
        incoming = asyncio.create_task(pool.preload('auto'))
        await wait_until(lambda: pool._entries[old.generation].retired)
        await pool.shutdown()
        with pytest.raises((ModelUnloaded, WorkerStartupError)):
            await incoming
        assert pool.occupied == 0 and old.worker.state == 'closed'


async def test_simultaneous_multidevice_admissions_are_atomic(devices):
    pool = devices()
    base = resolve_worker_config(BackendConfig(runtime_overrides={'device':'Vulkan0,Vulkan1'}), Path('/main'))
    pool.test.definitions.update(a=base, b=base)
    gate = asyncio.Event()
    async def use(name):
        async with pool.lease(name):
            gate.set()
            await asyncio.sleep(0.01)
    await asyncio.wait_for(asyncio.gather(use('a'),use('b')), 2)
    assert len(pool.test.ledger) == 2 and pool.test.ledger[0].process.returncode == 0
    assert pool.occupied == 1
