"""Definition ownership covers resolution, admission, retirement and child FDs."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.model_definitions import create_definition
from tllama.repository_pins import Reservation, snapshot
from tllama.repository_store import prepare_group
from tllama.residency import ModelCleanupError, ResidencyClosed
from tllama.worker import NativeWorker, WorkerError
from tllama.worker_config import WorkerLaunchSpec
from test_model_create import gguf_file
from test_native_residency import ControlledWorker, wait_until


@pytest.fixture
async def protected(tmp_path):
    ledger, options = [], {'hold_load': False, 'hold_close': False, 'error': None}
    def factory(spec):
        worker = ControlledWorker(spec, ledger)
        if options['hold_load']:
            worker.allow_start.clear()
        if options['hold_close']:
            worker.allow_close.clear()
        worker.start_error = options['error']
        return worker
    manager = ModelManager(BackendConfig(models_dir=str(tmp_path / 'models'), max_loaded_models=2,
        runtime_overrides={'device': 'none'}), native_worker_factory=factory)
    manager.ensure_storage()
    def group(arch='llama', *, split=False):
        if split:
            sources = [gguf_file(tmp_path / f'{arch}-{i+1:05d}-of-00002.gguf', arch=arch, shard=i, count=2)
                       for i in range(2)]
        else:
            sources = [gguf_file(tmp_path / (arch + '.gguf'), arch=arch)]
        paths = manager.repository.publish(prepare_group(sources))
        return paths[0].relative_to(manager.repository.root).as_posix()
    first = group()
    create_definition(manager, 'a', '[llm]\nmodel=' + json.dumps(first) + '\n', {})
    manager.test = SimpleNamespace(ledger=ledger, options=options, group=group, first=first)
    try:
        yield manager
    finally:
        for worker in ledger:
            worker.allow_start.set()
            worker.allow_close.set()
            worker.close_error = None
        await manager.shutdown()
        assert not snapshot(manager.repository).active
        snapshot(manager.repository, remove_stale=True)


def protected_groups(manager):
    state = snapshot(manager.repository)
    assert all(not row['all_data'] for row in state.active)
    return {group for row in state.active for group in row['groups']}


async def test_snapshot_pin_covers_llm_shards_projector_and_adapter_until_reap(protected):
    manager = protected
    llm = manager.test.group(split=True)
    projector = manager.test.group('clip')
    adapter = manager.test.group('adapter')
    create_definition(manager, 'a', f'[llm]\nmodel="{llm}"\n[mmproj]\nmodel="{projector}"\n'
        f'[runtime]\nlora_path="{adapter}"\n', {}, force=True)
    generation = await manager.preload_native_model('a', keep_alive=-1)
    expected = {Path(path).parts[1] for path in (llm, projector, adapter)}
    assert protected_groups(manager) == expected
    worker = manager.test.ledger[0]
    worker.allow_close.clear()
    await manager.unload_native_model('a')
    await wait_until(lambda: worker.state == 'closing')
    assert protected_groups(manager) == expected
    worker.allow_close.set()
    await manager.wait_native_closed(generation)
    assert not protected_groups(manager)


async def test_concurrent_reuse_keeps_one_generation_pin(protected):
    manager = protected
    generations = await asyncio.gather(*(manager.preload_native_model('a') for _ in range(5)))
    assert len(set(generations)) == len(manager.test.ledger) == 1
    assert len(snapshot(manager.repository).active) == 1


async def test_replaced_and_deleted_definitions_keep_their_admitted_snapshots(protected):
    manager = protected
    old_group = Path(manager.test.first).parts[1]
    async with manager.acquire_native_model('a') as old:
        second = manager.test.group('qwen3')
        create_definition(manager, 'a', f'[llm]\nmodel="{second}"\n', {}, force=True)
        async with manager.acquire_native_model('a') as new:
            assert old.generation != new.generation
            assert protected_groups(manager) == {old_group, Path(second).parts[1]}
            await manager.delete_model('a')
            assert protected_groups(manager) == {old_group, Path(second).parts[1]}
        await manager.wait_native_closed(new.generation)
        assert protected_groups(manager) == {old_group}
    await manager.wait_native_closed(old.generation)
    assert not protected_groups(manager)


async def test_cancelled_resolution_joins_thread_before_releasing_broad_pin(protected, monkeypatch):
    manager = protected
    entered, proceed = threading.Event(), threading.Event()
    original = manager._resolve_native_definition_sync
    def delayed(*args, **kwargs):
        entered.set()
        assert proceed.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(manager, '_resolve_native_definition_sync', delayed)
    task = asyncio.create_task(manager.preload_native_model('a'))
    assert await asyncio.to_thread(entered.wait, 5)
    state = snapshot(manager.repository)
    assert state.active[0]['all_data'] and state.protects_group('b' * 64)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and snapshot(manager.repository).active
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not snapshot(manager.repository).active and not manager.test.ledger


async def test_shutdown_during_resolution_keeps_pin_until_reader_settles(protected, monkeypatch):
    manager = protected
    entered, proceed = threading.Event(), threading.Event()
    original = manager._resolve_native_definition_sync
    def delayed(*args, **kwargs):
        entered.set()
        assert proceed.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(manager, '_resolve_native_definition_sync', delayed)
    task = asyncio.create_task(manager.preload_native_model('a'))
    assert await asyncio.to_thread(entered.wait, 5)
    await manager.shutdown()
    assert snapshot(manager.repository).active[0]['all_data']
    proceed.set()
    with pytest.raises(ResidencyClosed):
        await task
    assert not snapshot(manager.repository).active and not manager.test.ledger


async def test_worker_factory_failure_releases_unadmitted_snapshot(protected, monkeypatch):
    manager = protected
    def fail(spec):
        raise RuntimeError('factory failed')
    monkeypatch.setattr(manager._native, '_factory', fail)
    with pytest.raises(RuntimeError, match='factory failed'):
        await manager.preload_native_model('a')
    assert not snapshot(manager.repository).active


async def test_foreign_managed_group_cannot_receive_an_unrelated_repository_pin(protected, tmp_path, monkeypatch):
    from repository_fixtures import register_model
    from tllama.helpers.model_toml import parse_model_toml
    manager = protected
    path, text = register_model(tmp_path / 'foreign', gguf_file(tmp_path / 'foreign.gguf'))
    foreign = path.parent.parent / parse_model_toml(text).llm_model
    resolve = manager._resolve_native_definition
    async def moved(*args, **kwargs):
        definition = await resolve(*args, **kwargs)
        return replace(definition, resolved=replace(definition.resolved,
            launch=replace(definition.resolved.launch, model=foreign)))
    monkeypatch.setattr(manager, '_resolve_native_definition', moved)
    with pytest.raises(RuntimeError, match='reserved repository'):
        await manager.preload_native_model('a')
    assert not snapshot(manager.repository).active and not manager.test.ledger


async def test_failed_resolution_and_wrong_operation_release_unused_pin(protected):
    manager = protected
    with pytest.raises(FileNotFoundError):
        await manager.preload_native_model('absent')
    assert not snapshot(manager.repository).active
    generation = await manager.preload_native_model('a')
    with pytest.raises(ValueError, match='not configured'):
        async with manager.acquire_native_model('a', operation='embedding'):
            pass
    assert len(snapshot(manager.repository).active) == 1
    assert manager.native_snapshot()[0]['generation'] == generation


async def test_cancelled_loading_waiter_leaves_generation_pin_until_owned_load_finishes(protected):
    manager = protected
    manager.test.options['hold_load'] = True
    task = asyncio.create_task(manager.preload_native_model('a'))
    await wait_until(lambda: manager.test.ledger)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(snapshot(manager.repository).active) == 1
    manager.test.ledger[0].allow_start.set()
    await wait_until(lambda: manager.native_snapshot()[0]['state'] == 'ready')


async def test_load_failure_closes_worker_before_unpin(protected):
    manager = protected
    manager.test.options.update(error=WorkerError('cannot load'), hold_close=True)
    task = asyncio.create_task(manager.preload_native_model('a'))
    await wait_until(lambda: manager.test.ledger and manager.test.ledger[0].state == 'closing')
    assert len(snapshot(manager.repository).active) == 1
    manager.test.ledger[0].allow_close.set()
    with pytest.raises(WorkerError, match='cannot load'):
        await task
    assert not snapshot(manager.repository).active


async def test_cleanup_failure_retains_group_and_capacity_until_repair(protected):
    manager = protected
    generation = await manager.preload_native_model('a')
    worker = manager.test.ledger[0]
    worker.close_error = RuntimeError('reap failed')
    try:
        await manager.unload_native_model('a')
        with pytest.raises(ModelCleanupError):
            await manager.wait_native_closed(generation)
        assert protected_groups(manager) == {Path(manager.test.first).parts[1]}
        assert manager.native_snapshot()[0]['state'] == 'cleanup_failed'
    finally:
        worker.close_error = None
        await manager._native._close_entry(manager._native._entries[generation])


async def test_pin_release_failure_is_quarantined_after_child_cleanup(protected):
    manager = protected
    generation = await manager.preload_native_model('a')
    entry = manager._native._entries[generation]
    close = entry.definition.protection.close
    def fail():
        raise RuntimeError('cannot release lock record')
    entry.definition.protection.close = fail
    try:
        await manager.unload_native_model('a')
        with pytest.raises(ModelCleanupError):
            await manager.wait_native_closed(generation)
        assert entry.worker.process.returncode == 0
        assert protected_groups(manager) == {Path(manager.test.first).parts[1]}
    finally:
        entry.definition.protection.close = close
        await manager._native._close_entry(entry)


def test_internal_fds_do_not_change_configuration_identity_or_argv(tmp_path):
    spec = WorkerLaunchSpec(model=tmp_path / 'weights.gguf', device='none')
    protected = replace(spec, resource_fds=(7, 7, 9))
    assert protected.resource_fds == (7, 9)
    assert protected == spec and protected.command(tmp_path / 'key') == spec.command(tmp_path / 'key')


@pytest.mark.parametrize('value', [True, -1, 1.5, '3'])
def test_internal_fds_reject_invalid_descriptor_values(tmp_path, value):
    with pytest.raises(ValueError, match='resource_fds'):
        WorkerLaunchSpec(model=tmp_path / 'weights.gguf', device='none', resource_fds=(value,))


async def test_cancellation_during_unused_pin_cleanup_releases_admitted_reference(protected, monkeypatch):
    manager = protected
    await manager.preload_native_model('a')
    entered, proceed = threading.Event(), threading.Event()
    original = manager._resolve_protected_native_definition
    async def resolve(*args, **kwargs):
        definition = await original(*args, **kwargs)
        close = definition.protection.close
        def delayed():
            entered.set()
            assert proceed.wait(timeout=5)
            close()
        definition.protection.close = delayed
        return definition
    monkeypatch.setattr(manager._native, '_resolve', resolve)
    task = asyncio.create_task(manager.preload_native_model('a'))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.native_snapshot()[0]['leases'] == 0
    assert len(snapshot(manager.repository).active) == 1


def runtime(tmp_path):
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    root = tmp_path / 'native-data'
    root.mkdir()
    return executable, root


async def test_direct_diagnostic_launch_pins_managed_groups_and_cleans_them(protected, tmp_path):
    manager = protected
    executable, root = runtime(tmp_path)
    worker = NativeWorker(WorkerLaunchSpec(model=manager.models_dir / manager.test.first, device='none',
        executable=executable, transport='tcp', environment={'STUB_ROOT': str(root)}, load_timeout=5))
    try:
        await worker.start()
        assert protected_groups(manager) == {Path(manager.test.first).parts[1]}
    finally:
        await worker.close()
    assert not snapshot(manager.repository).active


async def test_direct_launch_freezes_symlink_target_before_reserving(protected, tmp_path, monkeypatch):
    manager = protected
    executable, root = runtime(tmp_path)
    original = manager.models_dir / manager.test.first
    other = manager.models_dir / manager.test.group('qwen3')
    alias = tmp_path / 'alias.gguf'
    alias.symlink_to(original)
    acquire = Reservation.acquire
    def repoint(pin):
        value = acquire(pin)
        alias.unlink()
        alias.symlink_to(other)
        return value
    monkeypatch.setattr(Reservation, 'acquire', repoint)
    worker = NativeWorker(WorkerLaunchSpec(model=alias, device='none', executable=executable,
        transport='tcp', environment={'STUB_ROOT': str(root)}, load_timeout=5))
    try:
        await worker.start()
        assert worker.spec.model == original
        assert worker.command[worker.command.index('--model') + 1] == str(original)
        assert protected_groups(manager) == {original.parent.name}
    finally:
        await worker.close()


async def test_unmanaged_alias_cannot_switch_into_unreserved_data_during_start(protected, tmp_path, monkeypatch):
    manager = protected
    executable, root = runtime(tmp_path)
    original = gguf_file(tmp_path / 'external.gguf')
    alias = tmp_path / 'alias.gguf'
    alias.symlink_to(original)
    worker = NativeWorker(WorkerLaunchSpec(model=alias, device='none', executable=executable,
        transport='tcp', environment={'STUB_ROOT': str(root)}, load_timeout=5))
    reserve = worker._reserve_repository_files
    def moved():
        reserve()
        alias.unlink()
        alias.symlink_to(manager.models_dir / manager.test.first)
    monkeypatch.setattr(worker, '_reserve_repository_files', moved)
    try:
        await worker.start()
        assert worker.spec.model == original
        assert not snapshot(manager.repository).active
    finally:
        await worker.close()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX pass_fds/flock')
async def test_native_child_retains_direct_pin_after_supervisor_fd_loss(protected, tmp_path):
    manager = protected
    executable, root = runtime(tmp_path)
    worker = NativeWorker(WorkerLaunchSpec(model=manager.models_dir / manager.test.first, device='none',
        executable=executable, transport='tcp', environment={'STUB_ROOT': str(root)}, load_timeout=5))
    try:
        await worker.start()
        pin = worker._repository_pins[0]
        os.close(pin._fd)
        pin._fd = None
        assert snapshot(manager.repository, remove_stale=True).protects_group(Path(manager.test.first).parts[1])
    finally:
        await worker.close()
    assert not snapshot(manager.repository, remove_stale=True).active


@pytest.mark.parametrize('stop', ['close', 'cancel'])
async def test_close_joins_direct_reservation_acquire_and_prevents_late_spawn(protected, tmp_path, monkeypatch, stop):
    manager = protected
    executable, root = runtime(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    acquire = Reservation.acquire
    def delayed(pin):
        entered.set()
        assert proceed.wait(timeout=5)
        return acquire(pin)
    monkeypatch.setattr(Reservation, 'acquire', delayed)
    worker = NativeWorker(WorkerLaunchSpec(model=manager.models_dir / manager.test.first, device='none',
        executable=executable, transport='tcp', environment={'STUB_ROOT': str(root)}, load_timeout=5))
    starting = asyncio.create_task(worker.start())
    assert await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(worker.close()) if stop == 'close' else None
    if stop == 'cancel':
        starting.cancel()
    await asyncio.sleep(0)
    proceed.set()
    with pytest.raises(asyncio.CancelledError if stop == 'cancel' else WorkerError):
        await starting
    if closing is not None:
        await closing
    assert worker.process is None and not (root / 'launch').exists()
    assert not snapshot(manager.repository).active


@pytest.mark.skipif(os.name != 'posix', reason='POSIX inherited generation lock')
async def test_automatic_backend_attempts_inherit_the_same_generation_pin(protected, tmp_path):
    template = protected
    executable, root = runtime(tmp_path)
    config = replace(template.config, runtime_overrides={}, runtime_executable=executable,
                     backend_priority=('ROCm', 'Vulkan'))
    ledger = []
    environment = {'STUB_ROOT': str(root), 'STUB_DISCOVERY': '["ROCm0", "Vulkan0"]',
                   'STUB_DEVICE_FAILURES': '{"ROCm0": "ROCm error: no kernel image is available"}'}
    def factory(spec):
        worker = NativeWorker(replace(spec, transport='tcp', environment=environment))
        ledger.append(worker)
        return worker
    manager = ModelManager(config, native_worker_factory=factory,
                           native_launch_options={'environment': environment})
    try:
        await manager.preload_native_model('a')
        assert len(ledger) == 2 and ledger[0].process.returncode == 23
        assert ledger[0].spec.resource_fds == ledger[1].spec.resource_fds != ()
        entry = next(iter(manager._native._entries.values()))
        pin = entry.definition.protection
        os.close(pin._fd)
        pin._fd = None
        assert snapshot(manager.repository, remove_stale=True).protects_group(Path(template.test.first).parts[1])
    finally:
        await manager.shutdown()
    assert not snapshot(manager.repository, remove_stale=True).active
