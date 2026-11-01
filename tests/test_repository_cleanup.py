"""Only retired, unreferenced and unreserved managed groups are reclaimed."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import httpx
import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.metadata_cache import (get_metadata_cache_path, get_digest_cache_path,
    save_digest_cache, save_metadata_cache, build_model_file_fingerprint)
from tllama.main import app
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_cleanup import QUEUE, collect_pending, remember_definition
from tllama.repository_pins import Reservation
from tllama.repository_store import RepositoryStore, prepare_group
from tllama.routers import ollama
from test_model_create import gguf_file
from test_native_residency import wait_until
from test_repository_pins import child
from test_repository_worker_pins import protected
from test_repository_runtime import hf


def text_for(path, projector=None, lora=None):
    text = '[llm]\nmodel=' + json.dumps(path) + '\n'
    if projector is not None:
        text += '[mmproj]\nmodel=' + json.dumps(projector) + '\n'
    if lora is not None:
        text += '[runtime]\nlora_path=' + json.dumps(lora) + '\n'
    return text


@pytest.fixture
def repo(manager, tmp_path):
    def publish(arch='llama', count=1, upload=False):
        sources = [gguf_file(tmp_path / f'{arch}-{i}.gguf', arch=arch,
                            shard=i if count > 1 else None, count=count if count > 1 else None)
                   for i in range(count)]
        if upload:
            blobs = []
            root = manager.repository._directory('.tllama/uploads')
            for source in sources:
                path = root / sha256(source.read_bytes()).hexdigest()
                os.link(source, path)
                save_digest_cache(manager.metadata_cache_dir, 'upload', path,
                                  {'content_sha256': path.name, 'source': 'local'})
                blobs.append(path)
            group = prepare_group(blobs, names=[f'model-{i+1:05d}-of-{count:05d}.gguf'
                                               if count > 1 else 'model.gguf' for i in range(count)])
        else:
            blobs = []
            group = prepare_group(sources)
        paths = manager.repository.publish(group, link_owned=upload)
        relative = paths[0].relative_to(manager.models_dir).as_posix()
        caches = [cache for path in [*paths, *blobs] for cache in
                  (get_metadata_cache_path(manager.metadata_cache_dir, path),
                   get_digest_cache_path(manager.metadata_cache_dir, path))]
        return SimpleNamespace(relative=relative, group=paths[0].parent.name,
                               paths=paths, blobs=blobs, caches=caches)
    manager.test_publish = publish
    return manager


@pytest.mark.parametrize('count', [1, 2])
def test_last_definition_removes_complete_group_caches_and_owned_upload_hardlinks(repo, count):
    data = repo.test_publish(count=count, upload=True)
    unrelated = repo.test_publish('qwen3')
    create_definition(repo, 'a', text_for(data.relative), {})
    result = repo.delete_model_definition('a')
    assert result['kept_model_file'] is None
    assert result['cleanup']['removed_groups'] == [data.group]
    assert not result['cleanup']['warnings']
    assert not data.paths[0].parent.exists()
    assert not any(path.exists() for path in [*data.blobs, *data.caches])
    assert unrelated.paths[0].is_file()  # Targeted cleanup is not whole-store prune.
    assert not (repo.models_dir / QUEUE).exists()


def test_copy_shares_all_groups_and_last_reference_reclaims_each(repo):
    llm, projector, lora = [repo.test_publish(arch) for arch in ('llama', 'clip', 'adapter')]
    create_definition(repo, 'a', text_for(llm.relative, projector.relative, lora.relative), {})
    copy_definition(repo, 'a', 'collection/copy')
    first = repo.delete_model_definition('a')['cleanup']
    expected = {item.group for item in (llm, projector, lora)}
    assert set(first['shared_groups']) == expected and not first['removed_files']
    assert all(path.exists() for item in (llm, projector, lora) for path in [*item.paths, *item.caches])
    last = repo.delete_model_definition('collection/copy')['cleanup']
    assert set(last['removed_groups']) == expected


def test_independent_model_keeps_shared_projector_and_lora_without_retaining_old_llm(repo):
    old, new, projector, lora = [repo.test_publish(arch) for arch in ('llama', 'qwen3', 'clip', 'adapter')]
    create_definition(repo, 'a', text_for(old.relative, projector.relative, lora.relative), {})
    create_definition(repo, 'b', text_for(new.relative, projector.relative, lora.relative), {})
    report = repo.delete_model_definition('a')['cleanup']
    assert report['removed_groups'] == [old.group]
    assert set(report['shared_groups']) == {projector.group, lora.group}
    assert all(path.exists() for item in (new, projector, lora) for path in item.paths)


@pytest.mark.parametrize('operation', ['create', 'copy'])
def test_forced_replacement_reclaims_only_old_unused_groups(repo, operation):
    old, new = repo.test_publish(), repo.test_publish('qwen3')
    create_definition(repo, 'target', text_for(old.relative), {})
    create_definition(repo, 'source', text_for(new.relative), {})
    if operation == 'create':
        result = create_definition(repo, 'target', text_for(new.relative), {}, force=True)
    else:
        result = copy_definition(repo, 'source', 'target', force=True)
    assert result['cleanup']['removed_groups'] == [old.group]
    assert not old.paths[0].exists() and new.paths[0].is_file()
    assert repo.read_model_definition('target') == text_for(new.relative)


def test_force_that_preserves_group_keeps_inode_and_caches(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    before = [path.stat() for path in [*data.paths, *data.caches]]
    result = create_definition(repo, 'a', text_for(data.relative) + '# Edited\n', {}, force=True)
    assert result['cleanup']['shared_groups'] == [data.group]
    assert [(path.stat().st_ino, path.stat().st_mtime_ns) for path in [*data.paths, *data.caches]] == [
        (value.st_ino, value.st_mtime_ns) for value in before]


def test_preflight_and_failed_force_do_not_schedule_or_reclaim_old_groups(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    assert create_definition(repo, 'a', text_for(data.relative), {}, force=True, dry_run=True)['status'] == 'ready'
    with pytest.raises(ValueError):
        create_definition(repo, 'a', 'not toml', {}, force=True)
    assert data.paths[0].is_file() and not (repo.models_dir / QUEUE).exists()


@pytest.mark.parametrize('kind', ['syntax', 'unmanaged', 'wrong-depth', 'symlink', 'unreadable'])
def test_incomplete_other_definition_inventory_blocks_all_physical_cleanup(repo, kind, tmp_path, monkeypatch):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    bad = repo.local_models_dir / 'bad.toml'
    bad.write_text('broken toml' if kind == 'syntax' else text_for(data.relative))
    if kind == 'unmanaged':
        bad.write_text(text_for('Local/legacy.gguf'))
    elif kind == 'wrong-depth':
        bad.unlink()
        bad = repo.local_models_dir / 'nested/bad.toml'
        bad.parent.mkdir()
        bad.write_text(text_for(data.relative))
    elif kind == 'symlink':
        bad.unlink()
        other = tmp_path / 'outside.toml'
        other.write_text(text_for(data.relative))
        bad.symlink_to(other)
    elif kind == 'unreadable':
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path == bad:
                raise PermissionError('unreadable definition')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'read_text', read)
    result = repo.delete_model_definition('a')
    assert result['cleanup']['warnings']
    assert data.paths[0].is_file() and (repo.models_dir / QUEUE).is_file()
    bad.unlink()
    assert collect_pending(repo.repository)['removed_groups'] == [data.group]


def test_broken_removed_definition_is_deleted_without_guessing_its_groups(repo):
    data = repo.test_publish()
    (repo.local_models_dir / 'bad.toml').write_text('broken toml')
    result = repo.delete_model_definition('bad')
    assert result['cleanup']['warnings'] and data.paths[0].is_file()
    assert not (repo.local_models_dir / 'bad.toml').exists()


@pytest.mark.parametrize('scope', ['group', 'all-data', 'upload-path'])
def test_live_reservation_defers_until_release_then_retries_automatically(repo, scope):
    data = repo.test_publish(upload=True)
    create_definition(repo, 'a', text_for(data.relative), {})
    options = {'groups': [data.group]} if scope == 'group' else (
        {'all_data': True} if scope == 'all-data' else {'paths': ['.tllama/uploads']})
    with Reservation(RepositoryStore(repo.models_dir), 'other daemon', **options):
        result = repo.delete_model_definition('a')
        assert result['cleanup']['deferred_groups'] == [data.group]
        assert data.paths[0].is_file() and all(path.exists() for path in data.blobs)
    assert not data.paths[0].exists() and not any(path.exists() for path in data.blobs)


@pytest.mark.parametrize('change', ['invalid-live', 'retained', 'unknown-entry'])
def test_unreadable_or_retained_reservations_never_authorize_deletion(repo, change):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    pin = Reservation(repo.repository, 'test', groups=[data.group], retain_on_crash=change == 'retained').acquire()
    try:
        if change == 'invalid-live':
            os.lseek(pin._fd, 1, os.SEEK_SET)
            os.write(pin._fd, b'!')
        elif change == 'retained':
            os.close(pin._fd)
            pin._fd = None  # Owner death leaves retain metadata.
        else:
            (pin.path.parent / 'unexpected').write_text('unknown')
        report = repo.delete_model_definition('a')['cleanup']
        assert data.paths[0].is_file()
        assert report['deferred_groups'] == [data.group]
        assert bool(report['warnings']) is (change != 'retained')
    finally:
        if change == 'unknown-entry':
            (pin.path.parent / 'unexpected').unlink()
        if pin._fd is not None:
            pin.close()
        else:
            pin.path.unlink()
            collect_pending(repo.repository)
    assert not data.paths[0].exists()


def test_new_definition_protects_a_previously_deferred_group(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    with Reservation(repo.repository, 'worker', groups=[data.group]):
        repo.delete_model_definition('a')
        create_definition(repo, 'rescued', text_for(data.relative), {})
    assert data.paths[0].is_file() and not (repo.models_dir / QUEUE).exists()


def test_narrowing_broad_resolution_pin_retries_unrelated_retired_groups(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    with Reservation(repo.repository, 'resolving', all_data=True) as pin:
        assert repo.delete_model_definition('a')['cleanup']['deferred_groups'] == [data.group]
        pin.update(groups=['f' * 64], all_data=False)
        assert not data.paths[0].exists()


def test_startup_retries_dead_owner_candidates_without_pruning_unrelated_data(repo):
    data, unrelated = repo.test_publish(), repo.test_publish('qwen3')
    create_definition(repo, 'a', text_for(data.relative), {})
    pin = Reservation(repo.repository, 'worker', groups=[data.group]).acquire()
    repo.delete_model_definition('a')
    os.close(pin._fd)
    pin._fd = None  # Simulate dead owner, avoiding a normal release callback.
    other = ModelManager(BackendConfig(models_dir=str(repo.models_dir)))
    other.ensure_storage()
    assert not data.paths[0].exists() and unrelated.paths[0].is_file()


def test_live_foreign_process_defers_and_death_allows_recheck(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    code = '''
import sys
from tllama.repository_pins import Reservation
from tllama.repository_store import RepositoryStore
pin = Reservation(RepositoryStore(sys.argv[1]), 'worker', groups=[sys.argv[2]]).acquire()
print('ready', flush=True)
sys.stdin.read()
'''
    with child(code, repo.models_dir, data.group) as process:
        assert repo.delete_model_definition('a')['cleanup']['deferred_groups'] == [data.group]
        assert data.paths[0].is_file()
        process.kill()
        process.wait(timeout=5)
        assert collect_pending(repo.repository)['removed_groups'] == [data.group]


@pytest.mark.parametrize('fault', ['extra-file', 'weight-symlink', 'cache-symlink', 'upload-symlink', 'queue-corrupt'])
def test_unexpected_data_or_control_entries_are_preserved_and_reported(repo, tmp_path, fault):
    data = repo.test_publish(upload=True)
    create_definition(repo, 'a', text_for(data.relative), {})
    outside = tmp_path / 'untouched'
    outside.write_bytes(b'keep me')
    if fault == 'extra-file':
        (data.paths[0].parent / 'notes.txt').write_text('keep')
    elif fault in {'weight-symlink', 'cache-symlink', 'upload-symlink'}:
        path = {'weight-symlink': data.paths[0], 'cache-symlink': data.caches[0],
                'upload-symlink': data.blobs[0]}[fault]
        path.unlink()
        path.symlink_to(outside)
    else:
        (repo.models_dir / QUEUE).write_text('invalid queue')
    result = repo.delete_model_definition('a')
    assert result['cleanup']['warnings']
    assert data.paths[0].exists() and outside.read_bytes() == b'keep me'
    assert not result['cleanup']['removed_files']


def test_interrupted_unlink_keeps_durable_file_list_and_resumes(repo, monkeypatch):
    data = repo.test_publish(count=2, upload=True)
    create_definition(repo, 'a', text_for(data.relative), {})
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == data.paths[1]:
            raise PermissionError('injected second-shard failure')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        report = repo.delete_model_definition('a')['cleanup']
    assert report['warnings'] and report['deferred_groups'] == [data.group]
    assert not data.paths[0].exists() and data.paths[1].is_file()
    assert not any(path.exists() for path in data.blobs)
    assert collect_pending(repo.repository)['removed_groups'] == [data.group]
    assert not data.paths[0].parent.exists()


def test_definition_and_staging_removal_are_durable_before_group_unlink(repo, monkeypatch):
    import tllama.repository_cleanup as cleanup
    import tllama.repository_store as storage
    data, new = repo.test_publish(upload=True), repo.test_publish('qwen3')
    create_definition(repo, 'a', text_for(data.relative), {})
    flushed = []
    real_sync, real_unlink = storage.sync_directory, Path.unlink
    def sync(path):
        flushed.append(path)
        return real_sync(path)
    def unlink(path, *args, **kwargs):
        if path == data.paths[0]:
            assert repo.local_models_dir in flushed
            assert data.blobs[0].parent in flushed
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(storage, 'sync_directory', sync)
    monkeypatch.setattr(cleanup, 'sync_directory', sync)
    monkeypatch.setattr(Path, 'unlink', unlink)
    create_definition(repo, 'a', text_for(new.relative), {}, force=True)
    assert not data.paths[0].exists()


def test_delayed_cache_writes_cannot_resurrect_reclaimed_group_cache(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    fingerprint = build_model_file_fingerprint(data.paths[0])
    repo.delete_model_definition('a')
    for save in (save_metadata_cache, save_digest_cache):
        save(repo.metadata_cache_dir, 'a', data.paths[0], {'captured': 'before deletion'},
             fingerprint=fingerprint, repository=repo.repository)
    assert not any(path.exists() for path in data.caches)


def test_independent_upload_copy_is_not_an_owned_hardlink_of_reclaimed_group(repo):
    data = repo.test_publish(upload=True)
    create_definition(repo, 'a', text_for(data.relative), {})
    raw = data.blobs[0].read_bytes()
    data.blobs[0].unlink()
    data.blobs[0].write_bytes(raw)
    repo.delete_model_definition('a')
    assert not data.paths[0].exists() and data.blobs[0].read_bytes() == raw


def test_queue_write_failure_keeps_weights_but_does_not_make_rm_fail(repo, monkeypatch):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    def fail(*args):
        raise OSError('cannot persist cleanup queue')
    monkeypatch.setattr('tllama.repository_runtime.write_json', fail)
    result = repo.delete_model_definition('a')
    assert result['cleanup']['warnings'] and data.paths[0].is_file()
    assert not (repo.local_models_dir / 'a.toml').exists()


@pytest.mark.parametrize('fault', ['unknown-version', 'boolean-version', 'traversal'])
def test_unknown_queue_schema_or_paths_cannot_authorize_unlink(repo, fault):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    value = {'version': 1, 'groups': {data.group: ['weights.gguf']}}
    if fault == 'traversal':
        value['groups'][data.group] = ['../../Local/a.toml']
    else:
        value['version'] = True if fault == 'boolean-version' else 999
    queue = repo.models_dir / QUEUE
    queue.write_text(json.dumps(value))
    before = queue.read_bytes()
    result = repo.delete_model_definition('a')
    assert result['cleanup']['warnings'] and data.paths[0].is_file()
    assert queue.read_bytes() == before


def test_crash_before_definition_commit_never_authorizes_removal_of_old_data(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    with repo.repository.lock():
        assert remember_definition(repo.repository, repo.local_models_dir / 'a.toml') == []
    assert (repo.models_dir / QUEUE).exists()
    other = ModelManager(BackendConfig(models_dir=str(repo.models_dir)))
    other.ensure_storage()
    assert data.paths[0].is_file() and not (repo.models_dir / QUEUE).exists()


def test_final_reference_and_pin_checks_hold_same_lock_through_unlink(repo, monkeypatch):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    entered, other_acquired = threading.Event(), threading.Event()
    other = RepositoryStore(repo.models_dir)
    def compete():
        assert entered.wait(5)
        with other.lock():
            other_acquired.set()
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == data.paths[0]:
            entered.set()
            assert not other_acquired.wait(0.1)
        return original(path, *args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempt = pool.submit(compete)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', unlink)
            repo.delete_model_definition('a')
        attempt.result(timeout=5)
    assert other_acquired.is_set() and not data.paths[0].exists()


async def test_draining_generation_keeps_weights_until_reaped(protected):
    manager = protected
    path = manager.models_dir / manager.test.first
    async with manager.acquire_native_model('a') as lease:
        result = await manager.delete_model('a')
        assert result['cleanup']['deferred_groups'] == [path.parent.name]
        assert path.is_file()
        worker = manager.test.ledger[0]
        worker.allow_close.clear()
    await wait_until(lambda: worker.state == 'closing')
    assert path.is_file()
    worker.allow_close.set()
    await manager.wait_native_closed(lease.generation)
    assert not path.exists()


async def test_old_generation_pins_survive_force_and_cleanup_failure(protected):
    manager = protected
    old = manager.models_dir / manager.test.first
    async with manager.acquire_native_model('a') as lease:
        new = manager.test.group('qwen3')
        result = create_definition(manager, 'a', text_for(new), {}, force=True)
        assert result['cleanup']['deferred_groups'] == [old.parent.name]
        worker = manager.test.ledger[0]
        worker.close_error = RuntimeError('cannot reap')
    await manager.unload_native_model('a')
    await wait_until(lambda: manager.native_snapshot()[0]['state'] == 'cleanup_failed')
    assert old.is_file()
    worker.close_error = None
    entry = manager._native._entries[lease.generation]
    await manager._native._close_entry(entry)
    assert not old.exists() and (manager.models_dir / new).is_file()


def test_changed_hf_origin_reclaims_old_group_without_touching_new_definition(repo, hf):
    hf.add('model.gguf')
    old = Path(repo._pull_hf_file_sync('ns/repo', 'model.gguf'))
    hf.add('model.gguf', arch='qwen3')
    new = Path(repo._pull_hf_file_sync('ns/repo', 'model.gguf'))
    assert not old.exists() and new.is_file()


async def test_ollama_delete_remains_successful_and_reports_physical_cleanup(repo, monkeypatch):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    monkeypatch.setattr(ollama, 'model_manager', repo)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.request('DELETE', '/api/delete', json={'name': 'a'})
    assert response.status_code == 200 and response.json()['status'] == 'success'
    assert response.json()['cleanup']['removed_groups'] == [data.group]
    assert response.json()['kept_model_file'] is None
