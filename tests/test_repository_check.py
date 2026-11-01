"""Read-only local daemon checks, independent hashes and protection."""

from hashlib import sha256
import json
import os
from pathlib import Path
import threading

import pytest

from tllama.helpers.metadata_cache import get_metadata_cache_path
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_check import check_repository, CheckCancelled
from tllama.repository_cleanup import QUEUE
from tllama.repository_pins import Reservation, snapshot
from tllama.repository_store import RepositoryStore
from test_repository_cleanup import repo, text_for


def before(root):
    return {path.relative_to(root).as_posix():
            (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in root.rglob('*') if not path.is_symlink()}


@pytest.mark.parametrize('hashes', [False, True])
def test_healthy_shared_shards_projector_and_adapter_are_checked_without_cache_writes(repo, hashes):
    llm = repo.test_publish(count=2)
    projector, lora = repo.test_publish('clip'), repo.test_publish('adapter')
    create_definition(repo, 'a', text_for(llm.relative, projector.relative, lora.relative), {})
    copy_definition(repo, 'a', 'collection/copy')
    previous = before(repo.models_dir)
    report = check_repository(repo.repository, hashes=hashes)
    assert report['ok'] and report['complete'] and not report['issues']
    assert report['summary'] == {'definitions': 2, 'groups': 3, 'caches': 8, 'staging': 0,
                                 'errors': 0, 'warnings': 0}
    assert all(row['valid'] and row['hash_verified'] is hashes and row['usage'] == 'referenced'
               and row['references'] == ['a', 'collection/copy'] for row in report['groups'])
    after = before(repo.models_dir)
    if hashes:
        # Only the temporary coordination record's directory mtime may differ.
        previous.pop('.tllama/pins')
        after.pop('.tllama/pins')
    assert previous == after


def test_structural_check_does_not_hash_or_claim_content_verification(repo, monkeypatch):
    import tllama.repository_check as checker
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    def forbidden(*args):
        pytest.fail('default check read full weight contents')
    monkeypatch.setattr(checker, '_hash_file', forbidden)
    report = check_repository(repo.repository)
    assert report['ok'] and not report['groups'][0]['hash_verified']


def test_full_hashes_ignore_cached_digests_and_detect_changed_bytes_with_restored_fingerprint(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    path = data.paths[0]
    info = path.stat()
    # Change the final architecture-string byte without breaking the GGUF
    # structure, and restore size/mtime so the old digest cache still matches.
    raw = path.read_bytes()
    assert raw[-1:] != b'X'
    path.write_bytes(raw[:-1] + b'X')
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert check_repository(repo.repository)['ok']
    report = check_repository(repo.repository, hashes=True)
    assert not report['ok'] and any(row['code'] == 'group_hash_invalid' for row in report['issues'])
    assert not report['groups'][0]['hash_verified']


@pytest.mark.parametrize('fault', ['missing-shard', 'extra-file', 'weight-symlink', 'bad-header', 'bad-name'])
def test_group_corruption_is_reported_and_preserved(repo, tmp_path, fault):
    data = repo.test_publish(count=2)
    create_definition(repo, 'a', text_for(data.relative), {})
    if fault == 'missing-shard':
        data.paths[1].unlink()
    elif fault == 'extra-file':
        (data.paths[0].parent / 'notes').write_text('keep')
    elif fault == 'weight-symlink':
        outside = tmp_path / 'outside'
        outside.write_bytes(b'keep')
        data.paths[1].unlink()
        data.paths[1].symlink_to(outside)
    elif fault == 'bad-header':
        data.paths[1].write_bytes(b'broken GGUF')
    else:
        data.paths[0].parent.rename(data.paths[0].parent.with_name('not-a-sha256'))
    old = before(repo.models_dir)
    report = check_repository(repo.repository)
    assert not report['ok'] and report['summary']['errors']
    assert before(repo.models_dir) == old


def test_bad_definition_blocks_orphan_classification_but_does_not_hide_other_findings(repo):
    data = repo.test_publish()
    (repo.local_models_dir / 'bad.toml').write_text('broken toml')
    (repo.models_dir / 'data/unknown').mkdir()
    report = check_repository(repo.repository)
    assert not report['references_complete'] and not report['complete']
    assert report['groups'][0]['usage'] == 'unknown'
    assert {item['code'] for item in report['issues']} >= {'definition_invalid', 'data_entry_invalid'}
    assert data.paths[0].is_file()


@pytest.mark.parametrize('kind', ['collision', 'wrong-depth', 'symlink', 'unreadable'])
def test_definition_inventory_errors_are_explicit(repo, tmp_path, monkeypatch, kind):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    if kind == 'collision':
        (repo.local_models_dir / 'A.toml').write_text(text_for(data.relative))
    elif kind == 'wrong-depth':
        nested = repo.local_models_dir / 'nested/a.toml'
        nested.parent.mkdir()
        nested.write_text(text_for(data.relative))
    elif kind == 'symlink':
        (repo.local_models_dir / 'other').symlink_to(tmp_path, target_is_directory=True)
    else:
        original = Path.iterdir
        def read(path):
            if path == repo.local_models_dir:
                raise PermissionError('blocked inventory')
            return original(path)
        monkeypatch.setattr(Path, 'iterdir', read)
    report = check_repository(repo.repository)
    assert not report['ok'] and not report['references_complete']
    assert report['summary']['errors']


def test_orphan_and_live_protection_are_distinguished_without_deleting_anything(repo):
    data, other = repo.test_publish(), repo.test_publish('qwen3')
    with Reservation(repo.repository, 'another reader', groups=[data.group]):
        report = check_repository(repo.repository)
        rows = {row['id']: row for row in report['groups']}
        assert rows[data.group]['usage'] == 'reserved' and rows[other.group]['usage'] == 'orphan'
        assert report['ok'] and report['summary']['warnings'] == 1
        assert data.paths[0].is_file() and other.paths[0].is_file()


def test_corrupt_live_pin_prevents_claiming_unreferenced_groups(repo):
    data = repo.test_publish()
    pin = Reservation(repo.repository, 'test', groups=[data.group]).acquire()
    try:
        os.lseek(pin._fd, 1, os.SEEK_SET)
        os.write(pin._fd, b'!')
        report = check_repository(repo.repository, hashes=True)
        assert not report['reservations_complete'] and not report['ok']
        assert report['groups'][0]['usage'] == 'unknown'
        assert not report['groups'][0]['hash_verified']
    finally:
        pin.close(retry_cleanup=False)


def test_dead_release_record_is_reported_without_parsing_or_removing_it(repo):
    path = repo.models_dir / '.tllama/pins' / ('release-' + '1' * 32 + '.json')
    path.write_bytes(b'incomplete after crash')
    report = check_repository(repo.repository)
    assert report['ok'] and report['reservations']['stale'] == [path.name]
    assert path.read_bytes() == b'incomplete after crash'


@pytest.mark.parametrize('fault', ['old-schema', 'bad-json', 'wrong-key', 'outside-path', 'missing-target'])
def test_regenerable_bad_caches_are_warnings_without_refresh_or_outside_access(repo, tmp_path, fault):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    cache = get_metadata_cache_path(repo.metadata_cache_dir, data.paths[0])
    value = json.loads(cache.read_text())
    if fault == 'old-schema':
        value['schema_version'] = 1
    elif fault == 'outside-path':
        value['model']['path'] = '../outside.gguf'
    elif fault == 'missing-target':
        value['model']['path'] = 'data/' + '0' * 64 + '/weights.gguf'
        target = repo.metadata_cache_dir / (sha256(value['model']['path'].encode()).hexdigest() + '.json')
        cache.rename(target)
        cache = target
    elif fault == 'wrong-key':
        new = cache.with_name('0' * 64 + '.json')
        cache.rename(new)
        cache = new
    cache.write_text('!invalid' if fault == 'bad-json' else json.dumps(value))
    old = before(repo.models_dir)
    report = check_repository(repo.repository)
    assert report['ok'] and any(item['code'] == 'cache_invalid' for item in report['issues'])
    assert before(repo.models_dir) == old


@pytest.mark.parametrize('hashes', [False, True])
def test_check_never_drains_pending_queue_even_when_full_hash_reservation_releases(repo, hashes):
    data, extra = repo.test_publish(), repo.test_publish('qwen3')
    create_definition(repo, 'a', text_for(data.relative), {})
    pin = Reservation(repo.repository, 'worker', groups=[data.group]).acquire()
    repo.delete_model_definition('a')
    pin.close(retry_cleanup=False)
    queue = (repo.models_dir / QUEUE).read_bytes()
    report = check_repository(repo.repository, hashes=hashes)
    assert report['pending_cleanup'] == [data.group]
    assert data.paths[0].is_file() and extra.paths[0].is_file()
    assert (repo.models_dir / QUEUE).read_bytes() == queue


@pytest.mark.parametrize('fault', ['absent', 'lock-missing', 'pins-missing', 'unknown-marker', 'invalid-queue'])
def test_check_does_not_initialize_or_repair_missing_unknown_control_state(repo, tmp_path, fault):
    store = repo.repository
    if fault == 'absent':
        store = RepositoryStore(tmp_path / 'never-created')
    elif fault == 'lock-missing':
        (store.root / '.tllama/repository.lock').unlink()
    elif fault == 'pins-missing':
        (store.root / '.tllama/pins').rmdir()
        repo.test_publish()
    elif fault == 'unknown-marker':
        (store.root / '.tllama/repository.json').write_text('{"version":999}')
    else:
        (store.root / QUEUE).write_text('unknown queue')
    old = before(store.root)
    report = check_repository(store, hashes=True)
    assert not report['ok']
    assert before(store.root) == old
    if fault == 'absent':
        assert not store.root.exists()


def test_hashing_releases_global_lock_but_protects_weights_from_rm(repo, monkeypatch):
    import tllama.repository_check as checker
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    original = checker._hash_file
    deleted = False
    def hash_file(path, expected, stop):
        nonlocal deleted
        if not deleted:
            # Distinct store/manager lock acquisition would time out if the
            # audit kept the global lock around its long byte reads.
            from tllama.backend import ModelManager
            from tllama.config import BackendConfig
            other = ModelManager(BackendConfig(models_dir=str(repo.models_dir)))
            result = other.delete_model_definition('a')
            assert result['cleanup']['deferred_groups'] == [data.group]
            assert data.paths[0].is_file()
            deleted = True
        return original(path, expected, stop)
    monkeypatch.setattr(checker, '_hash_file', hash_file)
    report = check_repository(repo.repository, hashes=True)
    assert not report['ok'] and not report['complete']  # Definition changed during long audit.
    assert data.paths[0].is_file()  # Audit release does not trigger deferred GC.
    assert not snapshot(repo.repository).active


def test_weight_change_after_hash_is_reported_as_unverified(repo, monkeypatch):
    import tllama.repository_check as checker
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    original = checker._hash_file
    def hash_file(path, expected, stop):
        result = original(path, expected, stop)
        path.write_bytes(path.read_bytes() + b'changed after reading')
        return result
    monkeypatch.setattr(checker, '_hash_file', hash_file)
    report = check_repository(repo.repository, hashes=True)
    assert not report['ok'] and not report['groups'][0]['hash_verified']
    assert any(item['code'] == 'repository_changed' for item in report['issues'])
