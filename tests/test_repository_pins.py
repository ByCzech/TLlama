"""Real OS locks protect active resources across stores and process failures."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from tllama.model_uploads import receive_blob, blob_path
from tllama.repository_pins import Reservation, ReservationError, snapshot, temporary_directory
from tllama.repository_store import RepositoryStore
from test_model_create import gguf_file


@contextmanager
def child(code, *args, pass_fds=()):
    source = str(Path(__file__).resolve().parents[1] / 'src')
    process = subprocess.Popen([sys.executable, '-c', code, *map(str, args)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={**os.environ, 'PYTHONPATH': os.pathsep.join([source, os.environ.get('PYTHONPATH', '')])},
        **({'pass_fds': pass_fds} if pass_fds else {}))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            reading = pool.submit(process.stdout.readline)
            try:
                ready = reading.result(timeout=10)
            except BaseException:
                process.kill()
                process.wait(timeout=5)
                raise
        if ready.strip() != 'ready':
            process.kill()
            _, error = process.communicate(timeout=5)
            pytest.fail(error)
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_live_pin_is_visible_to_an_independent_store_and_released_on_exit(tmp_path):
    store, other = RepositoryStore(tmp_path), RepositoryStore(tmp_path)
    group = 'a' * 64
    with Reservation(store, 'publication', groups=[group], paths=['.tllama/uploads']):
        state = snapshot(other, remove_stale=True)
        assert len(state.active) == 1 and not state.retained and not state.stale
        assert state.protects_group(group) and not state.protects_group('b' * 64)
        assert state.protects_path('.tllama/uploads/.upload-in-progress')
        assert not state.protects_path('.tllama/uploads-other/file')
    assert snapshot(other).active == ()
    assert list((tmp_path / '.tllama/pins').iterdir()) == []


@pytest.mark.parametrize('retain', [False, True])
def test_process_death_uses_lock_liveness_not_a_pid_or_timeout(tmp_path, retain):
    code = '''
import sys
from tllama.repository_store import RepositoryStore
from tllama.repository_pins import Reservation
pin = Reservation(RepositoryStore(sys.argv[1]), 'foreign', groups=['a' * 64],
                  retain_on_crash=sys.argv[2] == 'True').acquire()
print('ready', flush=True)
sys.stdin.read()
'''
    store = RepositoryStore(tmp_path)
    with child(code, tmp_path, retain) as process:
        assert snapshot(store).protects_group('a' * 64)
        process.kill()
        process.wait(timeout=5)
        state = snapshot(store)
        assert bool(state.retained) is retain
        assert state.protects_group('a' * 64) is retain
        assert bool(state.stale) is not retain
        assert len(list((tmp_path / '.tllama/pins').iterdir())) == 1  # Read-only snapshot.
        snapshot(store, remove_stale=True)
        assert bool(list((tmp_path / '.tllama/pins').iterdir())) is retain


@pytest.mark.skipif(os.name != 'posix', reason='POSIX inherited flock lifetime')
def test_child_keeps_pin_live_after_supervisor_descriptor_disappears(tmp_path):
    store = RepositoryStore(tmp_path)
    pin = Reservation(store, 'worker', groups=['a' * 64]).acquire()
    with child("import sys; print('ready', flush=True); sys.stdin.read()", pass_fds=pin.inherited_fds) as process:
        # Closing without LOCK_UN models the kernel closing the supervisor's
        # descriptors on death. The child still owns the same open description.
        os.close(pin._fd)
        pin._fd = None
        assert snapshot(store, remove_stale=True).protects_group('a' * 64)
        process.kill()
        process.wait(timeout=5)
        assert not snapshot(store, remove_stale=True).protects_group('a' * 64)
    assert not list((tmp_path / '.tllama/pins').iterdir())


def test_update_preserves_locked_inode_and_narrows_the_protection_set(tmp_path):
    store = RepositoryStore(tmp_path)
    with Reservation(store, 'resolve', all_data=True) as pin:
        inode, descriptor = pin.path.stat().st_ino, pin._fd
        assert snapshot(store).protects_group('b' * 64)
        pin.update(groups=['a' * 64, 'a' * 64], all_data=False)
        assert pin.path.stat().st_ino == inode and pin._fd == descriptor
        state = snapshot(store)
        assert state.protects_group('a' * 64) and not state.protects_group('b' * 64)
        assert state.active[0]['groups'] == ['a' * 64]


@pytest.mark.parametrize('change', ['unknown-version', 'boolean-version', 'float-version',
                                  'integer-retention', 'invalid-json', 'invalid-group'])
def test_unreadable_live_record_aborts_snapshot_instead_of_returning_partial_safety(tmp_path, change):
    store = RepositoryStore(tmp_path)
    with Reservation(store, 'publication', groups=['a' * 64]) as pin:
        document = json.loads(pin.path.read_text())
        if change == 'unknown-version':
            document['version'] = 999
        elif change == 'boolean-version':
            document['version'] = True
        elif change == 'float-version':
            document['version'] = 1.0
        elif change == 'integer-retention':
            document['retain_on_crash'] = 0
        elif change == 'invalid-group':
            document['groups'] = ['../outside']
        os.lseek(pin._fd, 1, os.SEEK_SET)
        os.ftruncate(pin._fd, 1)
        os.write(pin._fd, ('{broken' if change == 'invalid-json' else json.dumps(document)).encode())
        with pytest.raises(ReservationError, match='Cannot inspect reservation'):
            snapshot(store, remove_stale=True)
        assert pin.path.exists()


def test_truncated_record_from_dead_releasable_owner_can_be_removed(tmp_path):
    store = RepositoryStore(tmp_path)
    pin = Reservation(store, 'upload').acquire()
    os.lseek(pin._fd, 1, os.SEEK_SET)
    os.ftruncate(pin._fd, 1)
    os.write(pin._fd, b'{unfinished')
    os.close(pin._fd)
    pin._fd = None
    assert snapshot(store).stale == (pin.path.name,)
    snapshot(store, remove_stale=True)
    assert not pin.path.exists()


@pytest.mark.parametrize('entry', ['symlink-file', 'symlink-directory', 'unknown-file'])
def test_reservation_namespace_cannot_redirect_reads_or_cleanup(tmp_path, entry):
    store = RepositoryStore(tmp_path / 'models')
    outside = tmp_path / 'outside'
    outside.mkdir()
    victim = outside / 'retain.txt'
    victim.write_text('keep')
    store._directory('.tllama')
    if entry == 'symlink-directory':
        (store.root / '.tllama/pins').symlink_to(outside, target_is_directory=True)
    else:
        directory = store._directory('.tllama/pins')
        if entry == 'symlink-file':
            (directory / ('release-' + 'a' * 32 + '.json')).symlink_to(victim)
        else:
            (directory / 'unknown').write_text('unknown')
    with pytest.raises(ReservationError):
        snapshot(store, remove_stale=True)
    assert victim.read_text() == 'keep'


@pytest.mark.parametrize('kwargs', [
    {'groups': ['A' * 64]}, {'groups': ['../bad']}, {'all_data': 1},
    {'paths': ['/absolute']}, {'paths': ['.tllama/uploads/../file']},
    {'paths': ['.tllama//uploads/file']}, {'paths': ['Local/weights.gguf']},
    {'paths': ['.tllama/uploads\\outside']}, {'retain_on_crash': 1},
])
def test_invalid_reservations_do_not_touch_the_store(tmp_path, kwargs):
    root = tmp_path / 'absent'
    with pytest.raises(ValueError):
        Reservation(RepositoryStore(root), 'test', **kwargs)
    assert not root.exists()


def test_released_pin_is_idempotent_but_cannot_be_reopened(tmp_path):
    pin = Reservation(RepositoryStore(tmp_path), 'test').acquire()
    pin.close()
    pin.close()
    with pytest.raises(ReservationError):
        pin.acquire()


async def test_cancellation_during_acquisition_joins_then_releases(tmp_path, monkeypatch):
    store = RepositoryStore(tmp_path)
    pin = Reservation(store, 'upload', paths=['.tllama/uploads'])
    entered, proceed = threading.Event(), threading.Event()
    acquire = pin.acquire
    def delayed():
        entered.set()
        assert proceed.wait(timeout=5)
        return acquire()
    monkeypatch.setattr(pin, 'acquire', delayed)
    async def work():
        async with pin:
            raise AssertionError('Cancelled acquisition must not enter the body')
    task = asyncio.create_task(work())
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not snapshot(store).active
    assert not list((tmp_path / '.tllama/pins').iterdir())


def test_gc_transaction_blocks_a_new_pin_until_its_decision_finishes(tmp_path):
    store, writer = RepositoryStore(tmp_path), RepositoryStore(tmp_path)
    entered = threading.Event()
    def reserve():
        entered.set()
        return Reservation(writer, 'publication', groups=['a' * 64]).acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.lock():
            assert not snapshot(store).active
            pending = pool.submit(reserve)
            assert entered.wait(timeout=5)
            assert not pending.done()
        pin = pending.result(timeout=5)
        try:
            assert snapshot(store).protects_group('a' * 64)
        finally:
            pin.close()


def test_staging_job_is_reserved_until_cleanup_even_after_failure(tmp_path):
    store = RepositoryStore(tmp_path)
    with pytest.raises(OSError, match='network'):
        with temporary_directory(store, parent='.tllama/downloads', prefix='pull-', purpose='pull') as (job, pin):
            file = job / 'partial.gguf'
            file.write_bytes(b'partial')
            assert snapshot(store).protects_path(file.relative_to(store.root).as_posix())
            raise OSError('network')
    assert not job.exists() and not pin.path.exists()


async def test_streaming_upload_is_reserved_during_body_and_final_publication(manager, tmp_path):
    raw = gguf_file(tmp_path / 'input.gguf').read_bytes()
    digest = 'sha256:' + sha256(raw).hexdigest()
    async def receive():
        state = snapshot(RepositoryStore(manager.models_dir))
        assert [row['purpose'] for row in state.active] == ['upload']
        assert state.protects_path('.tllama/uploads/' + digest[7:])
        return {'type': 'http.request', 'body': raw, 'more_body': False}
    await receive_blob(manager, digest, Request({'type': 'http', 'headers': []}, receive))
    assert blob_path(manager, digest).read_bytes() == raw
    assert snapshot(manager.repository).active == ()


@pytest.mark.parametrize('failed', [False, True])
def test_pull_pins_staging_and_then_the_complete_group(manager, monkeypatch, failed):
    def download(*, local_dir, filename, **kwargs):
        state = snapshot(RepositoryStore(manager.models_dir))
        assert [row['purpose'] for row in state.active] == ['pull']
        assert state.protects_path((local_dir / filename).relative_to(manager.models_dir).as_posix())
        if failed:
            raise OSError('network')
        return str(gguf_file(local_dir / filename))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    publish = manager.repository.publish
    def checked(group, **kwargs):
        assert snapshot(manager.repository).protects_group(group.relative_paths[0].parts[1])
        return publish(group, **kwargs)
    monkeypatch.setattr(manager.repository, 'publish', checked)
    if failed:
        with pytest.raises(OSError, match='network'):
            manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    else:
        assert Path(manager._pull_hf_file_sync('ns/repo', 'model.gguf')).is_file()
    assert not snapshot(manager.repository).active
    assert not list((manager.models_dir / '.tllama/pins').iterdir())
