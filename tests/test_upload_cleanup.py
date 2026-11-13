"""Explicit offline upload cleanup, daemon exclusion, preview and retry."""

from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import subprocess
import sys

import pytest

from tllama.daemon import main
from tllama.helpers.metadata_cache import (get_digest_cache_path,
    save_digest_cache, save_metadata_cache)
from tllama.helpers.repository_files import repository_tempfile
from tllama.model_definitions import create_definition
from tllama.model_uploads import blob_path
from tllama.repository_pins import Reservation, snapshot
from tllama.repository_store import RepositoryStore, prepare_group
from tllama.repository_upload_cleanup import prune_uploads
from test_model_create import gguf_file
from test_repository_check import before
from test_repository_pins import child


@pytest.fixture
def uploads(manager, tmp_path):
    weight, = manager.repository.publish(prepare_group([gguf_file(tmp_path / 'main.gguf')]))
    create_definition(manager, 'model', '[llm]\nmodel=' + json.dumps(
        weight.relative_to(manager.models_dir).as_posix()) + '\n', {})
    root = manager.repository._directory('.tllama/uploads')
    raw = gguf_file(tmp_path / 'pending.gguf', arch='qwen3').read_bytes()
    pending = blob_path(manager, 'sha256:' + sha256(raw).hexdigest())
    pending.write_bytes(raw)
    for save in (save_digest_cache, save_metadata_cache):
        save(manager.metadata_cache_dir, 'pending', pending,
             {'content_sha256': pending.name, 'source': 'local'})
    alias = root / sha256(weight.read_bytes()).hexdigest()
    os.link(weight, alias)  # A legacy alias, created after the modern create.
    save_digest_cache(manager.metadata_cache_dir, 'alias', alias,
                      {'content_sha256': alias.name, 'source': 'local'})
    fd, partial = repository_tempfile(directory=root, prefix='.upload-')
    with os.fdopen(fd, 'wb') as stream:
        stream.write(b'incomplete upload')
    return manager, weight, pending, alias, partial


def test_preview_is_byte_for_byte_read_only_and_apply_preserves_all_published_files(uploads):
    manager, weight, pending, alias, partial = uploads
    old = before(manager.models_dir)
    preview = prune_uploads(manager.repository)
    assert preview['ok'] and preview['complete'] and preview['can_apply']
    assert preview['summary']['blobs'] == 2 and preview['summary']['partials'] == 1
    assert before(manager.models_dir) == old
    report = prune_uploads(manager.repository, apply=True)
    assert report['ok'] and report['removed_files']
    assert not any(p.exists() for p in (pending, alias, partial))
    after = before(manager.models_dir)
    expected = {k: v for k, v in old.items() if not k.startswith('.tllama/uploads/')
                and k not in report['removed_files'] and k not in ('.tllama/uploads', '.tllama/metadata-cache')}
    assert {k: v for k, v in after.items() if k not in ('.tllama/uploads', '.tllama/metadata-cache')} == expected
    assert weight.stat().st_nlink == 1 and manager.read_model_definition('model')
    assert prune_uploads(manager.repository, apply=True)['candidates'] == []


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('purpose', ['daemon', 'upload', 'worker', 'retained'])
def test_active_or_retained_owner_refuses_offline_cleanup(uploads, apply, purpose):
    manager, weight, pending, alias, partial = uploads
    pin = Reservation(manager.repository, purpose, retain_on_crash=purpose == 'retained').acquire()
    try:
        if purpose == 'retained':
            os.close(pin._fd)
            pin._fd = None
        old = before(manager.models_dir)
        report = prune_uploads(manager.repository, apply=apply)
        assert not report['ok'] and not report['can_apply'] and not report['removed_files']
        assert 'Stop all TLlama daemons' in report['issues'][0]['message']
        assert before(manager.models_dir) == old
    finally:
        if pin._fd is not None:
            pin.close(retry_cleanup=False)
        else:
            pin.path.unlink()


async def test_real_application_lifetime_marks_even_idle_daemon_and_releases_after_shutdown(uploads, monkeypatch):
    from tllama import main as application
    manager, weight, pending, alias, partial = uploads
    monkeypatch.setattr(application, 'model_manager', manager)
    async with application.lifespan(application.app):
        assert any(p['purpose'] == 'daemon' for p in snapshot(manager.repository).active)
        assert not prune_uploads(manager.repository, apply=True)['ok']
        assert pending.exists()
    assert not snapshot(manager.repository).active
    assert manager._janitor_task is None
    assert prune_uploads(manager.repository, apply=True)['ok']


async def test_lifetime_marker_is_released_even_when_shutdown_reports_error(manager, monkeypatch):
    from tllama import main as application
    monkeypatch.setattr(application, 'model_manager', manager)
    shutdown = manager.shutdown
    async def fail():
        await shutdown()
        raise RuntimeError('shutdown evidence')
    monkeypatch.setattr(manager, 'shutdown', fail)
    with pytest.raises(RuntimeError, match='shutdown evidence'):
        async with application.lifespan(application.app):
            pass
    assert not snapshot(manager.repository).active


async def test_failed_lifetime_marker_acquisition_still_shuts_down_manager(manager, monkeypatch):
    from tllama import main as application
    monkeypatch.setattr(application, 'model_manager', manager)
    def fail(pin):
        raise OSError('cannot acquire daemon marker')
    monkeypatch.setattr(Reservation, 'acquire', fail)
    with pytest.raises(OSError, match='daemon marker'):
        async with application.lifespan(application.app):
            pytest.fail('Startup must not admit requests without its lifetime marker')
    assert manager._janitor_task is None and manager._native.closed


def test_idle_application_in_other_process_blocks_cleanup(uploads):
    manager, *_ = uploads
    code = '''
import asyncio, sys
from tllama import main
from tllama.backend import ModelManager
from tllama.config import BackendConfig
main.model_manager = ModelManager(BackendConfig(models_dir=sys.argv[1]))
async def run():
    async with main.lifespan(main.app):
        print('ready', flush=True)
        await asyncio.to_thread(sys.stdin.read)
asyncio.run(run())
'''
    with child(code, manager.models_dir):
        assert not prune_uploads(manager.repository, apply=True)['ok']
    assert prune_uploads(manager.repository, apply=True)['ok']


@pytest.mark.parametrize('fault', ['unknown', 'symlink', 'directory', 'cache-symlink'])
@pytest.mark.parametrize('apply', [False, True])
def test_unknown_entries_or_links_abort_inventory_before_any_unlink(uploads, tmp_path, fault, apply):
    manager, weight, pending, alias, partial = uploads
    outside = tmp_path / 'outside'
    outside.write_bytes(b'keep')
    if fault == 'unknown':
        (pending.parent / 'notes.txt').write_text('keep')
    elif fault == 'symlink':
        partial.unlink()
        partial.symlink_to(outside)
    elif fault == 'directory':
        (pending.parent / ('0' * 64)).mkdir()
    else:
        cache = get_digest_cache_path(manager.metadata_cache_dir, pending)
        cache.unlink()
        cache.symlink_to(outside)
    old = before(manager.models_dir)
    report = prune_uploads(manager.repository, apply=apply)
    assert not report['ok'] and not report['removed_files']
    assert before(manager.models_dir) == old and outside.read_bytes() == b'keep'


def test_apply_rebuilds_inventory_after_preview(uploads):
    manager, weight, pending, alias, partial = uploads
    assert len(prune_uploads(manager.repository)['candidates']) == 3
    extra = pending.parent / ('0' * 64)
    extra.write_bytes(b'another explicitly abandoned upload')
    report = prune_uploads(manager.repository, apply=True)
    assert report['ok'] and len(report['candidates']) == 4 and not extra.exists()


def test_partial_unlink_failure_reports_progress_and_can_be_retried(uploads, monkeypatch):
    manager, weight, pending, alias, partial = uploads
    unlink = Path.unlink
    def denied(path, *args, **kwargs):
        if path == pending:
            raise PermissionError('test unlink denial')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    report = prune_uploads(manager.repository, apply=True)
    assert not report['ok'] and pending.exists() and report['removed_files']
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert prune_uploads(manager.repository, apply=True)['ok']
    assert weight.exists() and manager.read_model_definition('model')


def test_cache_unlink_failure_leaves_blob_key_for_complete_retry(uploads, monkeypatch):
    from tllama.repository_check import check_repository
    manager, weight, pending, alias, partial = uploads
    cache = get_digest_cache_path(manager.metadata_cache_dir, pending)
    unlink = Path.unlink
    def denied(path, *args, **kwargs):
        if path == cache:
            raise PermissionError('cache unlink denial')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    assert not prune_uploads(manager.repository, apply=True)['ok'] and pending.exists()
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert prune_uploads(manager.repository, apply=True)['ok']
    assert not pending.exists() and not cache.exists()
    assert check_repository(manager.repository)['ok']


def test_cancel_before_inventory_changes_nothing(uploads):
    manager, *_ = uploads
    old = before(manager.models_dir)
    stop = threading.Event()
    stop.set()
    report = prune_uploads(manager.repository, apply=True, stop=stop)
    assert report['cancelled'] and not report['ok'] and not report['removed_files']
    assert before(manager.models_dir) == old


@pytest.mark.parametrize('apply', [False, True])
def test_missing_or_legacy_repository_is_not_initialized_by_cleanup(tmp_path, apply):
    root = tmp_path / 'absent'
    report = prune_uploads(RepositoryStore(root), apply=apply)
    assert not report['ok'] and not root.exists()


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('as_json', [False, True])
def test_local_cli_preview_and_apply(uploads, capsys, apply, as_json):
    manager, weight, pending, alias, partial = uploads
    args = ['repo', 'prune-uploads', '--no-config', '--models-dir', str(manager.models_dir)]
    assert main(args + (['--apply'] if apply else []) + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    assert not output.err
    if as_json:
        assert json.loads(output.out)['apply'] is apply
    else:
        assert ('APPLY' if apply else 'PREVIEW') in output.out and 'Published data' in output.out
    assert pending.exists() is not apply and weight.exists()


def test_configured_repository_needs_no_models_dir_switch(uploads, tmp_path, capsys, monkeypatch):
    manager, *_ = uploads
    config = tmp_path / 'server.toml'
    config.write_text('[storage]\nmodels_dir=' + json.dumps(str(manager.models_dir)) + '\n')
    monkeypatch.delenv('TLLAMA_MODELS', raising=False)
    assert main(['repo', 'prune-uploads', '--config', str(config), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['summary']['blobs'] == 2


def test_local_command_never_imports_http_or_native_application(uploads):
    manager, weight, pending, alias, partial = uploads
    code = '''
import importlib.abc, sys
class LocalOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'tllama.backend', 'tllama.main', 'tllama.client', 'httpx', 'fastapi', 'uvicorn'}:
            raise AssertionError('Unexpected HTTP/server import: ' + fullname)
sys.meta_path.insert(0, LocalOnly())
from tllama.daemon import main
raise SystemExit(main(['repo', 'prune-uploads', '--no-config', '--models-dir', sys.argv[1], '--apply', '--json']))
'''
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable, '-c', code, str(manager.models_dir)],
        env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['summary']['blobs'] == 2
    assert weight.exists() and not pending.exists()


def test_terminal_interrupt_reports_partial_progress_and_restores_signal_handler(uploads, monkeypatch, capsys):
    import signal
    manager, weight, pending, alias, partial = uploads
    original = Path.unlink
    def interrupt(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == partial:
            signal.raise_signal(signal.SIGINT)
        return result
    previous = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(Path, 'unlink', interrupt)
    assert main(['repo', 'prune-uploads', '--no-config', '--models-dir', str(manager.models_dir),
                 '--apply', '--json']) == 130
    report = json.loads(capsys.readouterr().out)
    assert report['cancelled'] and report['removed_files'] == [partial.relative_to(manager.models_dir).as_posix()]
    assert weight.exists() and pending.exists() and signal.getsignal(signal.SIGINT) == previous
