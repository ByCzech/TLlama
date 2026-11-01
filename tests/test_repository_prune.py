"""Preview does not write; apply rechecks ownership while locked through unlink."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from tllama.daemon import main
from tllama.helpers.metadata_cache import get_metadata_cache_path
from tllama.model_definitions import create_definition
from tllama.repository_cleanup import QUEUE, collect_pending
from tllama.repository_pins import Reservation
from tllama.repository_prune import prune_repository
from tllama.repository_store import RepositoryStore
from test_repository_cleanup import repo, text_for
from test_repository_check import before
from test_repository_pins import child


def test_default_preview_lists_exact_paths_and_does_not_touch_any_repository_entry(repo):
    data = repo.test_publish(count=2, upload=True)
    old = before(repo.models_dir)
    report = prune_repository(repo.repository)
    assert report['ok'] and report['can_apply'] and not report['apply']
    assert report['candidates'][0]['group'] == data.group
    expected = {path.relative_to(repo.models_dir).as_posix() for path in [*data.paths, *(path.with_suffix('.inf') for path in data.paths), *data.blobs, *data.caches]
                if path.exists()}
    assert set(report['candidates'][0]['remove_paths']) == expected
    assert report['removed_files'] == [] and report['removed_groups'] == []
    assert before(repo.models_dir) == old and not (repo.models_dir / QUEUE).exists()


def test_apply_removes_only_orphan_groups_and_their_caches_and_upload_aliases(repo):
    orphan, used = repo.test_publish(count=2, upload=True), repo.test_publish('qwen3')
    create_definition(repo, 'used', text_for(used.relative), {})
    report = prune_repository(repo.repository, apply=True)
    assert report['ok'] and report['removed_groups'] == [orphan.group]
    assert report['referenced_groups'] == [used.group]
    assert not any(path.exists() for path in [*orphan.paths, *orphan.blobs, *orphan.caches])
    assert all(path.exists() for path in [*used.paths, *used.caches])
    assert not orphan.paths[0].parent.exists() and not (repo.models_dir / QUEUE).exists()
    assert prune_repository(repo.repository, apply=True)['candidates'] == []


def test_references_to_projector_and_lora_protect_independent_groups(repo):
    llm, projector, lora, orphan = [repo.test_publish(arch) for arch in ('llama', 'clip', 'adapter', 'qwen3')]
    create_definition(repo, 'model', text_for(llm.relative, projector.relative, lora.relative), {})
    result = prune_repository(repo.repository, apply=True)
    assert result['removed_groups'] == [orphan.group]
    assert set(result['referenced_groups']) == {llm.group, projector.group, lora.group}
    assert all(item.paths[0].is_file() for item in (llm, projector, lora))


@pytest.mark.parametrize('scope', ['group', 'all-data', 'upload'])
def test_live_reservation_protects_groups_or_aliases_across_store_instances(repo, scope):
    data = repo.test_publish(upload=True)
    options = {'groups': [data.group]} if scope == 'group' else (
        {'all_data': True} if scope == 'all-data' else {'paths': ['.tllama/uploads']})
    with Reservation(RepositoryStore(repo.models_dir), 'other operation', **options):
        result = prune_repository(repo.repository, apply=True)
        assert result['ok'] and not result['removed_files'] and data.paths[0].is_file()
        assert data.group in result['reserved_groups' if scope != 'upload' else 'deferred_groups']
        assert not (repo.models_dir / QUEUE).exists()
    # Prune skips protected groups; release alone does not schedule new GC.
    assert data.paths[0].is_file()
    assert prune_repository(repo.repository, apply=True)['removed_groups'] == [data.group]


@pytest.mark.parametrize('protection', ['definition', 'reservation'])
def test_apply_uses_current_state_instead_of_a_previous_preview(repo, protection):
    data = repo.test_publish()
    preview = prune_repository(repo.repository)
    assert preview['candidates'][0]['group'] == data.group
    pin = None
    try:
        if protection == 'definition':
            create_definition(repo, 'new-owner', text_for(data.relative), {})
        else:
            pin = Reservation(RepositoryStore(repo.models_dir), 'new-reader', groups=[data.group]).acquire()
        result = prune_repository(repo.repository, apply=True)
        assert result['ok'] and not result['removed_files'] and data.paths[0].is_file()
    finally:
        if pin is not None:
            pin.close()


def test_other_process_pin_is_observed_before_unlink(repo):
    data = repo.test_publish()
    code = '''
import sys
from tllama.repository_store import RepositoryStore
from tllama.repository_pins import Reservation
pin = Reservation(RepositoryStore(sys.argv[1]), 'foreign', groups=[sys.argv[2]]).acquire()
print('ready', flush=True)
sys.stdin.read()
'''
    with child(code, repo.models_dir, data.group):
        report = prune_repository(repo.repository, apply=True)
        assert report['reserved_groups'] == [data.group] and data.paths[0].is_file()


@pytest.mark.parametrize('fault', ['bad-definition', 'wrong-depth', 'missing-group', 'bad-data-entry',
                                 'unknown-queue', 'bad-header', 'extra-group-file', 'symlink'])
def test_incomplete_or_invalid_inventory_blocks_apply_without_partial_deletion(repo, tmp_path, fault):
    orphan, used = repo.test_publish(), repo.test_publish('qwen3')
    if fault == 'bad-definition':
        (repo.local_models_dir / 'bad.toml').write_text('broken toml')
    elif fault == 'wrong-depth':
        target = repo.local_models_dir / 'wrong/bad.toml'
        target.parent.mkdir()
        target.write_text(text_for(used.relative))
    elif fault == 'missing-group':
        create_definition(repo, 'bad', text_for(used.relative), {})
        used.paths[0].unlink()
    elif fault == 'bad-data-entry':
        (repo.models_dir / 'data/unknown').mkdir()
    elif fault == 'unknown-queue':
        (repo.models_dir / QUEUE).write_text('{"version":999,"groups":{}}')
    elif fault == 'bad-header':
        used.paths[0].write_bytes(b'broken GGUF')
    elif fault == 'extra-group-file':
        (used.paths[0].parent / 'notes.txt').write_text('keep')
    else:
        (repo.local_models_dir / 'other').symlink_to(tmp_path, target_is_directory=True)
    old = before(repo.models_dir)
    result = prune_repository(repo.repository, apply=True)
    assert not result['ok'] and not result['can_apply'] and result['summary']['errors']
    assert not result['removed_files'] and orphan.paths[0].is_file()
    assert before(repo.models_dir) == old


@pytest.mark.parametrize('state', ['invalid-live', 'retained'])
def test_invalid_live_or_retained_metadata_cannot_authorize_pruning(repo, state):
    data = repo.test_publish()
    pin = Reservation(repo.repository, 'test', groups=[data.group], retain_on_crash=state == 'retained').acquire()
    try:
        if state == 'invalid-live':
            os.lseek(pin._fd, 1, os.SEEK_SET)
            os.write(pin._fd, b'!')
        else:
            os.close(pin._fd)
            pin._fd = None
        report = prune_repository(repo.repository, apply=True)
        assert not report['removed_files'] and data.paths[0].is_file()
        if state == 'invalid-live':
            assert not report['ok']
        else:
            assert report['ok'] and report['reserved_groups'] == [data.group]
    finally:
        if pin._fd is not None:
            pin.close(retry_cleanup=False)
        else:
            pin.path.unlink()


@pytest.mark.parametrize('where', ['cache', 'upload'])
def test_symlinked_alias_or_cache_blocks_plan_and_preserves_outside_file(repo, tmp_path, where):
    data = repo.test_publish(upload=True)
    outside = tmp_path / 'outside'
    outside.write_text('keep me')
    target = data.blobs[0] if where == 'upload' else get_metadata_cache_path(repo.metadata_cache_dir, data.paths[0])
    target.unlink()
    target.symlink_to(outside)
    result = prune_repository(repo.repository, apply=True)
    assert not result['ok'] and not result['removed_files']
    assert data.paths[0].is_file() and outside.read_text() == 'keep me'


def test_unrelated_staging_caches_and_stale_pin_records_remain_untouched(repo):
    data = repo.test_publish()
    staging = repo.repository._directory('.tllama/staging/old-job') / 'unknown'
    download = repo.repository._directory('.tllama/downloads/old-pull') / 'leftover'
    upload = repo.repository._directory('.tllama/uploads') / ('b' * 64)
    cache = repo.metadata_cache_dir / ('c' * 64 + '.json')
    pin = repo.models_dir / '.tllama/pins' / ('release-' + '1' * 32 + '.json')
    for path in (staging, download, upload, cache, pin):
        path.write_bytes(b'preserve unrelated bytes')
    result = prune_repository(repo.repository, apply=True)
    assert result['ok'] and result['removed_groups'] == [data.group]
    assert all(path.read_bytes() == b'preserve unrelated bytes' for path in (staging, download, upload, cache, pin))


def test_existing_pending_queue_is_read_only_in_preview(repo):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    pin = Reservation(repo.repository, 'worker', groups=[data.group]).acquire()
    repo.delete_model_definition('a')
    pin.close(retry_cleanup=False)
    old = before(repo.models_dir)
    result = prune_repository(repo.repository)
    assert result['ok'] and result['candidates'][0]['group'] == data.group
    assert before(repo.models_dir) == old


def test_schedule_is_durable_before_unlink_and_failures_keep_resumable_queue(repo, monkeypatch):
    data = repo.test_publish(count=2, upload=True)
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path in data.paths:
            queue = json.loads((repo.models_dir / QUEUE).read_text())
            assert queue['groups'][data.group] == [item.name for item in data.paths]
        if path == data.paths[1]:
            raise PermissionError('injected unlink failure')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        result = prune_repository(repo.repository, apply=True)
    assert not result['ok'] and result['summary']['errors']
    assert not data.paths[0].exists() and data.paths[1].is_file()
    assert data.paths[0].relative_to(repo.models_dir).as_posix() in result['removed_files']
    assert (repo.models_dir / QUEUE).is_file()
    assert collect_pending(repo.repository)['removed_groups'] == [data.group]


def test_failed_queue_write_prevents_every_unlink(repo, monkeypatch):
    data = repo.test_publish()
    old = before(repo.models_dir)
    def fail(*args):
        raise PermissionError('queue cannot be persisted')
    monkeypatch.setattr('tllama.repository_runtime.write_json', fail)
    report = prune_repository(repo.repository, apply=True)
    assert not report['ok'] and not report['removed_files'] and data.paths[0].is_file()
    assert before(repo.models_dir) == old


def test_global_lock_is_retained_from_final_checks_until_weight_unlink(repo, monkeypatch):
    data = repo.test_publish()
    other = RepositoryStore(repo.models_dir)
    entered, acquired = threading.Event(), threading.Event()
    original = Path.unlink
    def compete():
        assert entered.wait(5)
        with other.lock():
            acquired.set()
    def unlink(path, *args, **kwargs):
        if path == data.paths[0]:
            entered.set()
            assert not acquired.wait(0.1)
        return original(path, *args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(compete)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', unlink)
            assert prune_repository(repo.repository, apply=True)['ok']
        waiting.result(timeout=5)
    assert acquired.is_set()


@pytest.mark.parametrize('apply', [False, True])
def test_missing_store_is_not_initialized_even_for_apply(tmp_path, apply):
    store = RepositoryStore(tmp_path / 'missing')
    report = prune_repository(store, apply=apply)
    assert not report['ok'] and not store.root.exists()


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('as_json', [False, True])
def test_local_cli_defaults_to_preview_and_formats_report(repo, capsys, apply, as_json):
    data = repo.test_publish()
    args = ['repo', 'prune', '--no-config', '--models-dir', str(repo.models_dir)]
    assert main(args + (['--apply'] if apply else []) + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    assert not output.err
    if as_json:
        report = json.loads(output.out)
        assert report['apply'] is apply and report['candidates'][0]['group'] == data.group
    else:
        assert ('APPLY' if apply else 'PREVIEW') in output.out
        assert data.relative in output.out
    assert data.paths[0].exists() is not apply


@pytest.mark.parametrize('selection', ['before', 'after', 'toml', 'environment'])
def test_prune_uses_optional_models_dir_and_normal_configuration_precedence(repo, tmp_path, capsys,
                                                                          monkeypatch, selection):
    data = repo.test_publish()
    config = tmp_path / 'server.toml'
    config.write_text('[storage]\nmodels_dir=' + json.dumps(str(repo.models_dir)) + '\n')
    monkeypatch.delenv('TLLAMA_MODELS', raising=False)
    command = ['repo', 'prune', '--json']
    if selection in ('before', 'after'):
        monkeypatch.setenv('TLLAMA_MODELS', str(tmp_path / 'unselected'))
        flags = ['--no-config', '--models-dir', str(repo.models_dir)]
        command = flags + command if selection == 'before' else command + flags
    elif selection == 'toml':
        command += ['--config', str(config)]
    else:
        monkeypatch.setenv('TLLAMA_MODELS', str(repo.models_dir))
        command += ['--no-config']
    old = before(repo.models_dir)
    assert main(command) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['candidates'][0]['group'] == data.group
    assert before(repo.models_dir) == old


@pytest.mark.parametrize('apply', [False, True])
def test_unknown_repository_marker_blocks_prune_without_writes(repo, apply):
    repo.test_publish()
    (repo.models_dir / '.tllama/repository.json').write_text('{"version":999}')
    old = before(repo.models_dir)
    report = prune_repository(repo.repository, apply=apply)
    assert not report['ok'] and not report['removed_files']
    assert before(repo.models_dir) == old


def test_ctrl_c_finishes_current_group_and_leaves_others_queued(repo, capsys, monkeypatch):
    groups = sorted((repo.test_publish(count=2), repo.test_publish('qwen3', count=2)), key=lambda item: item.group)
    original = Path.unlink
    previous = signal.getsignal(signal.SIGINT)
    def unlink(path, *args, **kwargs):
        if path == groups[0].paths[0]:
            signal.raise_signal(signal.SIGINT)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert main(['repo', 'prune', '--no-config', '--models-dir', str(repo.models_dir), '--apply', '--json']) == 130
    report = json.loads(capsys.readouterr().out)
    assert report['cancelled'] and report['removed_groups'] == [groups[0].group]
    assert not any(path.exists() for path in groups[0].paths)
    assert groups[1].paths[0].is_file()
    assert signal.getsignal(signal.SIGINT) == previous
    assert set(json.loads((repo.models_dir / QUEUE).read_text())['groups']) == {groups[1].group}


def test_prune_command_never_imports_http_or_starts_native(repo):
    data = repo.test_publish()
    code = '''
import importlib.abc
import sys
class LocalOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'tllama.backend', 'tllama.main', 'tllama.client', 'httpx', 'fastapi', 'uvicorn'}:
            raise AssertionError('Unexpected HTTP/server import: ' + fullname)
sys.meta_path.insert(0, LocalOnly())
from tllama.daemon import main
raise SystemExit(main(['repo', 'prune', '--no-config', '--models-dir', sys.argv[1], '--apply', '--json']))
'''
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable, '-c', code, str(repo.models_dir)],
        env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['removed_groups'] == [data.group]
