"""The daemon checks only local files; no HTTP client or server is involved."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tllama.daemon import main, build_parser
from tllama.model_definitions import create_definition
from tllama.repository_check import CheckCancelled
from tllama.repository_pins import snapshot
from test_repository_cleanup import repo, text_for
from test_repository_check import before


@pytest.mark.parametrize('hashes', [False, True])
@pytest.mark.parametrize('as_json', [False, True])
def test_daemon_local_command_checks_selected_repository_without_serving(repo, capsys, hashes, as_json):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    args = ['repo', 'check', '--no-config', '--models-dir', str(repo.models_dir)]
    assert main(args + (['--hashes'] if hashes else []) + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    assert not output.err
    if as_json:
        report = json.loads(output.out)
        assert report['ok'] and report['hashes'] is hashes
        assert report['groups'][0]['hash_verified'] is hashes
    else:
        assert 'Repository check: OK' in output.out


@pytest.mark.parametrize('placement', ['before', 'after'])
def test_local_selection_flags_before_and_after_command_are_respected(repo, capsys, placement):
    flags = ['--no-config', '--models-dir', str(repo.models_dir)]
    args = flags + ['repo', 'check', '--json'] if placement == 'before' else ['repo', 'check', '--json'] + flags
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['ok']


def test_local_check_uses_server_toml_storage_selection(repo, tmp_path, capsys, monkeypatch):
    config = tmp_path / 'server.toml'
    config.write_text('[storage]\nmodels_dir=' + json.dumps(str(repo.models_dir)) + '\n')
    monkeypatch.delenv('TLLAMA_MODELS', raising=False)
    assert main(['repo', 'check', '--config', str(config), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['ok']


def test_explicit_local_path_wins_over_environment(repo, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('TLLAMA_MODELS', str(tmp_path / 'unselected'))
    assert main(['repo', 'check', '--no-config', '--models-dir', str(repo.models_dir), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['ok']


def test_absent_repo_is_reported_without_initializing_it(tmp_path, capsys):
    path = tmp_path / 'absent'
    assert main(['repo', 'check', '--no-config', '--models-dir', str(path), '--json']) == 1
    assert not json.loads(capsys.readouterr().out)['ok'] and not path.exists()


def test_corrupt_repo_exit_status_is_one_without_repairs(repo, capsys):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    data.paths[0].unlink()
    old = before(repo.models_dir)
    assert main(['repo', 'check', '--no-config', '--models-dir', str(repo.models_dir), '--json']) == 1
    assert not json.loads(capsys.readouterr().out)['ok']
    assert before(repo.models_dir) == old


def test_local_command_does_not_import_or_open_http_or_start_native(repo, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
import importlib.abc
import json
import sys
class NoServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'tllama.backend', 'tllama.main', 'tllama.client', 'uvicorn', 'httpx', 'fastapi'}:
            raise AssertionError('Unexpected HTTP/server import: ' + fullname)
sys.meta_path.insert(0, NoServer())
from tllama.daemon import main
raise SystemExit(main(['repo', 'check', '--no-config', '--models-dir', sys.argv[1], '--hashes', '--json']))
'''
    environment = {**os.environ, 'PYTHONPATH': str(source)}
    result = subprocess.run([sys.executable, '-c', code, str(repo.models_dir)], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ok']


def test_ctrl_c_is_cooperative_and_releases_hash_reservation(repo, monkeypatch, capsys):
    import tllama.repository_check as checker
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    previous = signal.getsignal(signal.SIGINT)
    def hash_file(path, expected, stop):
        assert snapshot(repo.repository).protects_group(data.group)
        signal.raise_signal(signal.SIGINT)
        assert stop.is_set()
        raise CheckCancelled()
    monkeypatch.setattr(checker, '_hash_file', hash_file)
    assert main(['repo', 'check', '--no-config', '--models-dir', str(repo.models_dir), '--hashes', '--json']) == 130
    assert signal.getsignal(signal.SIGINT) == previous
    assert not snapshot(repo.repository).active and data.paths[0].is_file()
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('flag', ['--check-config', '--print-config', '--list-devices'])
def test_inspection_modes_cannot_be_silently_combined(flag):
    with pytest.raises(SystemExit) as exc:
        main([flag, 'repo', 'check'])
    assert exc.value.code == 2


def test_config_selection_conflicts_across_parser_levels_are_rejected():
    with pytest.raises(SystemExit) as exc:
        main(['--no-config', 'repo', 'check', '--config', '/unused.toml'])
    assert exc.value.code == 2


def test_client_and_public_http_routes_match_the_pre_check_patch():
    from tllama.client import TLlamaClient
    from tllama.cli import build_parser as client_parser
    from tllama.main import app
    assert not hasattr(TLlamaClient, 'repo_check')
    with pytest.raises(SystemExit):
        client_parser().parse_args(['repo', 'check'])
    assert not any(route.path.startswith('/api/repo') for route in app.routes)
    assert build_parser().parse_args(['repo', 'check']).command == 'repo'
