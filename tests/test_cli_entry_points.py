"""Canonical client and daemon entry points must work from a checkout."""

import os
import subprocess
import sys

import pytest


REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run_module(module, *args, models_dir, extra_env=None):
    env = dict(os.environ)
    env["TLLAMA_MODELS"] = str(models_dir)
    env.update(extra_env or {})
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize("module", ["tllama.daemon"])
def test_the_command_runs_from_a_checkout(module, tmp_path):
    """Repository maintenance works through the daemon entry point."""
    result = run_module(module, "rebuildrepo", "--dryrun", models_dir=tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["tllama.daemon"])
def test_nothing_is_raised_before_the_command_is_reached(module, tmp_path):
    """A NameError at import time is invisible to every test that imports
    the module normally, so it is asserted on directly."""
    result = run_module(module, "rebuildrepo", "--dryrun", models_dir=tmp_path)

    assert "NameError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_help_does_not_need_a_model_store(tmp_path):
    result = run_module("tllama", "--help", models_dir=tmp_path)

    assert result.returncode == 0
    assert "HTTP client" in result.stdout


def test_a_bad_environment_variable_stops_the_command_with_a_message(tmp_path):
    """The one place an operator sees a configuration mistake.

    A rejected value has to arrive as a line naming the variable and a
    non-zero exit, not as a traceback -- and not, as before this, as a
    server that comes up having quietly ignored the setting.
    """
    result = run_module(
        "tllama.daemon",
        "rebuildrepo",
        "--dryrun",
        models_dir=tmp_path,
        extra_env={"TLLAMA_MAX_LOADED_MODELS": "lots"},
    )

    assert result.returncode != 0
    assert "TLLAMA_MAX_LOADED_MODELS" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_kv_cache_type_no_ggml_type_matches_stops_the_command(tmp_path):
    """The native KV contract is checked before a command starts work."""
    result = run_module(
        "tllama.daemon",
        "rebuildrepo",
        "--dryrun",
        models_dir=tmp_path,
        extra_env={"TLLAMA_KV_CACHE_TYPE": "q9_9"},
    )

    assert result.returncode != 0
    assert "TLLAMA_KV_CACHE_TYPE" in result.stderr
    assert "Traceback" not in result.stderr


def test_overwrite_backups_without_force_is_refused(tmp_path):
    """A destructive flag that quietly does nothing is the fault it fixes.

    --overwrite-backups says an existing .toml.bak may be replaced. There
    is nothing to replace unless --force is also replacing definitions,
    so accepting it alone would mean somebody typed the dangerous option,
    saw a clean run, and learnt nothing about why nothing happened.
    """
    result = run_module(
        "tllama.daemon", "rebuildrepo", "--overwrite-backups", models_dir=tmp_path
    )

    assert result.returncode == 2
    assert "--force" in result.stderr


def test_overwrite_backups_with_force_is_accepted(tmp_path):
    result = run_module(
        "tllama.daemon",
        "rebuildrepo",
        "--force",
        "--overwrite-backups",
        "--dryrun",
        models_dir=tmp_path,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["tllama", "tllama.cli", "tllama.daemon"])
@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_lightweight_entry_points_do_not_import_server_dependencies(module, argument, tmp_path):
    code = '''
import importlib.abc
import runpy
import sys
class RefuseServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn', 'gguf', 'llama_cpp'} or fullname in {
            'tllama.main', 'tllama.backend', 'tllama.config', 'tllama.worker'}:
            raise AssertionError('Server dependency imported: ' + fullname)
sys.meta_path.insert(0, RefuseServer())
sys.argv = ['command', ARGUMENT]
runpy.run_module(MODULE, run_name='__main__')
'''.replace('ARGUMENT', repr(argument)).replace('MODULE', repr(module))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': REPO_SRC}, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('module', ['tllama', 'tllama.cli'])
def test_bare_client_shows_help_without_server_configuration(module, tmp_path):
    models = tmp_path / 'absent'
    result = run_module(module, models_dir=models,
                        extra_env={'TLLAMA_MAX_LOADED_MODELS': 'invalid'})
    assert result.returncode == 0, result.stderr
    assert 'HTTP client' in result.stdout
    assert not models.exists()


@pytest.mark.parametrize('command', ['serve', 'start', 'rebuildrepo'])
def test_client_does_not_launch_or_maintain_a_local_server(command, tmp_path):
    result = run_module('tllama', command, models_dir=tmp_path / 'absent')
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_installed_commands_target_separate_lightweight_entry_points():
    from pathlib import Path
    import tomlkit
    scripts = tomlkit.parse((Path(REPO_SRC).parent / 'pyproject.toml').read_text())['project']['scripts']
    assert scripts == {'tllama': 'tllama.cli:main', 'tllamad': 'tllama.daemon:main'}
