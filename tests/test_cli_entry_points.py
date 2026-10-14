"""The command has to actually start.

An entry point is the one piece of code that unit tests reach around: the
functions it calls can all be correct while the module that exposes them
fails before any of them runs. That is exactly what happened -- main()
was defined below the `if __name__ == "__main__"` block that calls it, so
`python -m tllama.main` raised NameError while the installed console
script, which imports the module first and looks the name up afterwards,
worked and hid it.

So these run the real thing in a subprocess. rebuildrepo against an empty
directory is the cheapest subcommand that exercises the whole path and
exits on its own.
"""

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


@pytest.mark.parametrize("module", ["tllama", "tllama.main", "tllama.cli"])
def test_the_command_runs_from_a_checkout(module, tmp_path):
    """Three spellings a person reaches for during development. All of
    them have to work; one of them silently did not."""
    result = run_module(module, "rebuildrepo", "--dryrun", models_dir=tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["tllama", "tllama.main", "tllama.cli"])
def test_nothing_is_raised_before_the_command_is_reached(module, tmp_path):
    """A NameError at import time is invisible to every test that imports
    the module normally, so it is asserted on directly."""
    result = run_module(module, "rebuildrepo", "--dryrun", models_dir=tmp_path)

    assert "NameError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_help_does_not_need_a_model_store(tmp_path):
    result = run_module("tllama", "--help", models_dir=tmp_path)

    assert result.returncode == 0
    assert "rebuildrepo" in result.stdout


def test_a_bad_environment_variable_stops_the_command_with_a_message(tmp_path):
    """The one place an operator sees a configuration mistake.

    A rejected value has to arrive as a line naming the variable and a
    non-zero exit, not as a traceback -- and not, as before this, as a
    server that comes up having quietly ignored the setting.
    """
    result = run_module(
        "tllama",
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
        "tllama",
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
        "tllama.cli", "rebuildrepo", "--overwrite-backups", models_dir=tmp_path
    )

    assert result.returncode == 2
    assert "--force" in result.stderr


def test_overwrite_backups_with_force_is_accepted(tmp_path):
    result = run_module(
        "tllama.cli",
        "rebuildrepo",
        "--force",
        "--overwrite-backups",
        "--dryrun",
        models_dir=tmp_path,
    )

    assert result.returncode == 0, result.stderr
