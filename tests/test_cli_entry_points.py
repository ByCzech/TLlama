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


def run_module(module, *args, models_dir):
    env = dict(os.environ)
    env["TLLAMA_MODELS"] = str(models_dir)
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
