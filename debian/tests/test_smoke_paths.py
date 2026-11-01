"""Packaging regression: every subprocess must use the selected wheel tree.

Run explicitly with pytest debian/tests/test_smoke_paths.py in a development
environment; pytest is not required by the Debian package build smoke itself.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / 'debian/tests/smoke'


@pytest.fixture
def trees(tmp_path):
    built, shadow = tmp_path / 'wheel', tmp_path / 'old-cwd'
    shutil.copytree(REPO / 'src/tllama', built / 'tllama',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    info = built / 'tllama-1.4.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: tllama\nVersion: 1.4.0\n')
    (info / 'entry_points.txt').write_text('[console_scripts]\ntllama = tllama.cli:main\ntllamad = tllama.daemon:main\n')
    (shadow / 'tllama').mkdir(parents=True)
    (shadow / 'tllama/__init__.py').write_text("raise AssertionError('imported old CWD package')\n")
    return built, shadow


@pytest.mark.parametrize('explicit_root', [False, True])
def test_full_smoke_ignores_old_package_in_invocation_cwd(trees, explicit_root):
    built, shadow = trees
    args = ['--package-root', str(built)] if explicit_root else []
    result = subprocess.run([sys.executable, str(SMOKE), '--expected-version', '1.4.0', *args],
        env={**os.environ, 'PYTHONPATH':str(built)}, cwd=shadow,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TLlama package smoke passed: 1.4.0' in result.stdout


def test_explicit_root_wins_over_inherited_pythonpath(trees):
    built, shadow = trees
    result = subprocess.run([sys.executable, str(SMOKE), '--expected-version', '1.4.0',
        '--package-root', str(built)], env={**os.environ,'PYTHONPATH':str(shadow)},
        cwd=shadow, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_selected_package_cannot_fall_back_to_another_install(trees, tmp_path):
    built, shadow = trees
    result = subprocess.run([sys.executable, str(SMOKE), '--package-root', str(tmp_path/'absent')],
        env={**os.environ,'PYTHONPATH':str(built)}, cwd=shadow,
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Missing built TLlama package' in result.stderr


def test_metadata_must_belong_to_selected_code_tree(trees, tmp_path):
    built, shadow = trees
    code_only = tmp_path / 'code-only'
    shutil.copytree(built/'tllama', code_only/'tllama')
    result = subprocess.run([sys.executable, str(SMOKE), '--package-root', str(code_only)],
        env={**os.environ,'PYTHONPATH':str(built)}, cwd=shadow,
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'TLlama metadata/code roots differ' in result.stderr
