"""Packaging regression: every subprocess must use the selected wheel tree.

Run explicitly with pytest debian/tests/test_smoke_paths.py in a development
environment; pytest is not required by the Debian package build smoke itself.
"""

import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / 'debian/tests/smoke'

OLD_DAEMON = '''
import argparse
import json
from tllama.daemon_options import add_daemon_options
from tllama.daemon_config import load_daemon_config
from tllama.version import get_version
def build_parser():
    parser = argparse.ArgumentParser(prog='tllamad')
    parser.add_argument('--version', action='version', version='tllamad ' + get_version())
    add_daemon_options(parser)
    parser.add_subparsers(dest='command').add_parser('rebuildrepo')
    return parser
if __name__ == '__main__':
    args = build_parser().parse_args()
    config = load_daemon_config(args.config)
    if args.check_config:
        print('Configuration is valid.')
    elif args.print_config:
        print(json.dumps(config.report()))
'''


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


@pytest.mark.parametrize('mode', [py_compile.PycInvalidationMode.UNCHECKED_HASH,
                                 py_compile.PycInvalidationMode.TIMESTAMP])
def test_full_build_smoke_discards_old_bytecode_even_when_python_accepts_it(trees, tmp_path, mode):
    built, shadow = trees
    path = built / 'tllama/daemon.py'
    current, stamp = path.read_bytes(), path.stat().st_mtime_ns
    old = OLD_DAEMON.encode()
    assert len(old) < len(current)
    # The timestamp case reproduces a cache collision: same size and mtime,
    # different contents. Unchecked-hash caches can also accept old contents.
    path.write_bytes(old + b' ' * (len(current) - len(old)))
    os.utime(path, ns=(stamp, stamp))
    for optimize in (0, 1):
        py_compile.compile(str(path), doraise=True, optimize=optimize, invalidation_mode=mode)
    path.write_bytes(current)
    os.utime(path, ns=(stamp, stamp))
    assert 'rebuildrepo' not in path.read_text()
    probe = '''
import argparse
import tllama.daemon as daemon
print(sorted(next(action.choices for action in daemon.build_parser()._actions
                  if isinstance(action, argparse._SubParsersAction))))
'''
    env = {**os.environ, 'PYTHONPATH':str(built)}
    before = subprocess.run([sys.executable, '-c', probe], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert before.returncode == 0 and before.stdout.strip() == "['rebuildrepo']", before.stderr
    result = subprocess.run([sys.executable, str(SMOKE), '--expected-version', '1.4.0',
        '--package-root', str(built)], env=env, cwd=shadow,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TLlama package smoke passed: 1.4.0' in result.stdout
    assert path.read_bytes() == current
    # The unused optimization cache must not be carried to installation either.
    assert not path.with_name('__pycache__').joinpath(
        'daemon.' + sys.implementation.cache_tag + '.opt-1.pyc').exists()


def test_auto_detected_installed_mode_does_not_delete_bytecode(trees):
    built, shadow = trees
    sentinel = built / 'tllama/__pycache__/unrelated.pyc'
    sentinel.parent.mkdir()
    sentinel.write_bytes(b'installed cache must not be removed')
    result = subprocess.run([sys.executable, str(SMOKE), '--expected-version', '1.4.0'],
        env={**os.environ,'PYTHONPATH':str(built)}, cwd=shadow,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_bytes() == b'installed cache must not be removed'
