import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_daemon_cli import environment


@pytest.mark.parametrize('exit_code', [0, 7])
def test_device_listing_is_raw_local_configured_and_preserves_status(tmp_path, exit_code):
    native = tmp_path / 'runtime'
    native.write_text('#!' + sys.executable + '\nimport os,sys\n'
                      'assert sys.argv[1:] == ["--list-devices"]\n'
                      'assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"\n'
                      'assert "LLAMA_ARG_MODEL" not in os.environ\n'
                      'sys.stdout.buffer.write(b"Available devices:\\n  CUDA0: test\\n")\n'
                      'sys.stderr.write("driver diagnostic\\n")\n'
                      f'sys.exit({exit_code})\n')
    native.chmod(0o755)
    store = tmp_path / 'models-not-created'
    config = tmp_path / 'server.toml'
    config.write_text(f'[engine]\nexecutable="{native}"\n[storage]\nmodels_dir="{store}"\n')
    env = environment(tmp_path)
    env.update(GGML_VK_VISIBLE_DEVICES='1', LLAMA_ARG_MODEL='must-not-load')
    code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn', 'gguf'} or fullname == 'tllama.backend':
            raise AssertionError('Imported application: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['tllamad', '--config', CONFIG, '--list-devices']
runpy.run_module('tllama.daemon', run_name='__main__')
'''.replace('CONFIG', repr(str(config)))
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    assert result.stdout == b'Available devices:\n  CUDA0: test\n'
    assert result.stderr == b'driver diagnostic\n'
    assert not store.exists()


def test_missing_runtime_fails_cleanly_and_modes_are_exclusive(tmp_path):
    base = [sys.executable, '-m', 'tllama.daemon', '--no-config', '--runtime-path', str(tmp_path / 'missing'), '--list-devices']
    result = subprocess.run(base, env=environment(tmp_path), capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and 'Device listing failed' in result.stderr and 'Traceback' not in result.stderr
    for suffix in [['--check-config'], ['--print-config'], ['rebuildrepo', '--dryrun']]:
        result = subprocess.run(base + suffix, env=environment(tmp_path), capture_output=True, text=True, timeout=10)
        assert result.returncode == 2 and 'Traceback' not in result.stderr
