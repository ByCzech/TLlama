"""Real daemon entry points, configuration inspection and HTTP startup."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

import pytest


SRC = str(Path(__file__).resolve().parents[1] / 'src')


def environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TLLAMA_', '_TLLAMA_'))}
    env.update(PYTHONPATH=SRC, XDG_CONFIG_HOME=str(tmp_path / 'config'))
    return env


@pytest.mark.parametrize('module', ['tllama.daemon', 'tllama.main'])
def test_inspection_avoids_application_imports_and_storage(module, tmp_path):
    file = tmp_path / 'server.toml'
    store = tmp_path / 'absent'
    file.write_text(f'[storage]\nmodels_dir="{store}"\n[server]\nport=1234\n')
    code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi','uvicorn','gguf','llama_cpp'} or fullname == 'tllama.backend':
            raise AssertionError('Imported application: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['tllamad', '--config', CONFIG, '--port', '2345', '--print-config']
runpy.run_module(MODULE, run_name='__main__')
'''.replace('CONFIG', repr(str(file))).replace('MODULE', repr(module))
    result = subprocess.run([sys.executable, '-c', code], env=environment(tmp_path),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['settings']['server.port'] == {'value': 2345, 'source': 'CLI'}
    assert not store.exists()


def test_bad_config_fails_cleanly_and_no_config_ignores_environment_selection(tmp_path):
    env = environment(tmp_path)
    env['TLLAMA_CONFIG'] = str(tmp_path / 'missing.toml')
    command = [sys.executable, '-m', 'tllama.daemon', '--check-config']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and 'missing.toml' in result.stderr
    assert 'Traceback' not in result.stderr
    result = subprocess.run(command + ['--no-config'], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and result.stdout == 'Configuration is valid.\n'


def test_maintenance_uses_daemon_configured_store(tmp_path):
    file = tmp_path / 'server.toml'
    store = tmp_path / 'configured'
    file.write_text(f'[storage]\nmodels_dir="{store}"\n')
    result = subprocess.run([sys.executable, '-m', 'tllama.daemon', '--config', str(file),
                             'rebuildrepo', '--dryrun'], env=environment(tmp_path),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert (store / 'Local').is_dir()


@pytest.mark.parametrize('reload', [False, True])
def test_real_daemon_uses_toml_cli_and_cors_before_import(tmp_path, reload):
    file = tmp_path / 'server.toml'
    store = tmp_path / 'configured'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    file.write_text(f'[server]\nport=1234\norigins=["https://test.example"]\n'
                    f'[storage]\nmodels_dir="{store}"\n')
    command = [sys.executable, '-m', 'tllama.daemon', '--config', str(file), '--port', str(port)]
    if reload:
        command += ['--reload']
    with (tmp_path / 'daemon.log').open('w+') as log:
        process = subprocess.Popen(command, env=environment(tmp_path), cwd=tmp_path, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    with urlopen(f'http://127.0.0.1:{port}/api/tags', timeout=0.3) as response:
                        assert json.load(response) == {'models': []}
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
            else:
                pytest.fail('Daemon did not become ready')
            log.seek(0)
            assert process.poll() is None, log.read()
            assert (store / 'Local').is_dir()
            from urllib.request import Request
            with urlopen(Request(f'http://127.0.0.1:{port}/health',
                                 headers={'Origin': 'https://test.example'}), timeout=2) as response:
                assert response.headers['Access-Control-Allow-Origin'] == 'https://test.example'
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
