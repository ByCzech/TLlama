"""Application and repository commands work when Python inference is unavailable."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.config import ConfigError
from tllama.native_config import resolve_worker_config
from test_native_manager import native_manager


BLOCK_BINDING = '''
import importlib.abc
import sys
class RefuseBinding(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'llama_cpp' or fullname.startswith('llama_cpp.'):
            raise AssertionError('Python inference binding was imported: ' + fullname)
sys.meta_path.insert(0, RefuseBinding())
'''


def run_without_binding(code, tmp_path, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith('TLLAMA_')}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'),
               TLLAMA_MODELS=str(tmp_path / 'models'), **overrides)
    return subprocess.run([sys.executable, '-c', BLOCK_BINDING + code], env=env,
                          capture_output=True, text=True, timeout=15)


def test_application_import_and_offline_endpoints_need_no_binding_or_worker(tmp_path):
    result = run_without_binding('''
from tllama.main import app
from tllama.backend import model_manager
from fastapi.testclient import TestClient
import gguf
assert not model_manager.models_dir.exists()
model_manager.ensure_storage()
path = model_manager.local_models_dir / 'a.gguf'
writer = gguf.GGUFWriter(path=str(path), arch='llama')
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.close()
(model_manager.local_models_dir / 'a.toml').write_text('[llm]\\nmodel="Local/a.gguf"\\n')
with TestClient(app) as client:
    assert client.get('/api/tags').json()['models'][0]['name'] == 'a'
    assert client.post('/api/show', json={'model':'a'}).json()['capabilities'] == ['completion']
    assert client.get('/api/ps').json() == {'models':[]}
    assert model_manager.native_snapshot() == []
assert 'llama_cpp' not in sys.modules
''', tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('module', ['tllama.daemon'])
def test_repository_command_runs_with_binding_imports_blocked(tmp_path, module):
    result = run_without_binding(f'''
import runpy
sys.argv = ['tllama', 'rebuildrepo', '--dryrun']
runpy.run_module({module!r}, run_name='__main__')
''', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'Nothing was written.' in result.stdout


@pytest.mark.parametrize('field,variable', [('kv_cache_type','TLLAMA_KV_CACHE_TYPE'),
    ('k_cache_type','TLLAMA_K_CACHE_TYPE'), ('v_cache_type','TLLAMA_V_CACHE_TYPE')])
def test_each_global_cache_type_is_rejected_at_construction(make_manager, field, variable):
    with pytest.raises(ConfigError, match=variable):
        make_manager(**{field:'q9_9'})


@pytest.mark.parametrize('device', ['none','Vulkan0','CUDA0','ROCm0'])
def test_manager_keeps_device_configuration_for_the_native_resolver(make_manager, device):
    manager = make_manager(runtime_overrides={'device':device})
    resolved = resolve_worker_config(manager.config, Path('/fake.gguf'))
    assert resolved.launch.device == device


async def test_pinned_generation_survives_expiry_sweep_until_explicit_unload(native_manager):
    manager = native_manager(janitor_interval_seconds=0.01)
    name, _ = manager.test.define()
    clock = [100.0]
    manager._native._clock = lambda: clock[0]
    generation = await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    clock[0] += 1_000_000
    await manager._native.expire()
    assert worker.process.returncode is None
    assert manager.native_snapshot()[0]['keep_alive'] is None
    await manager.unload_native_model(name)
    await manager.wait_native_closed(generation)
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()
