"""Agreed lifecycle.device TOML/JSON contract, without probing hardware."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.daemon_config import load_daemon_config, daemon_configuration, current_daemon_config
from tllama.device_limits import device_limits, worker_devices
from tllama.residency import NativeResidency
from tllama.worker_config import WorkerLaunchSpec


ROWS = [{'name': 'Vulkan1', 'max_loaded_models': 2},
        {'name': 'ROCm0', 'max_loaded_models': 1},
        {'name': 'CPU', 'max_loaded_models': 0}, {'max_loaded_models': 5}]


def toml(rows):
    return ''.join('[[lifecycle.device]]\n' + ''.join(f'{k}={json.dumps(v)}\n' for k, v in row.items()) for row in rows)


@pytest.mark.parametrize('maximum,rows,effective', [(None, [], 1), (None, ROWS, 0),
    (0, [], 0), (0, ROWS, 0), (3, [], 3)])
def test_capacity_modes_are_identical_for_toml_json_and_direct_config(tmp_path, maximum, rows, effective):
    path = tmp_path/'server.toml'
    body = '' if maximum is None else f'[lifecycle]\nmax_loaded_models={maximum}\n'
    path.write_text(body + toml(rows))
    a = load_daemon_config(path, environ={}, discover=False)
    env = {'TLLAMA_DEVICE_LIMITS': json.dumps(rows)}
    if maximum is not None:
        env['TLLAMA_MAX_LOADED_MODELS'] = str(maximum)
    b = load_daemon_config(False, environ=env, discover=False)
    c = BackendConfig(max_loaded_models=maximum, device_limits=rows)
    assert a.backend.max_loaded_models == b.backend.max_loaded_models == c.max_loaded_models == effective
    assert a.backend.device_limits == b.backend.device_limits == c.device_limits == device_limits(rows)
    assert b.report()['settings']['lifecycle.device'] == {'value': rows, 'source': 'TLLAMA_DEVICE_LIMITS'}
    if maximum is None and rows:
        assert a.sources['lifecycle.max_loaded_models'] == 'implicit per-device mode'


@pytest.mark.parametrize('maximum', [1, 3, 100])
def test_any_explicit_positive_global_conflicts_without_sum_heuristics(tmp_path, maximum):
    path = tmp_path/'server.toml'
    path.write_text(f'[lifecycle]\nmax_loaded_models={maximum}\n' + toml(ROWS))
    for kwargs in [dict(config_path=path, environ={}), dict(config_path=False, environ={
            'TLLAMA_DEVICE_LIMITS': json.dumps(ROWS)}, overrides={'lifecycle.max_loaded_models': str(maximum)})]:
        with pytest.raises(ConfigError, match='Conflicting.*lifecycle.max_loaded_models.*lifecycle.device'):
            load_daemon_config(discover=False, **kwargs)
    with pytest.raises(ValueError, match='Conflicting'):
        NativeResidency(None, max_loaded_models=maximum, device_limits=ROWS)
    with pytest.raises(ConfigError, match='Conflicting'):
        BackendConfig(max_loaded_models=maximum, device_limits=ROWS)


def test_environment_replaces_whole_array_and_cli_resolves_cross_source_conflict(tmp_path):
    path = tmp_path/'server.toml'
    path.write_text('[lifecycle]\nmax_loaded_models=2\n' + toml(ROWS))
    config = load_daemon_config(path, environ={'TLLAMA_DEVICE_LIMITS': '[]'}, discover=False)
    assert config.backend.max_loaded_models == 2 and config.backend.device_limits == ()
    path.write_text('[lifecycle]\nmax_loaded_models=2\n')
    with pytest.raises(ConfigError, match='TOML.*TLLAMA_DEVICE_LIMITS'):
        load_daemon_config(path, environ={'TLLAMA_DEVICE_LIMITS': json.dumps(ROWS)}, discover=False)
    config = load_daemon_config(path, overrides={'lifecycle.max_loaded_models': '0'},
        environ={'TLLAMA_DEVICE_LIMITS': json.dumps(ROWS)}, discover=False)
    assert config.backend.max_loaded_models == 0 and config.sources['lifecycle.max_loaded_models'] == 'CLI'


@pytest.mark.parametrize('rows', [{}, 0, None, 'CPU=1', [None], [True], [{}], [{'name': 'CPU'}],
    [{'name': None, 'max_loaded_models': 1}], [{'name': '', 'max_loaded_models': 1}],
    [{'name': ' Vulkan0', 'max_loaded_models': 1}], [{'name': 'Vulkan0,Vulkan1', 'max_loaded_models': 1}],
    [{'name': 'none', 'max_loaded_models': 1}], [{'name': 'CPU\x00', 'max_loaded_models': 1}],
    [{'max_loaded_models': True}], [{'max_loaded_models': 1.0}], [{'max_loaded_models': -1}],
    [{'max_loaded_models': '1'}], [{'max_loaded_models': 1, 'bogus': 2}],
    [{'max_loaded_models': 1}, {'max_loaded_models': 2}],
    [{'name': 'CPU', 'max_loaded_models': 1}, {'name': 'CPU', 'max_loaded_models': 2}]])
def test_invalid_arrays_fail_with_environment_source(rows):
    with pytest.raises(ConfigError, match='TLLAMA_DEVICE_LIMITS'):
        load_daemon_config(False, environ={'TLLAMA_DEVICE_LIMITS': json.dumps(rows)}, discover=False)


def test_duplicate_json_keys_are_not_last_wins_and_limits_are_immutable():
    with pytest.raises(ConfigError, match='duplicate JSON key'):
        load_daemon_config(False, environ={'TLLAMA_DEVICE_LIMITS': '[{"max_loaded_models":1,"max_loaded_models":2}]'}, discover=False)
    config = BackendConfig(device_limits=ROWS)
    with pytest.raises(FrozenInstanceError):
        config.device_limits[0].max_loaded_models = 99


def test_cpu_buffers_and_inactive_helpers_do_not_consume_cpu_slot():
    base = WorkerLaunchSpec(model=Path('/main.gguf'), device='Vulkan0,Vulkan1,Vulkan0')
    assert worker_devices(base) == {'Vulkan0', 'Vulkan1'}
    helper = WorkerLaunchSpec(model=Path('/main.gguf'), device='Vulkan0', draft=Path('/draft.gguf'),
        runtime_options={'spec_type': 'draft-mtp', 'draft_device': 'none'})
    assert worker_devices(helper) == {'Vulkan0', 'CPU'}
    assert worker_devices(WorkerLaunchSpec(model=Path('/m'), device='Vulkan0', runtime_options={'draft_device': 'none'})) == {'Vulkan0'}


def test_reload_bridge_keeps_json_array_and_effective_implicit_mode(monkeypatch):
    monkeypatch.setenv('TLLAMA_DEVICE_LIMITS', json.dumps(ROWS))
    config = load_daemon_config(False, discover=False)
    with daemon_configuration(config, {}):
        assert current_daemon_config().backend == config.backend


@pytest.mark.parametrize('conflict', [False, True])
def test_daemon_check_and_print_never_probe_or_touch_storage(tmp_path, conflict):
    path = tmp_path/'server.toml'
    store = tmp_path/'absent'
    path.write_text(f'[storage]\nmodels_dir="{store}"\n[lifecycle]\nmax_loaded_models={1 if conflict else 0}\n'
                    + toml([{'name': 'UnavailableFutureGPU77', 'max_loaded_models': 2}]))
    env = {k:v for k,v in os.environ.items() if not k.startswith(('TLLAMA_', '_TLLAMA_'))}
    env.update(PYTHONPATH=str(Path(__file__).parents[1]/'src'))
    code = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in {'tllama.backend','tllama.automatic_worker'} or fullname.split('.')[0] in {'fastapi','uvicorn','gguf'}:
   raise AssertionError('Probed/imported application: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['tllamad','--config',sys.argv[1],'--print-config']
runpy.run_module('tllama.daemon',run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == (2 if conflict else 0), result.stderr
    assert not store.exists() and 'Traceback' not in result.stderr
    if not conflict:
        assert json.loads(result.stdout)['settings']['lifecycle.device']['value'][0]['name'] == 'UnavailableFutureGPU77'
