"""Daemon source precedence and validation without application side effects."""

import json
from pathlib import Path

import pytest

from tllama.config import ConfigError
from tllama.daemon_config import load_daemon_config, current_daemon_config, daemon_configuration
from tllama.native_config import resolve_worker_config, build_native_sampling
from tllama.helpers.model_toml import parse_model_toml


def load(path=None, overrides=None, **kwargs):
    return load_daemon_config(path, overrides, environ=kwargs.pop('environ', {}), discover=False, **kwargs)


def test_layers_reach_worker_but_model_still_overrides_global_device(tmp_path):
    file = tmp_path / 'server.toml'
    file.write_text('[server]\nhost="[::1]:12000"\n[storage]\nmodels_dir="models"\n'
                    '[runtime]\ndevice="ROCm0"\nn_threads=2\nflash_attn=true\n'
                    '[sampling]\ntemperature=0.4\n')
    config = load(file, {'runtime.device': 'Vulkan0', 'server.port': '14000'},
                  environ={'TLLAMA_HOST': '127.0.0.1:13000', 'TLLAMA_RUNTIME_DEVICE': 'CUDA0'})
    assert (config.app.host, config.app.port) == ('127.0.0.1', 14000)
    assert config.backend.runtime_overrides == {'device': 'Vulkan0', 'n_threads': 2, 'flash_attn': True}
    assert config.sources['runtime.device'] == 'CLI'
    assert config.sources['runtime.n_threads'].startswith('TOML ')
    resolved = resolve_worker_config(config.backend, Path('/models/m.gguf'))
    assert resolved.launch.device == 'Vulkan0'
    spec = parse_model_toml('[llm]\nmodel="Local/m.gguf"\n[runtime]\ndevice="ROCm0"\n')
    resolved = resolve_worker_config(config.backend, Path('/models/m.gguf'), spec)
    assert resolved.launch.device == 'ROCm0'
    assert build_native_sampling({}, resolved=resolved)['temperature'] == 0.4


def test_relative_storage_and_config_paths_use_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file = tmp_path / 'definitions' / 'server.toml'
    file.parent.mkdir()
    file.write_text('[storage]\nmodels_dir="models"\n')
    config = load('definitions/server.toml')
    assert config.backend.models_dir == str(tmp_path / 'models')
    assert config.config_path == file
    assert not (tmp_path / 'models').exists()


@pytest.mark.parametrize('body', ['unknown=1', '[unknown]\nx=1', '[server]\nprot=1234',
                                '[runtime]\nmodel_path="x"', '[runtime]\nn_threads="2"',
                                '[runtime]\nflash_attn=1', '[sampling]\ntop_k=true',
                                '[server]\nport=true', '[server]\nport=0',
                                '[server]\nport=65536', '[lifecycle]\njanitor_interval=0',
                                '[lifecycle]\nmax_loaded_models=0', '[cache]\nmetadata_ttl=nan',
                                '[runtime]\ntype_kv="q9"', '[runtime]\nn_ctx=-1',
                                '[lifecycle]\nkeep_alive="tomorrow"',
                                '[engine]\nexecutable="relative"',
                                '[server]\norigins=["https://example/path"]',
                                '[sampling]\nstop=["END"]', '[sampling]\nlogit_bias={"1"=1.0}'])
def test_invalid_toml_is_refused_before_serving(tmp_path, body):
    file = tmp_path / 'server.toml'
    file.write_text(body)
    with pytest.raises(ConfigError):
        load(file)


@pytest.mark.parametrize('name,value', [('TLLAMA_RUNTIME_N_CTX', '10'), ('TLLAMA_RUNTIME_BOGUS', '1'),
                                      ('TLLAMA_RUNTIME_N_THREADS', 'bad'), ('TLLAMA_SAMPLING_STOP', 'END'),
                                      ('TLLAMA_PORT', '0'), ('TLLAMA_JANITOR_INTERVAL', 'nan'),
                                      ('TLLAMA_RUNTIME_EXECUTABLE', 'relative')])
def test_bad_environment_retains_variable_name(name, value):
    with pytest.raises(ConfigError, match=name):
        load(environ={name: value})


def test_aliases_empty_values_and_cors_keep_existing_semantics():
    config = load(environ={'TLLAMA_HOST': '[::1]:1234', 'TLLAMA_PORT': '2345',
                          'TLLAMA_FLASH_ATTENTION': 'false', 'TLLAMA_RUNTIME_FLASH_ATTN': 'true',
                          'TLLAMA_BACKEND_PRIORITY': '', 'TLLAMA_DEBUG': 'true',
                          'TLLAMA_LOG_LEVEL': 'warning', 'TLLAMA_ORIGINS': 'https://example,,',
                          'TLLAMA_CONTEXT_LENGTH': ' '})
    assert (config.app.host, config.app.port) == ('::1', 2345)
    assert config.backend.runtime_overrides['flash_attn'] is True
    assert config.backend.backend_priority == ()
    assert config.app.log_level == 'warning' and not config.app.debug
    assert 'https://example' in config.cors_origins and 'http://localhost' in config.cors_origins
    assert config.backend.context_length == 0


def test_stronger_kv_shorthand_overrides_lower_layer_sides(tmp_path):
    file = tmp_path / 'server.toml'
    file.write_text('[runtime]\ntype_k="q4_0"\ntype_v="q5_0"\n')
    config = load(file, {'runtime.type_kv': 'f16', 'runtime.type_v': 'q8_0'},
                  environ={'TLLAMA_KV_CACHE_TYPE': 'bf16'})
    assert config.backend.k_cache_type == 'f16' and config.backend.v_cache_type == 'q8_0'
    assert config.sources['runtime.type_k'] == 'CLI'


def test_cli_boolean_flags_and_no_config(tmp_path):
    config = load_daemon_config(False, {'runtime.flash_attn': False, 'server.reload': True},
                                environ={'TLLAMA_CONFIG': str(tmp_path / 'absent')}, discover=False)
    assert config.backend.runtime_overrides['flash_attn'] is False
    assert config.app.reload


def test_discovery_uses_first_file_and_explicit_path_never_falls_back(tmp_path, monkeypatch):
    user, system = tmp_path / 'user.toml', tmp_path / 'system.toml'
    system.write_text('[server]\nport=1234\n')
    monkeypatch.setattr('tllama.daemon_config.default_config_paths', lambda env: (user, system))
    assert load_daemon_config(environ={}).app.port == 1234
    user.write_text('[server]\nport=2345\n')
    assert load_daemon_config(environ={}).app.port == 2345
    with pytest.raises(ConfigError, match='missing.toml'):
        load_daemon_config(tmp_path / 'missing.toml', environ={})
    assert load_daemon_config(environ={'TLLAMA_CONFIG': str(system)}).app.port == 1234


def test_reload_bridge_restores_environment_and_reports_only_owned_values(monkeypatch):
    monkeypatch.delenv('_TLLAMA_DAEMON_OPTIONS', raising=False)
    monkeypatch.setenv('HF_TOKEN', 'do-not-report')
    config = load(overrides={'server.port': '1234'})
    with daemon_configuration(config, {'server.port': '1234'}):
        assert current_daemon_config() is config
        import tllama.daemon_config as module
        token = module._ACTIVE.set(None)
        try:
            assert current_daemon_config().app.port == 1234
        finally:
            module._ACTIVE.reset(token)
    assert '_TLLAMA_DAEMON_OPTIONS' not in __import__('os').environ
    assert 'do-not-report' not in json.dumps(config.report())
