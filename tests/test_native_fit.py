"""Fit controls must reach the private worker without global native variables."""

from dataclasses import replace
from pathlib import Path

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.daemon_config import load_daemon_config
from tllama.helpers.model_toml import TomlModelError, parse_model_toml
from tllama.native_cli import runtime_cli_args
from tllama.native_config import resolve_worker_config
from tllama.worker_config import WorkerLaunchSpec


MODEL = Path('/models/m.gguf')
LLM = '[llm]\nmodel="Local/m.gguf"\n'


def argv(resolved):
    spec = resolved.launch
    if spec.device is None:
        spec = replace(spec, device='CUDA0')
    return spec.command(Path('/private/key'))


def option(args, flag):
    return args[args.index(flag) + 1]


def test_absent_margin_keeps_native_default():
    resolved = resolve_worker_config(BackendConfig(), MODEL)
    assert '--fit-target' not in argv(resolved)


@pytest.mark.parametrize('margin', [0, 256, 512, 1024])
def test_margin_is_one_native_mib_value_without_byte_conversion(margin):
    resolved = resolve_worker_config(BackendConfig(), MODEL,
        parse_model_toml(LLM + f'[runtime]\nfit_target_mib={margin}\n'))
    assert option(argv(resolved), '--fit-target') == str(margin)


def test_model_margin_overrides_global_environment(tmp_path):
    server = tmp_path / 'server.toml'
    server.write_text('[runtime]\nfit_target_mib=1024\n')
    config = load_daemon_config(server, environ={'TLLAMA_RUNTIME_FIT_TARGET_MIB': '768'})
    assert config.report()['settings']['runtime.fit_target_mib']['value'] == 768
    assert option(argv(resolve_worker_config(config.backend, MODEL)), '--fit-target') == '768'
    model = parse_model_toml(LLM + '[runtime]\nfit_target_mib=512\n')
    resolved = resolve_worker_config(config.backend, MODEL, model)
    assert option(argv(resolved), '--fit-target') == '512'
    assert resolved.sources['fit_target_mib'] == 'TOML'


@pytest.mark.parametrize('value', ['-1', 'true', '1.5', '"512"', '[512]', '2147483648'])
def test_invalid_margin_is_rejected_before_model_work(value):
    with pytest.raises(TomlModelError, match='fit_target_mib'):
        parse_model_toml(LLM + f'[runtime]\nfit_target_mib={value}\n')


@pytest.mark.parametrize('value', ['-1', 'true', '1.5', '2147483648'])
def test_invalid_global_margin_is_a_config_error(value):
    with pytest.raises(ConfigError, match='FIT_TARGET_MIB'):
        load_daemon_config(False, environ={'TLLAMA_RUNTIME_FIT_TARGET_MIB': value}, discover=False)


def test_direct_launch_options_validate_margin_and_keep_native_environment_owned():
    spec = WorkerLaunchSpec(model=MODEL, device='CUDA0', fit=True,
                            runtime_options={'fit_target_mib': 512})
    assert option(spec.command(Path('/private/key')), '--fit-target') == '512'
    assert 'LLAMA_ARG_FIT_TARGET' not in spec.child_environment({'LLAMA_ARG_FIT_TARGET': '0'})
    with pytest.raises(ValueError, match='fit_target_mib'):
        runtime_cli_args({'fit_target_mib': -1})
