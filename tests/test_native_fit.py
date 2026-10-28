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


@pytest.mark.parametrize('device,settings,fit,layers', [
    ('CUDA0', '', 'off', 'all'),
    ('CUDA0', 'fit=true', 'on', 'auto'),
    ('CUDA0', 'fit=false', 'off', 'all'),
    ('CUDA0', 'fit=true\nn_gpu_layers=-1', 'on', 'all'),
    ('CUDA0', 'fit=false\nn_gpu_layers=5', 'off', '5'),
    ('none', 'fit=true', 'off', '0'),
])
def test_explicit_device_fit_policy_preserves_explicit_counts(device, settings, fit, layers):
    model = parse_model_toml(LLM + f'[runtime]\ndevice="{device}"\n{settings}\n')
    resolved = resolve_worker_config(BackendConfig(), MODEL, model)
    args = argv(resolved)
    assert option(args, '--fit') == fit and option(args, '--n-gpu-layers') == layers


def test_fit_global_environment_and_model_precedence():
    config = load_daemon_config(False, environ={'TLLAMA_RUNTIME_FIT': 'false'}, discover=False)
    assert config.backend.runtime_overrides['fit'] is False
    model = parse_model_toml(LLM + '[runtime]\nfit=true\n')
    resolved = resolve_worker_config(config.backend, MODEL, model)
    assert resolved.launch.fit is True and resolved.sources['fit'] == 'TOML'


@pytest.mark.parametrize('value', ['0', '1', '"false"', '[]'])
def test_fit_toml_requires_a_boolean(value):
    with pytest.raises(TomlModelError, match='fit'):
        parse_model_toml(LLM + f'[runtime]\nfit={value}')


def test_direct_device_launch_without_fit_override_keeps_all_off_default():
    args = WorkerLaunchSpec(model=MODEL, device='CUDA0').command(Path('/private/key'))
    assert option(args, '--fit') == 'off' and option(args, '--n-gpu-layers') == 'all'


@pytest.mark.parametrize('value', [0, 1, 'false'])
def test_private_fit_override_is_strictly_boolean(value):
    with pytest.raises(ValueError, match='fit must be a boolean'):
        WorkerLaunchSpec(model=MODEL, device='CUDA0', fit=value)


@pytest.mark.parametrize('settings,fit,layers', [
    ('', 'on', 'auto'), ('fit=false', 'off', 'all'), ('fit=true', 'on', 'auto'),
    ('n_gpu_layers=-1', 'off', 'all'), ('fit=false\nn_gpu_layers=-1', 'off', 'all'),
    ('fit=true\nn_gpu_layers=-1', 'on', 'all'), ('fit=false\nn_gpu_layers=3', 'off', '3'),
])
async def test_automatic_selection_respects_fit_override(monkeypatch, settings, fit, layers):
    from tllama.automatic_worker import AutomaticWorker
    async def discovery(spec):
        return ('CUDA0',)
    class Worker:
        def __init__(self, spec):
            self.spec = spec
        async def start(self):
            return self
        async def close(self):
            pass
    monkeypatch.setattr('tllama.automatic_worker.discover_devices', discovery)
    model = parse_model_toml(LLM + f'[runtime]\nn_ctx=132864\nfit_target_mib=512\n{settings}\n')
    resolved = resolve_worker_config(BackendConfig(), MODEL, model)
    worker = AutomaticWorker(resolved.launch, worker_factory=Worker)
    try:
        await worker.start()
        args = worker.spec.command(Path('/private/key'))
        assert option(args, '--fit') == fit and option(args, '--n-gpu-layers') == layers
        assert option(args, '--fit-target') == '512' and option(args, '--ctx-size') == '132864'
    finally:
        await worker.close()


@pytest.mark.parametrize('failure,retry_cpu', [('CUDA driver version is insufficient', True),
                                             ('out of memory', False)])
async def test_cpu_fallback_disables_fit_but_oom_never_changes_placement(monkeypatch, failure, retry_cpu):
    from tllama.automatic_worker import AutomaticWorker
    from tllama.worker import WorkerStartupError
    async def discovery(spec):
        return ('CUDA0',)
    attempts = []
    class Worker:
        def __init__(self, spec):
            self.spec = spec
            attempts.append(spec)
        async def start(self):
            if self.spec.device == 'CUDA0':
                raise WorkerStartupError(failure)
            return self
        async def close(self):
            pass
    monkeypatch.setattr('tllama.automatic_worker.discover_devices', discovery)
    model = parse_model_toml(LLM + '[runtime]\nfit=true\n')
    resolved = resolve_worker_config(BackendConfig(backend_priority=('CUDA',)), MODEL, model)
    worker = AutomaticWorker(resolved.launch, worker_factory=Worker)
    try:
        if retry_cpu:
            await worker.start()
            args = worker.spec.command(Path('/private/key'))
            assert option(args, '--fit') == 'off' and option(args, '--n-gpu-layers') == '0'
            assert [spec.device for spec in attempts] == ['CUDA0', 'none']
        else:
            with pytest.raises(WorkerStartupError, match='out of memory'):
                await worker.start()
            assert len(attempts) == 1
    finally:
        await worker.close()
