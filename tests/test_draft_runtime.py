"""External draft tuning is typed, scoped and preserved through native launches."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tllama.automatic_worker import AutomaticWorker
from tllama.config import BackendConfig, ConfigError
from tllama.daemon_config import load_daemon_config
from tllama.helpers.model_toml import parse_model_toml, TomlModelError
from tllama.model_identity import parameter_changes
from tllama.native_cli import runtime_cli_args
from tllama.native_config import resolve_worker_config
from tllama.worker import WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec

from test_helper_definitions import publish, reference
from test_native_manager import native_manager
from test_native_public_api import public


VALUES = {'draft_device': 'none', 'draft_n_gpu_layers': 0,
          'draft_type_k': 'q8_0', 'draft_type_v': 'q4_0', 'draft_n_ubatch': 128}
FLAGS = {'draft_device': '--spec-draft-device', 'draft_n_gpu_layers': '--spec-draft-ngl',
         'draft_type_k': '--spec-draft-type-k', 'draft_type_v': '--spec-draft-type-v',
         'draft_n_ubatch': '--spec-draft-n-ubatch'}


def text(strategy='draft-dflash', **values):
    return ('[llm]\nmodel="Local/m.gguf"\n[draft]\nmodel="Local/d.gguf"\n[runtime]\n'
            + f'spec_type="{strategy}"\n'
            + ''.join(f'{name}={json.dumps(value)}\n' for name, value in values.items()))


def resolve(strategy='draft-dflash', **values):
    return resolve_worker_config(BackendConfig(runtime_overrides={'device': 'Vulkan0'}),
        Path('/models/m.gguf'), parse_model_toml(text(strategy, **values)), draft=Path('/models/d.gguf'),
        draft_metadata={'arch': 'dflash'})


def args(resolved):
    return resolved.launch.command(Path('/private/key'))


def option(argv, flag):
    return argv[argv.index(flag) + 1]


def test_all_tuning_controls_reach_native_without_changing_main_placement_or_kv():
    resolved = resolve(**VALUES)
    command = args(resolved)
    for name, value in VALUES.items():
        assert option(command, FLAGS[name]) == str(value)
    assert option(command, '--device') == 'Vulkan0'
    assert option(command, '--cache-type-k') == option(command, '--cache-type-v') == 'f16'
    assert option(command, '--ubatch-size') == '512'
    assert option(command, '--spec-draft-model') == '/models/d.gguf'
    assert '--offline' in command


@pytest.mark.parametrize('strategy', ['none', 'draft-mtp'])
def test_external_tuning_is_dormant_outside_dflash_even_if_it_would_conflict(strategy):
    values = dict(VALUES, draft_n_gpu_layers='all')
    command = args(resolve(strategy, **values))
    assert all(flag not in command for flag in FLAGS.values())
    assert '--spec-draft-model' not in command
    assert parameter_changes(parse_model_toml(text(strategy, **values)), {}, draft_metadata={'arch': 'dflash'}) == parameter_changes(
        parse_model_toml(text(strategy)), {}, draft_metadata={'arch': 'dflash'})


@pytest.mark.parametrize('value,expected', [('auto', 'auto'), ('all', 'all'), (-1, 'all'), (0, '0'), (17, '17')])
def test_layer_spellings_are_typed_and_use_captured_native_flag(value, expected):
    assert option(args(resolve(draft_n_gpu_layers=value)), '--spec-draft-ngl') == expected


@pytest.mark.parametrize('side', ['k', 'v'])
@pytest.mark.parametrize('value,expected', [('Q8_0', 'q8_0'), (8, 'q8_0'), ('bf16', 'bf16'), (1, 'f16')])
def test_draft_kv_names_and_enums_are_canonical_in_resolved_configuration(side, value, expected):
    key = 'draft_type_' + side
    resolved = resolve(**{key: value})
    assert resolved.runtime[key] == expected
    assert option(args(resolved), FLAGS[key]) == expected


@pytest.mark.parametrize('key,value', [
    ('draft_device', ''), ('draft_device', 'none,Vulkan0'), ('draft_device', ' Vulkan0'),
    ('draft_device', True), ('draft_device', ['Vulkan0']),
    ('draft_n_gpu_layers', -2), ('draft_n_gpu_layers', 2**31), ('draft_n_gpu_layers', True),
    ('draft_n_gpu_layers', '1'), ('draft_n_gpu_layers', 'AUTO'), ('draft_n_gpu_layers', 1.5),
    ('draft_type_k', 'q4_k'), ('draft_type_v', 999), ('draft_type_k', True), ('draft_type_v', 1.0),
    ('draft_n_ubatch', -1), ('draft_n_ubatch', 2**31), ('draft_n_ubatch', True),
    ('draft_n_ubatch', 1.5), ('draft_n_ubatch', '128'),
])
def test_invalid_values_fail_before_native_launch_including_dormant_settings(key, value):
    with pytest.raises(TomlModelError, match=key):
        parse_model_toml(text('none', **{key: value}))
    with pytest.raises(ValueError, match=key):
        runtime_cli_args({'spec_type': 'none', key: value})


@pytest.mark.parametrize('layers', [-1, 1, 'all'])
def test_explicit_cpu_draft_rejects_conflicting_layers_when_active(layers):
    with pytest.raises(ConfigError, match='draft_device.*conflicts'):
        resolve(draft_device='none', draft_n_gpu_layers=layers)


def test_daemon_file_environment_and_model_layers_keep_role_specific_settings(tmp_path):
    server = tmp_path / 'server.toml'
    server.write_text('[runtime]\ndraft_device="ROCm0"\ndraft_n_gpu_layers=8\n'
                      'draft_type_k="bf16"\ndraft_type_v="bf16"\ndraft_n_ubatch=256\n')
    config = load_daemon_config(server, discover=False, environ={
        'TLLAMA_RUNTIME_DRAFT_DEVICE': 'Vulkan0', 'TLLAMA_RUNTIME_DRAFT_N_GPU_LAYERS': 'all',
        'TLLAMA_RUNTIME_DRAFT_TYPE_K': '8', 'TLLAMA_RUNTIME_DRAFT_TYPE_V': 'Q8_0',
        'TLLAMA_RUNTIME_DRAFT_N_UBATCH': '128'})
    resolved = resolve_worker_config(config.backend, Path('/models/m.gguf'),
        parse_model_toml(text(draft_n_gpu_layers='auto', draft_type_v='q4_0', draft_n_ubatch=0)),
        draft=Path('/models/d.gguf'))
    command = args(replace(resolved, launch=replace(resolved.launch, device='ROCm0')))
    assert option(command, '--spec-draft-device') == 'Vulkan0'
    assert option(command, '--spec-draft-ngl') == 'auto'
    assert option(command, '--spec-draft-type-k') == 'q8_0'
    assert option(command, '--spec-draft-type-v') == 'q4_0'
    assert option(command, '--spec-draft-n-ubatch') == '0'
    assert resolved.sources['draft_type_k'] == 'global'
    assert resolved.sources['draft_n_gpu_layers'] == 'TOML'
    inherited = resolve_worker_config(config.backend, Path('/models/m.gguf'),
        parse_model_toml(text()), draft=Path('/models/d.gguf'))
    assert inherited.runtime['draft_n_gpu_layers'] == -1


@pytest.mark.parametrize('key,value', [('DRAFT_N_GPU_LAYERS', '-2'), ('DRAFT_TYPE_K', 'q4_k'),
                                     ('DRAFT_N_UBATCH', '-1'), ('DRAFT_DEVICE', 'none,CUDA0')])
def test_bad_environment_keeps_variable_name(key, value):
    with pytest.raises(ConfigError, match='TLLAMA_RUNTIME_' + key):
        load_daemon_config(False, environ={'TLLAMA_RUNTIME_' + key: value}, discover=False)


def test_native_defaults_and_equivalent_spellings_preserve_identity():
    def changes(**values):
        return parameter_changes(parse_model_toml(text(**values)), {})
    assert changes() == changes(draft_type_k=1, draft_type_v='F16', draft_n_gpu_layers='auto', draft_n_ubatch=0)
    assert changes(draft_n_gpu_layers='all', draft_type_k='Q8_0') == changes(draft_n_gpu_layers=-1, draft_type_k=8)
    for name, value in VALUES.items():
        assert changes(**{name: value}) != changes()


@pytest.mark.parametrize('draft_device', [None, 'none', 'CUDA1'])
async def test_automatic_cpu_fallback_preserves_explicit_draft_placement(monkeypatch, draft_device):
    async def discover(spec):
        return ('CUDA0',)
    attempts = []
    class Worker:
        def __init__(self, spec):
            self.spec = spec
            attempts.append(spec)
        async def start(self):
            if self.spec.device == 'CUDA0':
                raise WorkerStartupError('CUDA driver version is insufficient')
            return self
        async def close(self):
            pass
    monkeypatch.setattr('tllama.automatic_worker.discover_devices', discover)
    values = {'draft_type_k': 'q8_0', 'draft_n_gpu_layers': 'auto', 'draft_n_ubatch': 128}
    if draft_device is not None:
        values['draft_device'] = draft_device
    requested = resolve_worker_config(BackendConfig(backend_priority=('CUDA',)), Path('/models/m.gguf'),
        parse_model_toml(text(**values)), draft=Path('/models/d.gguf'))
    worker = AutomaticWorker(requested.launch, worker_factory=Worker)
    try:
        await worker.start()
        assert [spec.device for spec in attempts] == ['CUDA0', 'none']
        for spec in attempts:
            command = spec.command(Path('/private/key'))
            assert option(command, '--spec-draft-type-k') == 'q8_0'
            assert option(command, '--spec-draft-ngl') == 'auto'
            assert option(command, '--spec-draft-n-ubatch') == '128'
            assert ('--spec-draft-device' in command) == (draft_device is not None)
            if draft_device is not None:
                assert option(command, '--spec-draft-device') == draft_device
    finally:
        await worker.close()


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('key,value', list(VALUES.items()))
async def test_tuning_does_not_become_new_public_inference_options(public, path, key, value):
    manager, client, name, _ = public
    response = await client.post(path, json={'model': name, 'messages': [], 'prompt': 'x',
                                            'options': {key: value}})
    assert response.status_code == 400 and not manager.test.ledger


async def test_tuning_change_uses_new_generation_and_zero_request_omits_all_external_flags(native_manager, tmp_path):
    manager = native_manager(limit=2)
    name, toml = manager.test.define()
    _, draft = publish(manager, tmp_path, 'dflash')
    original = toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="draft-dflash"\n'
    toml.write_text(original + 'draft_n_ubatch=128\ndraft_type_k="q8_0"\n')
    async with manager.acquire_native_model(name, keep_alive=0) as first:
        assert option(first.worker.command, '--spec-draft-n-ubatch') == '128'
        toml.write_text(original + 'draft_n_ubatch=64\ndraft_type_k="q4_0"\n')
        async with manager.acquire_native_model(name, keep_alive=0) as second:
            assert first.generation != second.generation
            assert option(second.worker.command, '--spec-draft-n-ubatch') == '64'
            assert option(first.worker.command, '--spec-draft-type-k') == 'q8_0'
            assert first.worker.process.returncode is None
    await manager.wait_native_closed(first.generation)
    await manager.wait_native_closed(second.generation)
    async with manager.acquire_native_model(name, runtime_overrides={'draft_num_predict': 0}) as off:
        assert all(flag not in off.worker.command for flag in FLAGS.values())
        assert '--spec-draft-model' not in off.worker.command
    spec = WorkerLaunchSpec(model=Path('/m.gguf'), draft=Path('/d.gguf'), device='none',
                           runtime_options={'spec_type': 'draft-dflash', **VALUES})
    inherited = {'PATH': '/bin', 'LLAMA_ARG_SPEC_DRAFT_DEVICE': 'CUDA99',
                 'LLAMA_ARG_SPEC_DRAFT_N_UBATCH': '999'}
    assert spec.child_environment(inherited) == {'PATH': '/bin'}
