"""External draft selection and ownership; no claim of real GPU inference."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.backend import ModelManager
from tllama.helpers.model_toml import parse_model_toml, TomlModelError
from tllama.model_definitions import create_definition
from tllama.model_identity import parameter_changes
from tllama.ollama_options import ollama_runtime_overrides
from tllama.repository_names import definition_path
from tllama.repository_pins import snapshot
from tllama.repository_workers import resolve_launch
from tllama.native_config import resolve_worker_config
from tllama.worker import NativeWorker, WorkerError
from tllama.worker_config import WorkerLaunchSpec

from test_helper_definitions import publish, reference
from test_native_manager import native_manager
from test_native_public_api import public, request_body
from test_native_residency import wait_until
from test_repository_worker_pins import protected, protected_groups, runtime


def definition(main, draft, strategy='draft-dflash', limit=None):
    return (reference('llm', main) + (reference('draft', draft) if draft else '')
            + f'[runtime]\nspec_type="{strategy}"\n'
            + (f'draft_n_max={limit}\n' if limit is not None else ''))


def option(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize('strategy', ['none', 'draft-mtp', 'draft-dflash'])
@pytest.mark.parametrize('length', [None, 0, 2])
def test_configuration_selects_only_active_external_draft(strategy, length):
    text = definition('Local/main.gguf', 'Local/draft.gguf', strategy, 5)
    overrides = {} if length is None else ollama_runtime_overrides({'draft_num_predict': length})
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}),
        Path('/models/main.gguf'), parse_model_toml(text), draft=Path('/models/draft.gguf'),
        runtime_overrides=overrides, draft_metadata={'arch': 'dflash'})
    active = 'none' if length == 0 else strategy if strategy != 'none' or length is None else 'draft-mtp'
    args = resolved.launch.command(Path('/private/key'))
    assert option(args, '--spec-type') == active
    assert (resolved.launch.draft is not None) == (active == 'draft-dflash')
    if active == 'draft-dflash':
        assert option(args, '--spec-draft-model') == '/models/draft.gguf'
    else:
        assert '--spec-draft-model' not in args and '/models/draft.gguf' not in args
    if active == 'none':
        assert '--spec-draft-n-max' not in args
    else:
        assert option(args, '--spec-draft-n-max') == str(5 if length is None else length)
    assert '--offline' in args and '--draft' not in args


@pytest.mark.parametrize('device', ['none', 'Vulkan0', 'ROCm0', 'CUDA0'])
def test_captured_native_defaults_keep_draft_and_main_in_same_fit_command(device):
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': device, 'fit': 'true'}),
        Path('/models/main.gguf'), parse_model_toml(definition('main.gguf', 'draft.gguf')),
        draft=Path('/models/draft.gguf'))
    args = resolved.launch.command(Path('/private/key'))
    assert option(args, '--spec-draft-n-max') == '3'
    assert option(args, '--device') == device
    assert option(args, '--fit') == ('off' if device == 'none' else 'on')
    assert option(args, '--spec-draft-model') == '/models/draft.gguf'
    assert not any(flag in args for flag in ('--spec-draft-device', '--spec-draft-ngl',
                                           '--spec-draft-type-k', '--spec-draft-type-v'))


@pytest.mark.parametrize('strategy', ['draft-dflash2', 'draft-mtp,draft-dflash', 'draft-dflash,draft-mtp'])
def test_combined_or_invented_methods_are_rejected(strategy):
    with pytest.raises(TomlModelError, match='spec_type'):
        parse_model_toml(definition('main.gguf', 'draft.gguf', strategy))


@pytest.mark.parametrize('strategy,draft', [('draft-dflash', None), ('none', Path('/d.gguf')),
                                         ('draft-mtp', Path('relative.gguf')),
                                         ('draft-dflash', Path('relative.gguf'))])
def test_direct_launch_refuses_inconsistent_or_relative_draft(strategy, draft):
    with pytest.raises(ValueError, match='draft'):
        WorkerLaunchSpec(model=Path('/m.gguf'), device='none', draft=draft,
                         runtime_options={'spec_type': strategy})


@pytest.mark.parametrize('mode', ['embedding', 'decision'])
def test_external_draft_is_still_forbidden_for_non_generation_workers(mode):
    with pytest.raises(ValueError, match='speculative'):
        WorkerLaunchSpec(model=Path('/m.gguf'), device='none', draft=Path('/d.gguf'),
            runtime_options={'spec_type': 'draft-dflash'},
            **({'embedding': True} if mode == 'embedding' else {'decision_type': 'clef'}))


def test_dflash_length_identity_normalizes_captured_default_and_ignores_dormant_limit():
    def changes(strategy, limit):
        return parameter_changes(parse_model_toml(definition('main.gguf', 'draft.gguf', strategy, limit)), {})
    assert changes('draft-dflash', None) == changes('draft-dflash', 3)
    assert changes('draft-dflash', 2) != changes('draft-dflash', 3)
    assert changes('none', 2) == changes('none', 3)


@pytest.mark.parametrize('arch', ['llama', 'clip'])
async def test_wrong_draft_architecture_fails_before_worker_without_fallback(protected, arch):
    manager = protected
    draft = manager.test.group(arch)
    # Manually edited TOML bypasses create's projector role validation.
    path = manager.models_dir / definition_path('a')
    path.write_text(definition(manager.test.first, draft))
    with pytest.raises(ValueError, match='requires dflash weights'):
        await manager.preload_native_model('a')
    assert not manager.test.ledger and not snapshot(manager.repository).active


@pytest.mark.parametrize('missing', ['binding', 'file', 'shard'])
async def test_missing_draft_never_starts_or_falls_back(protected, missing):
    manager = protected
    draft = None if missing == 'binding' else manager.test.group('dflash', split=missing == 'shard')
    path = manager.models_dir / definition_path('a')
    path.write_text(definition(manager.test.first, draft))
    if draft:
        files = sorted((manager.models_dir / draft).parent.glob('*.gguf'))
        files[-1].unlink()
    with pytest.raises((ValueError, FileNotFoundError, ConfigError)):
        await manager.preload_native_model('a')
    assert not manager.test.ledger and not snapshot(manager.repository).active


@pytest.mark.parametrize('strategy', ['none', 'draft-mtp', 'draft-dflash'])
async def test_only_loaded_draft_group_is_pinned_until_reap(protected, strategy):
    manager = protected
    draft = manager.test.group('dflash', split=True)
    create_definition(manager, 'a', definition(manager.test.first, draft, strategy), {}, force=True)
    generation = await manager.preload_native_model('a')
    expected = {Path(manager.test.first).parts[1]}
    if strategy == 'draft-dflash':
        expected.add(Path(draft).parts[1])
    assert protected_groups(manager) == expected
    worker = manager.test.ledger[0]
    worker.allow_close.clear()
    await manager.delete_model('a')
    await wait_until(lambda: worker.state == 'closing')
    assert protected_groups(manager) == expected
    assert (manager.models_dir / draft).exists() == (strategy == 'draft-dflash')
    worker.allow_close.set()
    await manager.wait_native_closed(generation)
    assert not protected_groups(manager)
    assert not (manager.models_dir / draft).parent.exists()


async def test_draft_rebinding_keeps_old_generation_and_groups_until_last_lease(protected):
    manager = protected
    draft = manager.test.group('dflash', split=True)
    create_definition(manager, 'a', definition(manager.test.first, draft), {}, force=True)
    async with manager.acquire_native_model('a', keep_alive=0) as old:
        other = manager.test.group('dflash')
        create_definition(manager, 'a', definition(manager.test.first, other), {}, force=True)
        async with manager.acquire_native_model('a', keep_alive=0) as new:
            assert old.generation != new.generation
            assert old.resolved.launch.draft == manager.models_dir / draft
            assert new.resolved.launch.draft == manager.models_dir / other
            assert protected_groups(manager) == {Path(path).parts[1] for path in (manager.test.first, draft, other)}
        await manager.wait_native_closed(new.generation)
        assert (manager.models_dir / draft).exists()
    await manager.wait_native_closed(old.generation)
    assert not (manager.models_dir / draft).parent.exists()


async def test_changed_nonfirst_draft_shard_changes_snapshot_even_without_toml_edit(protected):
    manager = protected
    draft = manager.test.group('dflash', split=True)
    create_definition(manager, 'a', definition(manager.test.first, draft), {}, force=True)
    old = manager._resolve_native_definition_sync('a', None)
    shards = sorted((manager.models_dir / draft).parent.glob('*.gguf'))
    # Simulate an external replacement in the same group directory.
    with shards[1].open('ab') as stream:
        stream.write(b'changed payload')
    new = manager._resolve_native_definition_sync('a', None)
    assert old.fingerprint['draft'] != new.fingerprint['draft'] and old != new


async def test_direct_native_child_keeps_draft_pin_and_frozen_alias_after_supervisor_loss(protected, tmp_path):
    manager = protected
    executable, root = runtime(tmp_path)
    draft = manager.models_dir / manager.test.group('dflash', split=True)
    alias = tmp_path / 'draft alias.gguf'
    alias.symlink_to(draft)
    spec = WorkerLaunchSpec(model=manager.models_dir / manager.test.first, draft=alias,
        device='none', executable=executable, transport='tcp', load_timeout=5,
        environment={'STUB_ROOT': str(root)}, runtime_options={'spec_type': 'draft-dflash'})
    launch, resources = resolve_launch(spec)
    assert launch.draft == draft and resources[manager.repository.root] == {draft.parent.name, Path(manager.test.first).parts[1]}
    worker = NativeWorker(spec)
    try:
        await worker.start()
        alias.unlink()
        assert option(worker.command, '--spec-draft-model') == str(draft)
        pin = worker._repository_pins[0]
        os.close(pin._fd)
        pin._fd = None
        assert snapshot(manager.repository, remove_stale=True).protects_group(draft.parent.name)
    finally:
        await worker.close()
    assert not snapshot(manager.repository, remove_stale=True).active


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_public_request_off_on_and_omitted_preserves_dflash_and_definition(public, tmp_path, path, stream):
    manager, client, name, toml = public
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    toml.write_text(toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="draft-dflash"\ndraft_n_max=5\n')
    original = toml.read_bytes()
    digest = manager._resolve_native_definition_sync(name, None).file_info['model_digest']
    for value in (0, 0, 2, 2, None):
        response = await client.post(path, json={**request_body(name, path, stream),
            'options': {} if value is None else {'draft_num_predict': value}})
        assert response.status_code == 200, response.text
        args = manager.test.ledger[-1].command
        assert option(args, '--spec-type') == ('none' if value == 0 else 'draft-dflash')
        assert ('--spec-draft-model' in args) == (value != 0)
        if value != 0:
            assert option(args, '--spec-draft-model') == str(manager.models_dir / draft)
            assert option(args, '--spec-draft-n-max') == str(5 if value is None else value)
    assert len(manager.test.ledger) == 3 and toml.read_bytes() == original
    assert manager._resolve_native_definition_sync(name, None).file_info['model_digest'] == digest


async def test_native_startup_error_is_reported_without_strategy_fallback(native_manager, tmp_path):
    manager = native_manager(mode='fatal')
    name, toml = manager.test.define()
    _, draft = publish(manager, tmp_path, 'dflash')
    toml.write_text(toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="draft-dflash"\n')
    with pytest.raises(WorkerError):
        await manager.preload_native_model(name)
    assert len(manager.test.ledger) == 1
    worker = manager.test.ledger[0]
    assert option(worker.command, '--spec-type') == 'draft-dflash'
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()
    assert not snapshot(manager.repository).active


async def test_automatic_backend_attempts_keep_draft_in_fit_and_inherited_reservation(protected, tmp_path):
    template = protected
    draft = template.test.group('dflash', split=True)
    create_definition(template, 'a', definition(template.test.first, draft), {}, force=True)
    executable, root = runtime(tmp_path)
    config = replace(template.config, runtime_overrides={}, runtime_executable=executable,
                     backend_priority=('ROCm', 'Vulkan'))
    ledger = []
    environment = {'STUB_ROOT': str(root), 'STUB_DISCOVERY': '["ROCm0", "Vulkan0"]',
                   'STUB_DEVICE_FAILURES': '{"ROCm0": "ROCm error: no kernel image is available"}'}
    def factory(spec):
        worker = NativeWorker(replace(spec, transport='tcp', environment=environment))
        ledger.append(worker)
        return worker
    manager = ModelManager(config, native_worker_factory=factory,
                           native_launch_options={'environment': environment})
    try:
        await manager.preload_native_model('a')
        assert len(ledger) == 2 and ledger[0].process.returncode == 23
        for worker in ledger:
            assert option(worker.command, '--spec-draft-model') == str(manager.models_dir / draft)
            assert option(worker.command, '--spec-type') == 'draft-dflash'
            assert option(worker.command, '--fit') == 'on'
        assert ledger[0].spec.resource_fds == ledger[1].spec.resource_fds != ()
        expected = {Path(path).parts[1] for path in (template.test.first, draft)}
        assert protected_groups(manager) == expected
        pin = next(iter(manager._native._entries.values())).definition.protection
        os.close(pin._fd)
        pin._fd = None
        state = snapshot(manager.repository, remove_stale=True)
        assert all(state.protects_group(group) for group in expected)
    finally:
        await manager.shutdown()
    assert not snapshot(manager.repository, remove_stale=True).active
