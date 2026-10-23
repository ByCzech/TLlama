"""Opt-in embedded MTP load settings, immutable generations and native failures."""

from pathlib import Path

import pytest

from tllama.config import BackendConfig, ConfigError, load_backend_config_from_env
from tllama.helpers.model_toml import parse_model_toml, TomlModelError
from tllama.native_config import resolve_worker_config
from tllama.worker import WorkerError

from test_native_manager import native_manager
from test_native_public_api import public


LLM = '[llm]\nmodel="Local/m.gguf"\n[runtime]\n'


def resolve(extra='', **global_values):
    return resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none', **global_values}),
                                 Path('/models/m.gguf'), parse_model_toml(LLM + extra))


def command(resolved):
    return resolved.launch.command(Path('/private/key'))


def option(args, flag):
    return args[args.index(flag) + 1]


def test_unset_mtp_does_not_add_new_flags_or_require_newer_engines():
    args = command(resolve())
    assert '--spec-type' not in args and '--spec-draft-n-max' not in args


@pytest.mark.parametrize('draft', [None, 0, 1, 3, 8])
def test_mtp_is_explicit_with_typed_draft_length(draft):
    settings = 'spec_type="draft-mtp"\n' + ('' if draft is None else f'draft_n_max={draft}\n')
    resolved = resolve(settings)
    args = command(resolved)
    assert option(args, '--spec-type') == 'draft-mtp'
    assert option(args, '--spec-draft-n-max') == str(3 if draft is None else draft)
    assert resolved.runtime['draft_n_max'] == (3 if draft is None else draft)
    assert '--model-draft' not in args and '--mtp' not in args


@pytest.mark.parametrize('value', ['""', '"mtp"', '"draft-simple"', 'true', '42', '["draft-mtp"]'])
def test_invalid_or_unintegrated_strategy_is_not_passed_through(value):
    with pytest.raises(TomlModelError, match='spec_type'):
        resolve('spec_type=' + value)


@pytest.mark.parametrize('value', ['-1', 'true', '3.5', '2147483648'])
def test_draft_limit_refuses_invalid_native_integer(value):
    with pytest.raises(TomlModelError, match='draft_n_max'):
        resolve('draft_n_max=' + value)


def test_toml_can_disable_global_mtp_and_keep_its_dormant_draft_default():
    resolved = resolve('spec_type="none"', spec_type='draft-mtp', draft_n_max='5')
    args = command(resolved)
    assert option(args, '--spec-type') == 'none' and '--spec-draft-n-max' not in args
    assert resolved.runtime['draft_n_max'] == 5 and resolved.sources['spec_type'] == 'TOML'
    assert '--spec-type' not in command(resolve(draft_n_max='5'))
    assert option(command(resolve('draft_n_max=2', spec_type='draft-mtp', draft_n_max='5')),
                  '--spec-draft-n-max') == '2'


def test_environment_uses_the_same_config_contract(monkeypatch):
    monkeypatch.setenv('TLLAMA_RUNTIME_DEVICE', 'none')
    monkeypatch.setenv('TLLAMA_RUNTIME_SPEC_TYPE', 'draft-mtp')
    monkeypatch.setenv('TLLAMA_RUNTIME_DRAFT_N_MAX', '4')
    resolved = resolve_worker_config(load_backend_config_from_env(), Path('/models/m.gguf'))
    assert option(command(resolved), '--spec-draft-n-max') == '4'
    monkeypatch.setenv('TLLAMA_RUNTIME_DRAFT_N_MAX', '-1')
    with pytest.raises(ConfigError, match='draft_n_max'):
        resolve_worker_config(load_backend_config_from_env(), Path('/models/m.gguf'))


async def test_changes_retire_the_old_configuration_without_mutating_a_live_lease(native_manager):
    manager = native_manager(limit=2)
    name, toml = manager.test.define(extra='[runtime]\nspec_type="draft-mtp"\ndraft_n_max=3\n')
    original = toml.read_text()
    async with manager.acquire_native_model(name, keep_alive=-1) as first:
        args = first.worker.command
        assert option(args, '--spec-draft-n-max') == '3'
        toml.write_text(original.replace('draft_n_max=3', 'draft_n_max=2'))
        async with manager.acquire_native_model(name, keep_alive=-1) as second:
            assert first.generation != second.generation
            assert option(second.worker.command, '--spec-draft-n-max') == '2'
            assert first.worker.command == args and first.worker.process.returncode is None
            old = next(r for r in manager.native_snapshot() if r['generation'] == first.generation)
            assert old['state'] == 'draining' and old['leases'] == 1
    await manager.wait_native_closed(first.generation)
    toml.write_text(original.replace('draft-mtp', 'none'))
    async with manager.acquire_native_model(name) as disabled:
        assert disabled.generation != second.generation
        assert option(disabled.worker.command, '--spec-type') == 'none'
        assert '--spec-draft-n-max' not in disabled.worker.command


async def test_native_mtp_failure_cleans_up_without_relaunching_disabled(native_manager):
    manager = native_manager(mode='fatal')
    name, _ = manager.test.define(extra='[runtime]\nspec_type="draft-mtp"\n')
    with pytest.raises(WorkerError):
        await manager.preload_native_model(name)
    await manager.shutdown()
    assert len(manager.test.ledger) == 1 and manager.native_snapshot() == []
    worker = manager.test.ledger[0]
    assert option(worker.command, '--spec-type') == 'draft-mtp'
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('name,value', [('spec_type', 'draft-mtp'), ('draft_n_max', 3)])
async def test_request_cannot_silently_change_a_worker_load_setting(public, path, name, value):
    manager, client, model, _ = public
    response = await client.post(path, json={'model': model, 'messages': [], 'prompt': 'x', 'options': {name: value}})
    assert response.status_code == 400 and not manager.test.ledger
