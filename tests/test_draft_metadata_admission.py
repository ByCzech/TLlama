"""Optional Python header evidence must not replace native draft validation."""

from pathlib import Path

import pytest

from tllama.helpers.gguf_metadata import build_model_metadata_payload
from tllama.helpers.metadata_cache import get_metadata_cache_path
from tllama.helpers.model_toml import TomlModelError
from tllama.worker import WorkerError
from test_helper_definitions import publish, reference
from test_native_manager import native_manager
from test_repository_worker_pins import protected_groups


def setup(manager, tmp_path, monkeypatch, evidence, *, split=False):
    _, relative = publish(manager, tmp_path, 'dflash', count=2 if split else 1)
    draft = manager.models_dir / relative
    name, toml = manager.test.define(extra=reference('draft', relative)
                                   + '[runtime]\nspec_type="draft-dflash"\n')
    get_metadata_cache_path(manager.metadata_cache_dir, draft).unlink(missing_ok=True)
    original = manager._get_model_metadata_sync

    def scan(path):
        if Path(path) == draft:
            if isinstance(evidence, Exception):
                raise evidence
            return evidence
        return original(path)

    monkeypatch.setattr(manager, '_get_model_metadata_sync', scan)
    return name, toml, draft


@pytest.mark.parametrize('evidence', [ValueError('unsupported Python header metadata'),
    OSError('optional header scan unavailable'), build_model_metadata_payload({})])
async def test_unavailable_or_unknown_unsharded_draft_metadata_reaches_native_loader(
        native_manager, tmp_path, monkeypatch, evidence):
    manager = native_manager()
    name, toml, draft = setup(manager, tmp_path, monkeypatch, evidence)
    original = toml.read_bytes()
    generation = await manager.preload_native_model(name)
    [worker] = manager.test.ledger
    args = worker.command
    assert args[args.index('--spec-type') + 1] == 'draft-dflash'
    assert args[args.index('--spec-draft-model') + 1] == str(draft)
    assert draft.parent.name in protected_groups(manager)
    assert toml.read_bytes() == original
    await manager.unload_native_model(name)
    await manager.wait_native_closed(generation)
    assert worker.process.returncode is not None and not protected_groups(manager)


@pytest.mark.parametrize('arch', ['llama', 'clip'])
async def test_known_wrong_architecture_is_still_rejected_before_native_start(
        native_manager, tmp_path, monkeypatch, arch):
    manager = native_manager()
    name, _, _ = setup(manager, tmp_path, monkeypatch, {'arch': arch})
    with pytest.raises(ValueError, match='requires dflash weights'):
        await manager.preload_native_model(name)
    assert not manager.test.ledger and not protected_groups(manager)


@pytest.mark.parametrize('evidence', [ValueError('unreadable shard header'), build_model_metadata_payload({})])
async def test_unverifiable_shard_set_cannot_use_optional_metadata_fallback(
        native_manager, tmp_path, monkeypatch, evidence):
    manager = native_manager()
    name, _, _ = setup(manager, tmp_path, monkeypatch, evidence, split=True)
    with pytest.raises((ValueError, TomlModelError)):
        await manager.preload_native_model(name)
    assert not manager.test.ledger and not protected_groups(manager)


async def test_native_startup_error_remains_authoritative_without_strategy_fallback(
        native_manager, tmp_path, monkeypatch):
    manager = native_manager(mode='fatal')
    name, toml, _ = setup(manager, tmp_path, monkeypatch, ValueError('unsupported Python header metadata'))
    original = toml.read_bytes()
    with pytest.raises(WorkerError, match='kernel image is invalid'):
        await manager.preload_native_model(name)
    assert len(manager.test.ledger) == 1
    [worker] = manager.test.ledger
    assert worker.command[worker.command.index('--spec-type') + 1] == 'draft-dflash'
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()
    assert not protected_groups(manager) and toml.read_bytes() == original
