"""Configured I/O prefixes and portable identity must agree across owners."""

import json
from pathlib import Path

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers import metadata_cache as cache
from tllama.helpers.model_toml import TomlModelError, resolve_repo_relative_path
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_pins import snapshot
from tllama.repository_store import RepositoryStore, prepare_group
from test_model_create import gguf_file, create_client


def publish(manager, tmp_path, *, arch='llama', shards=1):
    names = ([f'{arch}-{i+1:05d}-of-{shards:05d}.gguf' for i in range(shards)]
             if shards > 1 else [arch + '.gguf'])
    sources = [gguf_file(tmp_path / name, arch=arch,
                         shard=i if shards > 1 else None,
                         count=shards if shards > 1 else None) for i, name in enumerate(names)]
    paths = manager.repository.publish(prepare_group(sources))
    return paths[0].relative_to(manager.models_dir).as_posix()


def cache_snapshot(root):
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (root / '.tllama/metadata-cache').iterdir()}


@pytest.mark.parametrize('kind', ['direct', 'alias', 'parent-alias', 'relative'])
async def test_shared_shards_keep_prefix_identity_cache_and_protection(tmp_path, monkeypatch, kind):
    root = tmp_path / 'models'
    direct = ModelManager(BackendConfig(models_dir=str(root), runtime_overrides={'device': 'none'}))
    direct.ensure_storage()
    llm = publish(direct, tmp_path, shards=2)
    projector = publish(direct, tmp_path, arch='clip')
    draft = publish(direct, tmp_path, arch='dflash')
    adapter = publish(direct, tmp_path, arch='adapter')
    text = (f'[llm]\nmodel={json.dumps(llm)}\n[mmproj]\nmodel={json.dumps(projector)}\n'
            f'[draft]\nmodel={json.dumps(draft)}\n[runtime]\nspec_type="draft-dflash"\n'
            f'lora_path={json.dumps(adapter)}\n')
    create_definition(direct, 'a', text, {})
    copy_definition(direct, 'a', 'b')
    expected = direct._resolve_native_definition_sync('a', 4096)
    warm = cache_snapshot(root)
    configured = root
    if kind == 'alias':
        configured = tmp_path / 'alias'
        configured.symlink_to(root, target_is_directory=True)
    elif kind == 'parent-alias':
        parent = tmp_path / 'parent-alias'
        parent.symlink_to(tmp_path, target_is_directory=True)
        configured = parent / root.name
    elif kind == 'relative':
        monkeypatch.chdir(tmp_path)
        configured = Path('models')
    manager = ModelManager(BackendConfig(models_dir=str(configured), runtime_overrides={'device': 'none'}))
    prefix = configured.absolute()
    assert manager.models_dir == manager.repository.root == prefix
    assert manager.metadata_cache_dir == prefix / '.tllama/metadata-cache'
    # A relative configuration is anchored once, even if a caller changes cwd.
    monkeypatch.chdir(tmp_path.parent)

    def fail(*args, **kwargs):
        pytest.fail('Valid published caches must not reread headers or hash weights')

    monkeypatch.setattr(manager, '_get_model_metadata_sync', fail)
    monkeypatch.setattr(manager, '_compute_content_sha256', fail)
    definition = None
    try:
        rows = await manager.list_local_models_with_metadata()
        assert {row['id'] for row in rows} == {'a', 'b'}
        assert len({row['digest'] for row in rows}) == 1
        info = manager._build_model_file_info('a')
        assert Path(info['path']) == prefix / llm
        assert info['sha256'] == direct._build_model_file_info('a')['sha256']
        definition = await manager._resolve_protected_native_definition('a', 4096)
        assert definition.fingerprint == expected.fingerprint
        assert definition.fingerprint['model']['path'] == llm
        assert definition.fingerprint['toml']['path'] == 'Local/a.toml'
        assert definition.fingerprint['projector']['path'] == projector
        assert definition.fingerprint['draft'][0]['path'] == draft
        assert definition.resolved.launch.model == root / llm
        assert set(snapshot(direct.repository).active[0]['groups']) == {
            Path(value).parts[1] for value in (llm, projector, draft, adapter)}
        assert cache_snapshot(root) == warm
        # Physical reclamation waits for this snapshot, including every shard
        # and both attached helpers, even when deletion uses the other prefix.
        direct.delete_model_definition('a')
        manager.delete_model_definition('b')
        assert all((root / value).is_file() for value in (llm, projector, draft, adapter))
        definition.protection.close()
        assert not snapshot(direct.repository).active
        assert all(not (root / value).parent.exists() for value in (llm, projector, draft, adapter))
    finally:
        if definition is not None:
            definition.protection.close()
        await manager.shutdown()
        await direct.shutdown()


async def test_moved_managed_repository_keeps_fingerprints_and_existing_caches(tmp_path, monkeypatch):
    root = tmp_path / 'old'
    manager = ModelManager(BackendConfig(models_dir=str(root), runtime_overrides={'device': 'none'}))
    manager.ensure_storage()
    relative = publish(manager, tmp_path, shards=2)
    create_definition(manager, 'a', f'[llm]\nmodel={json.dumps(relative)}\n', {})
    expected = manager._resolve_native_definition_sync('a', 4096)
    warm = cache_snapshot(root)
    await manager.shutdown()
    moved = tmp_path / 'moved'
    root.rename(moved)
    other = ModelManager(BackendConfig(models_dir=str(moved), runtime_overrides={'device': 'none'}))
    def fail(*args, **kwargs):
        pytest.fail('Move must preserve measured caches')
    monkeypatch.setattr(other, '_get_model_metadata_sync', fail)
    monkeypatch.setattr(other, '_compute_content_sha256', fail)
    try:
        actual = other._resolve_native_definition_sync('a', 4096)
        assert actual.fingerprint == expected.fingerprint
        assert actual.file_info['model_digest'] == expected.file_info['model_digest']
        assert actual.resolved.launch.model == moved / relative
        assert cache_snapshot(moved) == warm
    finally:
        await other.shutdown()


@pytest.mark.parametrize('section', ['metadata', 'digest'])
def test_store_cache_load_resolves_one_reference_and_accepts_previous_document(tmp_path, monkeypatch, section):
    store = RepositoryStore(tmp_path / 'models')
    path = store.root / 'Local/a.gguf'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'unchanged weights')
    save = getattr(cache, f'save_{section}_cache')
    load = getattr(cache, f'load_{section}_cache')
    payload = {'value': section}
    # The pre-A5 standalone call writes the same persistent schema/key.
    save(store.cache_dir, 'a', path, payload)
    warm = cache_snapshot(store.root)
    calls = []
    resolve = Path.resolve
    def counted(value, *args, **kwargs):
        calls.append(value)
        return resolve(value, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', counted)
    assert load(store.cache_dir, path, repository=store) == payload
    assert calls == [path]
    assert cache_snapshot(store.root) == warm
    captured = cache.build_model_file_fingerprint(path, repository=store)
    assert captured['path'] == 'Local/a.gguf'
    path.write_bytes(b'replacement weights of different size')
    save(store.cache_dir, 'a', path, payload, fingerprint=captured, repository=store)
    assert load(store.cache_dir, path, repository=store) is None


def test_alias_prefix_does_not_allow_internal_escape(tmp_path):
    root = tmp_path / 'models'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    outside = tmp_path / 'outside.gguf'
    outside.write_bytes(b'outside')
    (root / 'escape.gguf').symlink_to(outside)
    store = RepositoryStore(alias)
    with pytest.raises(TomlModelError, match='outside'):
        resolve_repo_relative_path('escape.gguf', alias, repository=store)
    with pytest.raises(ValueError):
        cache.build_model_file_fingerprint(alias / 'escape.gguf', repository=store)
    cache.save_digest_cache(store.cache_dir, 'escape', alias / 'escape.gguf', {'x': 1}, repository=store)
    assert not store.cache_dir.exists()


async def test_upload_and_repeat_create_share_managed_data_through_alias(manager, create_client, tmp_path, monkeypatch):
    from tllama.client_import import create_from_file
    from tllama.routers import ollama
    alias = tmp_path / 'alias'
    alias.symlink_to(manager.models_dir, target_is_directory=True)
    other = ModelManager(BackendConfig(models_dir=str(alias)))
    monkeypatch.setattr(ollama, 'model_manager', other)
    source = gguf_file(tmp_path / 'client-weights.gguf')
    definition = tmp_path / 'client-model.toml'
    definition.write_text(f'[llm]\nfrom={json.dumps(str(source))}\n')
    try:
        await create_from_file(create_client, 'a', definition, as_json=True)
        await create_from_file(create_client, 'b', definition, as_json=True)
        rows = await other.list_local_models_with_metadata()
        assert {row['id'] for row in rows} == {'a', 'b'}
        assert len({row['digest'] for row in rows}) == 1
        assert len(list((alias / 'data').iterdir())) == 1
        assert manager.read_model_definition('a') == other.read_model_definition('b')
    finally:
        await other.shutdown()


def test_alias_delete_recognizes_and_removes_unused_helper(manager, tmp_path):
    llm = publish(manager, tmp_path)
    projector = publish(manager, tmp_path, arch='clip')
    create_definition(manager, 'helper', f'[mmproj]\nmodel={json.dumps(projector)}\n', {})
    create_definition(manager, 'a', f'[llm]\nmodel={json.dumps(llm)}\n[mmproj]\nmodel={json.dumps(projector)}\n', {})
    alias = tmp_path / 'alias'
    alias.symlink_to(manager.models_dir, target_is_directory=True)
    other = ModelManager(BackendConfig(models_dir=str(alias)))
    result = other.delete_model_definition('a', remove_unused_helpers=True)
    assert result['cleanup']['removed_helpers'] == ['helper']
    assert not result['cleanup']['warnings']
    assert manager.read_model_definition('helper') is None
    assert not (manager.models_dir / projector).exists()
