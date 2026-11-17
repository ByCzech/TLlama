"""Local artifact reuse keeps revision checks, role identity and I/O ownership."""

import asyncio
from hashlib import sha256
import os
from pathlib import Path
import threading

import pytest

from tllama.helpers.model_toml import TomlModelError, parse_model_toml
from tllama.model_artifacts import ModelArtifacts, gguf_files
from tllama.model_identity import model_digest, parameter_changes
from test_model_artifacts import artifact, define


@pytest.mark.parametrize('count', [1, 2, 3])
def test_shard_headers_are_read_once_per_member(manager, artifact, count):
    paths = ([artifact('model.gguf')] if count == 1 else
             [artifact(f'model-{i+1:05d}-of-{count:05d}.gguf',
                       **{'split.count': count, 'split.no': i}) for i in range(count)])
    calls = []
    def read(path):
        calls.append(path)
        return manager._read_metadata_for_path(path)
    assert gguf_files(paths[0], read, manager.models_dir, repository=manager.repository) == tuple(paths)
    assert calls == paths


def test_missing_first_shard_remains_a_model_error(manager):
    path = manager.local_models_dir / 'missing-00001-of-00002.gguf'
    with pytest.raises(TomlModelError, match='Cannot inspect GGUF shard set'):
        gguf_files(path, manager._read_metadata_for_path, manager.models_dir, repository=manager.repository)


@pytest.mark.parametrize('atomic', [False, True])
def test_first_shard_replacement_during_inspection_is_rejected(manager, artifact, atomic):
    paths = [artifact(f'model-{i+1:05d}-of-00002.gguf', **{'split.count': 2, 'split.no': i})
             for i in range(2)]
    def read(path):
        value = manager._read_metadata_for_path(path)
        if path == paths[1]:
            before = paths[0].stat()
            if atomic:
                replacement = paths[0].with_suffix('.replacement')
                replacement.write_bytes(paths[0].read_bytes())
                os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
                replacement.replace(paths[0])
            else:
                with paths[0].open('ab') as handle:
                    handle.write(b'new tensor bytes')
        return value
    with pytest.raises(TomlModelError, match='first shard changed'):
        gguf_files(paths[0], read, manager.models_dir, repository=manager.repository)


def test_unique_size_uses_one_fresh_inode_walk(manager, artifact, monkeypatch):
    first, second = artifact('first.gguf'), artifact('second.gguf')
    alias = first.with_name('alias.gguf')
    os.link(first, alias)
    expected = first.stat().st_size + second.stat().st_size
    data = ModelArtifacts((first, second), adapter=(alias,))
    calls = []
    original = Path.stat
    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', counted)
    assert data.size == expected
    assert calls == [first, second, alias]
    with second.open('ab') as handle:
        handle.write(b'changed bytes')
    assert data.size == expected + len(b'changed bytes')
    assert calls == [first, second, alias] * 2


def test_digest_walk_is_local_and_hardlinked_roles_remain_distinct(manager, artifact, monkeypatch):
    path = artifact('llm.gguf')
    aliases = [path.with_name(name + '.gguf') for name in ('projector', 'adapter', 'draft')]
    for alias in aliases:
        os.link(path, alias)
    data = ModelArtifacts((path,), (aliases[0],), (aliases[1],), (aliases[2],))
    spec = parse_model_toml('[llm]\nmodel="Local/llm.gguf"\n')
    digest = sha256(path.read_bytes()).hexdigest()
    hashes, walks = [], []
    def measured(file):
        hashes.append(file)
        return {'content_sha256': digest}
    original = ModelArtifacts.files.fget
    def files(value):
        walks.append(value)
        return original(value)
    monkeypatch.setattr(manager, '_get_content_digest_sync', measured)
    monkeypatch.setattr(ModelArtifacts, 'files', property(files))
    expected = model_digest(digest, digest, parameter_changes(spec, {}, adapter_digest=digest), draft=digest)
    assert manager._artifact_digest(spec, data, {}) == expected
    assert hashes == [path] and walks == [data]


def test_changed_hardlink_alias_is_a_model_error(manager, artifact, monkeypatch):
    path = artifact('llm.gguf')
    alias = path.with_name('draft.gguf')
    os.link(path, alias)
    data = ModelArtifacts((path,), draft=(alias,))
    spec = parse_model_toml('[llm]\nmodel="Local/llm.gguf"\n')
    def measured(file):
        alias.unlink()
        alias.write_bytes(b'new independent file')
        return {'content_sha256': sha256(file.read_bytes()).hexdigest()}
    monkeypatch.setattr(manager, '_get_content_digest_sync', measured)
    with pytest.raises(TomlModelError, match='aliases changed'):
        manager._artifact_digest(spec, data, {})


def test_projector_overrides_use_the_metadata_that_validated_it(manager, artifact, toml_file, monkeypatch):
    artifact('model.gguf')
    projector = artifact('projector.gguf', 'clip', **{'clip.has_vision_encoder': 1})
    define(toml_file, projector=projector.name)
    calls = []
    original = manager._read_metadata_for_path
    def read(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(manager, '_read_metadata_for_path', read)
    result = manager._model_metadata_overrides('model')
    assert result['has_projector'] and result['vision_configured']
    assert result['projector_info']['general.architecture'] == 'clip'
    assert calls == [projector]


@pytest.mark.parametrize('phase', ['inventory', 'identity'])
async def test_cancelled_listing_joins_owned_thread_before_return(manager, artifact, toml_file, monkeypatch, phase):
    artifact('model.gguf')
    define(toml_file)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    name = '_list_local_models_sync' if phase == 'inventory' else '_listing_identity'
    original = getattr(manager, name)
    def work(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(manager, name, work)
    contexts = []
    async def listing():
        try:
            return await manager.list_local_models_with_metadata()
        finally:
            contexts.append(manager._definition_reads.get())
    task = asyncio.create_task(listing())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert await asyncio.sleep(0, result='responsive') == 'responsive'
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set() and contexts == [None]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    [row] = await manager.list_local_models_with_metadata()
    assert row['id'] == 'model'
