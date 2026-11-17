"""Only stable definitions are shared within one listing/metadata request."""

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import threading

import pytest

from tllama import backend
from tllama.helpers.model_toml import TomlModelError
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_store import prepare_group
from test_model_create import gguf_file


@pytest.fixture
def repo(manager, tmp_path):
    paths = [gguf_file(tmp_path / f'model-{i+1:05d}-of-00002.gguf', shard=i, count=2)
             for i in range(2)]
    first, _ = manager.repository.publish(prepare_group(paths))
    text = f'[llm]\nmodel={json.dumps(first.relative_to(manager.models_dir).as_posix())}\n'
    text += '[system]\nprompt="old"\n[sampling]\ntop_k=10\n'
    create_definition(manager, 'a', text, {})
    copy_definition(manager, 'a', 'b')
    return manager


@pytest.fixture
def parses(monkeypatch):
    calls = []
    parse = backend.parse_model_toml
    def counted(text, **kwargs):
        calls.append((Path(kwargs['source']).name, threading.get_ident()))
        return parse(text, **kwargs)
    monkeypatch.setattr(backend, 'parse_model_toml', counted)
    return calls


def rewrite(path, text, *, atomic=False, keep_mtime=False):
    before = path.stat()
    if atomic:
        other = path.with_suffix('.replacement')
        other.write_text(text)
        if keep_mtime:
            os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    else:
        path.write_text(text)
        if keep_mtime:
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


async def test_one_parse_per_definition_per_listing_with_shared_shards(repo, parses):
    loop_thread = threading.get_ident()
    for _ in range(2):
        parses.clear()
        rows = await repo.list_local_models_with_metadata()
        assert {row['id'] for row in rows} == {'a', 'b'}
        assert len({row['digest'] for row in rows}) == 1
        assert Counter(name for name, _ in parses) == {'a.toml': 1, 'b.toml': 1}
        assert all(thread != loop_thread for _, thread in parses)
        assert repo._definition_reads.get() is None


async def test_metadata_scope_does_not_cache_across_requests_or_mutate_toml(repo, parses):
    path = repo.local_models_dir / 'a.toml'
    original = path.read_text()
    first = await repo.get_model_metadata('a')
    assert first['sampling_defaults']['top_k'] == 10
    # Caller-owned response changes must not affect the next parsed definition.
    first['sampling_defaults']['top_k'] = 99
    assert (await repo.get_model_metadata('a'))['sampling_defaults']['top_k'] == 10
    assert path.read_text() == original
    assert [name for name, _ in parses] == ['a.toml', 'a.toml']


@pytest.mark.parametrize('atomic', [False, True])
@pytest.mark.parametrize('keep_mtime', [False, True])
async def test_edit_between_metadata_phases_invalidates_same_size_definition(repo, monkeypatch, parses,
                                                                          atomic, keep_mtime):
    path = repo.local_models_dir / 'a.toml'
    work = repo._model_metadata_overrides
    def edit(name):
        rewrite(path, path.read_text().replace('"old"', '"new"').replace('top_k=10', 'top_k=20'),
                atomic=atomic, keep_mtime=keep_mtime)
        return work(name)
    monkeypatch.setattr(repo, '_model_metadata_overrides', edit)
    result = await repo.get_model_metadata('a')
    assert result['default_system_prompt'] == 'new'
    assert result['sampling_defaults']['top_k'] == 20
    assert [name for name, _ in parses] == ['a.toml', 'a.toml']
    assert repo._definition_reads.get() is None


async def test_definition_edited_during_parse_is_retried_before_reuse(repo, monkeypatch):
    path = repo.local_models_dir / 'a.toml'
    parse = backend.parse_model_toml
    calls = []
    def changing(text, **kwargs):
        spec = parse(text, **kwargs)
        calls.append(text)
        if len(calls) == 1:
            path.write_text(text.replace('"old"', '"new"'))
        return spec
    monkeypatch.setattr(backend, 'parse_model_toml', changing)
    result = await repo.get_model_metadata('a')
    assert result['default_system_prompt'] == 'new'
    assert len(calls) == 2


async def test_unstable_definition_is_rejected_without_retaining_scope(repo, monkeypatch):
    path = repo.local_models_dir / 'a.toml'
    parse = backend.parse_model_toml
    calls = []
    def changing(text, **kwargs):
        calls.append(text)
        path.write_text(text + '# next revision\n')
        return parse(text, **kwargs)
    monkeypatch.setattr(backend, 'parse_model_toml', changing)
    with pytest.raises(TomlModelError, match='kept changing'):
        await repo.get_model_metadata('a')
    assert len(calls) == 3
    assert repo._definition_reads.get() is None


async def test_parse_error_after_inventory_is_isolated_and_next_request_recovers(repo, monkeypatch):
    path = repo.local_models_dir / 'a.toml'
    original = path.read_text()
    metadata = repo.get_model_metadata
    async def break_a(name):
        if name == 'a':
            path.write_text('[broken')
        return await metadata(name)
    monkeypatch.setattr(repo, 'get_model_metadata', break_a)
    assert [row['id'] for row in await repo.list_local_models_with_metadata()] == ['b']
    assert repo._definition_reads.get() is None
    path.write_text(original)
    monkeypatch.setattr(repo, 'get_model_metadata', metadata)
    assert {row['id'] for row in await repo.list_local_models_with_metadata()} == {'a', 'b'}


def test_delete_and_recreate_inside_scope_never_returns_previous_spec(repo):
    path = repo.local_models_dir / 'a.toml'
    text = path.read_text()
    with repo._definition_read_scope():
        assert repo._resolve_virtual_model_spec('a').system_prompt == 'old'
        path.unlink()
        assert repo._resolve_virtual_model_spec('a') is None
        path.write_text(text.replace('"old"', '"new"'))
        assert repo._resolve_virtual_model_spec('a').system_prompt == 'new'
    assert repo._definition_reads.get() is None


async def test_symlink_loop_remains_a_missing_definition(repo):
    path = repo.local_models_dir / 'a.toml'
    path.unlink()
    path.symlink_to(path.name)
    assert await repo.get_model_metadata('a') is None


async def test_overlapping_requests_have_distinct_scopes_and_cancel_clears_owner(repo, monkeypatch, parses):
    entered, release = asyncio.Event(), asyncio.Event()
    scopes = []
    metadata = repo._get_raw_model_metadata
    async def wait_a(name):
        value = await metadata(name)
        scopes.append(repo._definition_reads.get())
        if len(scopes) == 1:
            entered.set()
            await release.wait()
        return value
    monkeypatch.setattr(repo, '_get_raw_model_metadata', wait_a)
    first = asyncio.create_task(repo.get_model_metadata('a'))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert repo._definition_reads.get() is None
        assert (await repo.get_model_metadata('a'))['default_system_prompt'] == 'old'
        assert scopes[0] is not scopes[1] and all(scope is not None for scope in scopes)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert repo._definition_reads.get() is None
        assert (await repo.get_model_metadata('a'))['default_system_prompt'] == 'old'
        assert Counter(name for name, _ in parses) == {'a.toml': 3}
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
