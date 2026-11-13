"""Published bytes replace upload aliases without breaking blob/create reuse."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.client_import import create_from_file
from tllama.helpers.metadata_cache import get_digest_cache_path, get_metadata_cache_path
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import create_definition
from tllama.model_uploads import blob_path, has_blob, verified_blob
from tllama.repository_blobs import remove_published_uploads
from tllama.repository_pins import Reservation
from tllama.repository_store import prepare_group
from test_model_create import create_client, gguf_file


def sources(tmp_path, count=1, arch='llama'):
    return [gguf_file(tmp_path / ('original.gguf' if count == 1 else f'original-{i+1:05d}-of-{count:05d}.gguf'),
        arch=arch, shard=i if count > 1 else None, count=count if count > 1 else None) for i in range(count)]


def staged(manager, paths):
    uploads = []
    for path in paths:
        digest = 'sha256:' + sha256(path.read_bytes()).hexdigest()
        blob = blob_path(manager, digest)
        blob.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, blob)
        uploads.append(SimpleNamespace(filename=path.name, digest=digest))
    return uploads


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('role,arch', [('llm', 'llama'), ('mmproj', 'clip'), ('draft', 'dflash')])
async def test_create_unlinks_aliases_but_repeat_import_needs_no_upload_or_rehash(
        create_client, manager, tmp_path, monkeypatch, count, role, arch):
    paths = sources(tmp_path, count, arch)
    definition = tmp_path / 'import.toml'
    definition.write_text(f'[{role}]\nfrom={json.dumps(str(paths[0]))}\n')
    await create_from_file(create_client, 'first', definition, as_json=True)
    assert list((manager.models_dir / '.tllama/uploads').iterdir()) == []
    spec = parse_model_toml(manager.read_model_definition('first'))
    target = manager.models_dir / spec.primary_model
    assert target.with_suffix('.inf').read_text().splitlines() == [paths[0].name]
    inodes = [p.stat().st_ino for p in sorted(target.parent.glob('*.gguf'))]
    async def no_upload(*args):
        raise AssertionError('Published bytes must satisfy the blob HEAD/create protocol')
    def no_hash(*args):
        raise AssertionError('Warm locally measured digest caches must avoid full weight reads')
    monkeypatch.setattr(create_client, 'upload_blob', no_upload)
    monkeypatch.setattr('tllama.repository_store.hash_file', no_hash)
    await create_from_file(create_client, 'second', definition, as_json=True)
    assert parse_model_toml(manager.read_model_definition('second')).primary_model == spec.primary_model
    assert [p.stat().st_ino for p in sorted(target.parent.glob('*.gguf'))] == inodes
    assert all(p.stat().st_nlink == 1 for p in target.parent.glob('*.gguf'))
    assert list((manager.models_dir / '.tllama/uploads').iterdir()) == []


@pytest.mark.parametrize('count', [1, 2])
def test_cold_digest_caches_are_measured_once_then_reused(manager, tmp_path, monkeypatch, count):
    paths = manager.repository.publish(prepare_group(sources(tmp_path, count)))
    digests = ['sha256:' + sha256(p.read_bytes()).hexdigest() for p in paths]
    for path in paths:
        get_digest_cache_path(manager.metadata_cache_dir, path).unlink()
    from tllama.repository_store import hash_file
    calls = []
    def measured(path):
        calls.append(path)
        return hash_file(path)
    monkeypatch.setattr('tllama.repository_store.hash_file', measured)
    for digest, path in zip(digests, paths):
        assert has_blob(manager, digest)
        assert verified_blob(manager, digest)[0] == path
    assert calls == list(paths)


@pytest.mark.parametrize('damage', ['missing-shard', 'changed-bytes', 'wrong-address', 'symlink'])
def test_unverified_or_incomplete_published_group_is_not_a_blob(manager, tmp_path, damage):
    paths = manager.repository.publish(prepare_group(sources(tmp_path, 2)))
    digest = 'sha256:' + sha256(paths[0].read_bytes()).hexdigest()
    if damage == 'missing-shard':
        paths[1].unlink()
    elif damage == 'changed-bytes':
        paths[0].write_bytes(paths[0].read_bytes() + b'changed')
    elif damage == 'wrong-address':
        paths[0].parent.rename(paths[0].parent.with_name('0' * 64))
    else:
        original = tmp_path / 'outside.gguf'
        paths[0].rename(original)
        paths[0].symlink_to(original)
    assert not has_blob(manager, digest)
    with pytest.raises(FileNotFoundError):
        verified_blob(manager, digest)


@pytest.mark.parametrize('count', [1, 2])
def test_old_upload_aliases_are_removed_by_content_after_data_inode_replacement(manager, tmp_path, count):
    original = sources(tmp_path, count)
    uploads = staged(manager, original)
    group = prepare_group([blob_path(manager, u.digest) for u in uploads], names=[p.name for p in original])
    paths = manager.repository.publish(group, link_owned=True)
    for path in paths:
        replacement = path.with_suffix('.replacement')
        shutil.copy2(path, replacement)
        replacement.replace(path)  # Same bytes, different inode/filesystem placement.
    assert all(p.stat().st_ino != blob_path(manager, u.digest).stat().st_ino for p, u in zip(paths, uploads))
    assert remove_published_uploads(manager.repository) == []
    assert all(not blob_path(manager, u.digest).exists() for u in uploads)
    assert all(p.read_bytes() == source.read_bytes() for p, source in zip(paths, original))
    assert all(p.with_suffix('.inf').is_file() for p in paths)


def test_failed_definition_keeps_staging_for_retry_then_success_removes_it(manager, tmp_path, monkeypatch):
    uploads = staged(manager, sources(tmp_path))
    from tllama.model_definitions import write_model_toml
    def fail(*args, **kwargs):
        raise OSError('definition publication failed')
    monkeypatch.setattr('tllama.model_definitions.write_model_toml', fail)
    with pytest.raises(OSError, match='publication failed'):
        create_definition(manager, 'model', '[llm]\nfrom="original.gguf"\n', {'llm': uploads})
    assert blob_path(manager, uploads[0].digest).is_file()
    monkeypatch.setattr('tllama.model_definitions.write_model_toml', write_model_toml)
    assert create_definition(manager, 'model', '[llm]\nfrom="original.gguf"\n', {'llm': uploads})['status'] == 'success'
    assert not blob_path(manager, uploads[0].digest).exists()


def test_active_upload_pin_defers_alias_cleanup_then_retry_removes_it(manager, tmp_path):
    uploads = staged(manager, sources(tmp_path))
    with Reservation(manager.repository, 'other upload', paths=['.tllama/uploads']):
        create_definition(manager, 'model', '[llm]\nfrom="original.gguf"\n', {'llm': uploads})
        assert blob_path(manager, uploads[0].digest).is_file()
    assert remove_published_uploads(manager.repository) == []
    assert not blob_path(manager, uploads[0].digest).exists()
    assert has_blob(manager, uploads[0].digest)


def test_unlink_failure_is_warning_after_success_and_retryable(manager, tmp_path, monkeypatch):
    uploads = staged(manager, sources(tmp_path))
    target = blob_path(manager, uploads[0].digest)
    unlink = Path.unlink
    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError('test denial')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    result = create_definition(manager, 'model', '[llm]\nfrom="original.gguf"\n', {'llm': uploads})
    assert result['status'] == 'success' and target.is_file()
    assert any('test denial' in warning for warning in result['cleanup']['warnings'])
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert remove_published_uploads(manager.repository) == [] and not target.exists()


def test_two_managers_create_from_one_staged_blob_without_second_upload(manager, tmp_path):
    uploads = staged(manager, sources(tmp_path))
    other = ModelManager(BackendConfig(models_dir=str(manager.models_dir)))
    def create(pair):
        owner, name = pair
        return create_definition(owner, name, '[llm]\nfrom="original.gguf"\n', {'llm': uploads})
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(create, [(manager, 'first'), (other, 'second')]))
    assert all(r['status'] == 'success' for r in results)
    assert not blob_path(manager, uploads[0].digest).exists()
    assert parse_model_toml(manager.read_model_definition('first')).primary_model == parse_model_toml(
        manager.read_model_definition('second')).primary_model


def test_unknown_upload_and_its_caches_are_left_for_explicit_offline_cleanup(manager, tmp_path):
    uploads = staged(manager, sources(tmp_path))
    target = blob_path(manager, uploads[0].digest)
    assert remove_published_uploads(manager.repository) == []
    assert target.is_file()


async def test_last_real_upload_release_removes_alias_deferred_by_create(create_client, manager, tmp_path):
    uploads = staged(manager, sources(tmp_path))
    other = gguf_file(tmp_path / 'other.gguf', arch='qwen3').read_bytes()
    digest = 'sha256:' + sha256(other).hexdigest()
    async def chunks():
        yield other[:30]
        await asyncio.to_thread(create_definition, manager, 'model',
            '[llm]\nfrom="original.gguf"\n', {'llm': uploads})
        assert blob_path(manager, uploads[0].digest).exists()
        yield other[30:]
    await create_client.upload_blob(digest, chunks())
    assert not blob_path(manager, uploads[0].digest).exists()
    assert blob_path(manager, digest).read_bytes() == other


async def test_post_checks_its_existing_staging_inode_even_when_data_is_valid(create_client, manager, tmp_path):
    from tllama.client import ClientError
    paths = sources(tmp_path)
    manager.repository.publish(prepare_group(paths))
    uploads = staged(manager, paths)
    blob_path(manager, uploads[0].digest).write_bytes(b'corrupt existing staging')
    with pytest.raises(ClientError, match='does not match'):
        await create_client.upload_blob(uploads[0].digest, paths[0].read_bytes())
    assert has_blob(manager, uploads[0].digest)


def test_cache_unlink_failure_keeps_alias_as_retry_key(manager, tmp_path, monkeypatch):
    uploads = staged(manager, sources(tmp_path))
    target = blob_path(manager, uploads[0].digest)
    manager.repository.publish(prepare_group([target], names=['original.gguf']))
    # verify_file warms this cache before cleanup tries to remove it.
    cache = get_digest_cache_path(manager.metadata_cache_dir, target)
    unlink = Path.unlink
    def denied(path, *args, **kwargs):
        if path == cache:
            raise PermissionError('cache unlink denial')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    assert remove_published_uploads(manager.repository) and target.exists()
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert remove_published_uploads(manager.repository) == []
    assert not target.exists() and not cache.exists()


def test_symlinked_cache_directory_cannot_redirect_alias_cache_deletion(manager, tmp_path):
    from test_repository_check import before
    uploads = staged(manager, sources(tmp_path))
    target = blob_path(manager, uploads[0].digest)
    manager.repository.publish(prepare_group([target], names=['original.gguf']))
    outside = tmp_path / 'outside-cache'
    manager.metadata_cache_dir.rename(outside)
    manager.metadata_cache_dir.symlink_to(outside, target_is_directory=True)
    old = before(outside)
    assert remove_published_uploads(manager.repository) and target.exists()
    assert before(outside) == old
