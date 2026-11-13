"""Warm blob reuse avoids repeated headers/writes, while cold checks stay real."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tllama import repository_blobs, repository_runtime
from tllama.helpers import gguf_import, metadata_cache
from tllama.model_definitions import create_definition
from tllama.model_uploads import has_blob
from tllama.repository_store import prepare_group
from test_published_blobs import sources


@pytest.fixture
def count_work():
    def measure(operation):
        counts = {'headers': 0, 'cache_writes': 0}
        raw, temp = gguf_import.read_gguf_metadata, metadata_cache.repository_tempfile
        def read(*args, **kwargs):
            counts['headers'] += 1
            return raw(*args, **kwargs)
        def write(*args, **kwargs):
            counts['cache_writes'] += 1
            return temp(*args, **kwargs)
        with ExitStack() as stack:
            for obj, name, value in ((gguf_import, 'read_gguf_metadata', read),
                    (repository_blobs, 'read_gguf_metadata', read),
                    (repository_runtime, 'read_gguf_metadata', read),
                    (metadata_cache, 'repository_tempfile', write)):
                stack.enter_context(patch.object(obj, name, value))
            result = operation()
        return result, counts
    return measure


@pytest.mark.parametrize('count', [2, 4, 8])
def test_warm_heads_do_not_read_headers_or_rewrite_cache_and_create_work_is_linear(manager, tmp_path, count_work, count):
    original = sources(tmp_path, count)
    group = prepare_group(original)
    paths = manager.repository.publish(group)
    files = [SimpleNamespace(filename=path.name, digest='sha256:' + digest)
             for path, digest in zip(original, group.digests)]
    result, head = count_work(lambda: [has_blob(manager, item.digest) for item in files])
    assert all(result) and head == {'headers': 0, 'cache_writes': 0}
    result, created = count_work(lambda: create_definition(manager, 'reused', '[llm]\nfrom="client.gguf"', {'llm': files}))
    assert result['status'] == 'success'
    assert created['headers'] <= 2 * count + 2 and created['cache_writes'] == 0
    assert all(path.read_bytes() == old.read_bytes() for path, old in zip(paths, original))


@pytest.mark.parametrize('count', [1, 4])
def test_cold_header_cache_is_read_and_written_once_then_warm(manager, tmp_path, count_work, count):
    group = prepare_group(sources(tmp_path, count))
    paths = manager.repository.publish(group)
    for path in paths:
        metadata_cache.get_metadata_cache_path(manager.metadata_cache_dir, path).unlink()
    def heads():
        return [has_blob(manager, 'sha256:' + digest) for digest in group.digests]
    results, cold = count_work(heads)
    assert all(results) and cold == {'headers': count, 'cache_writes': count}
    results, warm = count_work(heads)
    assert all(results) and warm == {'headers': 0, 'cache_writes': 0}


@pytest.mark.parametrize('fault', ['missing-fields', 'wrong-count', 'wrong-number', 'unknown-architecture'])
def test_unusable_cached_shard_header_falls_back_to_actual_bytes(manager, tmp_path, count_work, fault):
    group = prepare_group(sources(tmp_path, 2))
    paths = manager.repository.publish(group)
    metadata = metadata_cache.load_metadata_cache(manager.metadata_cache_dir, paths[0])
    if fault == 'missing-fields':
        metadata['metadata_raw'] = {}
    elif fault == 'wrong-count':
        metadata['metadata_raw']['split.count'] = '2'
    elif fault == 'wrong-number':
        metadata['metadata_raw']['split.no'] = 1
    else:
        metadata['metadata_raw'].pop('general.architecture')
    metadata_cache.save_metadata_cache(manager.metadata_cache_dir, 'bad', paths[0], metadata)
    result, work = count_work(lambda: has_blob(manager, 'sha256:' + group.digests[0]))
    assert result and work == {'headers': 1, 'cache_writes': 1}
    assert metadata_cache.load_metadata_cache(manager.metadata_cache_dir, paths[0])['metadata_raw']['split.no'] == 0


@pytest.mark.parametrize('fault', ['missing-shard', 'changed-bytes', 'reordered-files', 'wrong-names'])
def test_exact_group_shortcut_does_not_bypass_integrity(manager, tmp_path, fault):
    original = sources(tmp_path, 2)
    group = prepare_group(original)
    paths = manager.repository.publish(group)
    files = [SimpleNamespace(filename=path.name, digest='sha256:' + digest)
             for path, digest in zip(original, group.digests)]
    if fault == 'missing-shard':
        paths[1].unlink()
    elif fault == 'changed-bytes':
        paths[1].write_bytes(paths[1].read_bytes() + b'changed')
    elif fault == 'reordered-files':
        files.reverse()
    else:
        files[1].filename = 'not-a-continuation.gguf'
    with pytest.raises((ValueError, FileNotFoundError)):
        create_definition(manager, 'bad', '[llm]\nfrom="client.gguf"', {'llm': files})
    assert not (manager.local_models_dir / 'bad.toml').exists()


def test_new_original_names_are_recorded_when_existing_group_skips_republication(manager, tmp_path):
    original = sources(tmp_path, 2)
    group = prepare_group(original)
    paths = manager.repository.publish(group)
    files = [SimpleNamespace(filename=f'another-{i+1:05d}-of-00002.gguf', digest='sha256:' + digest)
             for i, digest in enumerate(group.digests)]
    create_definition(manager, 'renamed', '[llm]\nfrom="client.gguf"', {'llm': files})
    for path, old, item in zip(paths, original, files):
        assert path.with_suffix('.inf').read_text().splitlines() == [old.name, item.filename]


def test_valid_digest_cache_extra_fields_do_not_cause_rewrite(manager, tmp_path, count_work):
    group = prepare_group(sources(tmp_path))
    path, = manager.repository.publish(group)
    metadata_cache.save_digest_cache(manager.metadata_cache_dir, 'extended', path,
        {'content_sha256': group.digests[0], 'source': 'local', 'annotation': 'preserve'})
    cache = metadata_cache.get_digest_cache_path(manager.metadata_cache_dir, path)
    old = cache.read_bytes()
    result, work = count_work(lambda: has_blob(manager, 'sha256:' + group.digests[0]))
    assert result and work == {'headers': 0, 'cache_writes': 0}
    assert cache.read_bytes() == old


def test_file_changed_after_exact_group_resolution_is_not_published(manager, tmp_path, monkeypatch):
    original = sources(tmp_path, 2)
    group = prepare_group(original)
    paths = manager.repository.publish(group)
    files = [SimpleNamespace(filename=path.name, digest='sha256:' + digest)
             for path, digest in zip(original, group.digests)]
    resolve = repository_blobs.published_group
    def change(*args, **kwargs):
        prepared = resolve(*args, **kwargs)
        paths[1].write_bytes(paths[1].read_bytes() + b'changed-after-resolution')
        return prepared
    monkeypatch.setattr(repository_blobs, 'published_group', change)
    with pytest.raises(ValueError, match='changed'):
        create_definition(manager, 'bad', '[llm]\nfrom="client.gguf"', {'llm': files})
    assert not (manager.local_models_dir / 'bad.toml').exists()
