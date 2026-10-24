"""One shared repository cache must not depend on its mount point."""

from hashlib import sha256
import json
import os

import pytest

from tllama.helpers import metadata_cache as cache


@pytest.fixture(params=['metadata', 'digest'])
def entry(request):
    if request.param == 'metadata':
        return (cache.save_metadata_cache, cache.load_metadata_cache,
                cache.get_metadata_cache_path, cache.delete_metadata_cache,
                '.json', {'arch': 'qwen3'})
    return (cache.save_digest_cache, cache.load_digest_cache,
            cache.get_digest_cache_path, cache.delete_digest_cache,
            '.digest.json',
            {'content_sha256': 'a' * 64, 'source': 'local'})


def store(root):
    weights = root / 'Local' / 'můj model.gguf'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'GGUF shared weights')
    directory = root / '.tllama' / 'metadata-cache'
    directory.mkdir(parents=True)
    return directory, weights


def test_distinct_roots_reuse_one_shared_entry(tmp_path, entry):
    save, load, path_for, _, suffix, payload = entry
    directory, weights = store(tmp_path / 'server-models')
    other = tmp_path / 'client-mount'
    mirrored = other / 'Local' / weights.name
    mirrored.parent.mkdir(parents=True)
    # Two paths to the same weights and cache, without root-only bind mounts.
    os.link(weights, mirrored)
    (other / '.tllama').mkdir()
    shared = other / '.tllama' / 'metadata-cache'
    shared.symlink_to(directory, target_is_directory=True)

    save(directory, 'model', weights, payload)
    assert load(shared, mirrored) == payload
    expected = sha256(('Local/' + weights.name).encode()).hexdigest() + suffix
    assert path_for(directory, weights).name == path_for(shared, mirrored).name == expected
    save(shared, 'other alias', mirrored, payload)
    assert list(directory.iterdir()) == [directory / expected]
    assert json.loads((directory / expected).read_text())['model']['path'] == 'Local/' + weights.name
    assert load(directory, weights) == payload


def test_moving_repository_keeps_entries_valid(tmp_path, entry):
    save, load, path_for, _, _, payload = entry
    root = tmp_path / 'original'
    directory, weights = store(root)
    save(directory, 'model', weights, payload)
    filename = path_for(directory, weights).name
    root.rename(tmp_path / 'moved')
    directory = tmp_path / 'moved' / '.tllama' / 'metadata-cache'
    weights = tmp_path / 'moved' / 'Local' / weights.name
    assert path_for(directory, weights).name == filename
    assert load(directory, weights) == payload


def test_repository_symlink_uses_same_identity(tmp_path, entry):
    save, load, path_for, _, _, payload = entry
    root = tmp_path / 'models'
    directory, weights = store(root)
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    alias_cache = alias / '.tllama' / 'metadata-cache'
    alias_weights = alias / 'Local' / weights.name
    save(directory, 'model', weights, payload)
    assert path_for(directory, weights).name == path_for(alias_cache, alias_weights).name
    assert load(alias_cache, alias_weights) == payload


@pytest.mark.parametrize('change', ['size', 'mtime'])
def test_relative_entry_still_invalidates_changed_weights(tmp_path, entry, change):
    save, load, _, _, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    if change == 'size':
        weights.write_bytes(b'different and longer weights')
    else:
        before = weights.stat()
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    assert load(directory, weights) is None


def test_captured_fingerprint_survives_delayed_write(tmp_path, entry):
    save, load, path_for, _, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    captured = cache.build_model_file_fingerprint(weights)
    weights.write_bytes(b'new and substantially longer weights after scan')
    save(directory, 'model', weights, payload, fingerprint=captured)
    stored = json.loads(path_for(directory, weights).read_text())['model']
    assert stored['size'] == captured['size']
    assert stored['mtime_ns'] == captured['mtime_ns']
    assert stored['path'] == 'Local/' + weights.name
    assert load(directory, weights) is None


def test_absolute_path_entries_are_ignored_and_left_untouched(tmp_path, entry):
    save, load, path_for, _, suffix, payload = entry
    directory, weights = store(tmp_path / 'models')
    section = 'metadata' if suffix == '.json' else 'digest'
    version = cache.SCHEMA_VERSION if section == 'metadata' else cache.DIGEST_SCHEMA_VERSION
    old = directory / (sha256(str(weights.resolve()).encode()).hexdigest() + suffix)
    document = json.dumps({'schema_version': version,
                           'model': cache.build_model_file_fingerprint(weights), section: payload})
    old.write_text(document)
    assert load(directory, weights) is None
    assert not path_for(directory, weights).exists()
    save(directory, 'model', weights, payload)
    assert load(directory, weights) == payload
    assert old.read_text() == document


def test_deleting_relative_entry_removes_only_that_section(tmp_path, entry):
    save, load, path_for, delete, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    delete(directory, weights)
    assert not path_for(directory, weights).exists()
    assert load(directory, weights) is None


def test_same_basename_in_distinct_directories_does_not_collide(tmp_path, entry):
    save, load, path_for, _, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    other = weights.parent.parent / 'TLlama' / 'author' / weights.name
    other.parent.mkdir(parents=True)
    other.write_bytes(weights.read_bytes())
    save(directory, 'model', weights, payload)
    save(directory, 'author/model', other, {'other': True})
    assert path_for(directory, weights) != path_for(directory, other)
    assert load(directory, weights) == payload
    assert load(directory, other) == {'other': True}


def test_outside_weights_are_not_cached(tmp_path, entry):
    save, load, _, _, _, payload = entry
    directory, _ = store(tmp_path / 'models')
    outside = tmp_path / 'outside.gguf'
    outside.write_bytes(b'outside')
    save(directory, 'model', outside, payload)
    assert load(directory, outside) is None
    assert list(directory.iterdir()) == []


async def test_second_manager_lists_shared_repository_without_rescanning_or_hashing(tmp_path, monkeypatch):
    from tllama.backend import ModelManager
    from tllama.config import BackendConfig

    root = tmp_path / 'server'
    directory, weights = store(root)
    definition = '[llm]\nmodel = "Local/' + weights.name + '"\n'
    (root / 'Local' / 'model.toml').write_text(definition)
    other = tmp_path / 'mount'
    (other / 'Local').mkdir(parents=True)
    os.link(weights, other / 'Local' / weights.name)
    (other / 'Local' / 'model.toml').write_text(definition)
    (other / '.tllama').mkdir()
    (other / '.tllama' / 'metadata-cache').symlink_to(directory, target_is_directory=True)
    managers = [ModelManager(BackendConfig(models_dir=str(path))) for path in (root, other)]

    monkeypatch.setattr(managers[0], '_get_model_metadata_sync', lambda path: {'arch': 'qwen3'})
    [first] = await managers[0].list_local_models_with_metadata()

    def fail(path):
        raise AssertionError('second daemon reread the GGUF')

    monkeypatch.setattr(managers[1], '_get_model_metadata_sync', fail)
    monkeypatch.setattr(managers[1], '_compute_content_sha256', fail)
    [second] = await managers[1].list_local_models_with_metadata()
    assert second['id'] == first['id'] == 'model'
    assert second['arch'] == first['arch'] == 'qwen3'
    assert second['digest'] == first['digest'] == sha256(weights.read_bytes()).hexdigest()
    assert len(list(directory.iterdir())) == 2  # Header and digest, shared by both daemons.
