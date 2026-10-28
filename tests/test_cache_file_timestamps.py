"""Persistent validity follows the cache inode mtime, never JSON mtime_ns."""

import json
import os

import pytest

from test_repository_cache_identity import entry, store
from tllama.helpers import metadata_cache as cache


@pytest.mark.parametrize('timestamp', [0, 1_700_000_000_000_000_000, 1_700_000_000_123_456_789])
def test_cache_inode_carries_exact_model_time_and_json_has_no_mtime(tmp_path, entry, timestamp):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    os.utime(weights, ns=(timestamp, timestamp))
    save(directory, 'model', weights, payload)
    path = path_for(directory, weights)
    document = json.loads(path.read_text())
    assert document['cache_format_version'] == cache.CACHE_FORMAT_VERSION
    assert 'mtime_ns' not in document['model']
    assert path.stat().st_mtime_ns == weights.stat().st_mtime_ns == timestamp
    assert load(directory, weights) == payload


@pytest.mark.parametrize('offset', [1, 10, 100, 1_000_000, 1_000_000_000])
def test_any_visible_model_time_change_invalidates_without_tolerance(tmp_path, entry, offset):
    save, load, _, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    timestamp = 1_700_000_000_000_000_000
    os.utime(weights, ns=(timestamp, timestamp))
    save(directory, 'model', weights, payload)
    os.utime(weights, ns=(timestamp + offset, timestamp + offset))
    assert load(directory, weights) is None


def test_cache_timestamp_change_invalidates_even_when_payload_is_unchanged(tmp_path, entry):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    path = path_for(directory, weights)
    before = path.read_bytes()
    timestamp = weights.stat().st_mtime_ns
    os.utime(path, ns=(timestamp, timestamp + 1))
    assert path.read_bytes() == before
    assert load(directory, weights) is None


def test_json_timestamp_cannot_make_a_different_cache_inode_valid(tmp_path, entry):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    path = path_for(directory, weights)
    document = json.loads(path.read_text())
    document['model']['mtime_ns'] = weights.stat().st_mtime_ns
    path.write_text(json.dumps(document))
    timestamp = weights.stat().st_mtime_ns + 1
    os.utime(path, ns=(timestamp, timestamp))
    assert load(directory, weights) is None


def test_previous_cache_format_is_rejected_even_with_matching_file_time(tmp_path, entry):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    path = path_for(directory, weights)
    document = json.loads(path.read_text())
    document.pop('cache_format_version')
    document['model']['mtime_ns'] = weights.stat().st_mtime_ns
    path.write_text(json.dumps(document))
    timestamp = weights.stat().st_mtime_ns
    os.utime(path, ns=(timestamp, timestamp))
    assert load(directory, weights) is None


@pytest.mark.parametrize('operation', ['utime', 'replace'])
def test_failed_publication_keeps_old_cache_and_removes_temporary(tmp_path, entry, monkeypatch, operation):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    save(directory, 'model', weights, payload)
    path = path_for(directory, weights)
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    def fail(*args, **kwargs):
        raise PermissionError('injected cache publication failure')
    monkeypatch.setattr(cache.os, operation, fail)
    save(directory, 'changed alias', weights, {'wrong': True})
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == timestamp
    assert list(directory.iterdir()) == [path]
    assert load(directory, weights) == payload


def test_delayed_same_size_write_keeps_captured_file_timestamp(tmp_path, entry):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    captured = cache.build_model_file_fingerprint(weights)
    weights.write_bytes(b'x' * captured['size'])
    timestamp = captured['mtime_ns'] + 1
    os.utime(weights, ns=(timestamp, timestamp))
    save(directory, 'model', weights, payload, fingerprint=captured)
    assert path_for(directory, weights).stat().st_mtime_ns == captured['mtime_ns']
    assert load(directory, weights) is None


def test_reader_stats_the_open_inode_not_a_replacement_path(tmp_path, entry, monkeypatch):
    save, load, path_for, _, payload = entry
    directory, weights = store(tmp_path / 'models')
    timestamp = 1_700_000_000_123_456_789
    os.utime(weights, ns=(timestamp, timestamp))
    save(directory, 'old', weights, {'old': True})
    original_load = cache.json.load
    def replace_after_read(handle):
        document = original_load(handle)
        os.utime(weights, ns=(timestamp + 1, timestamp + 1))
        save(directory, 'new', weights, payload)
        return document
    # Before opening the old cache the reader sees the new model's timestamp.
    os.utime(weights, ns=(timestamp + 1, timestamp + 1))
    monkeypatch.setattr(cache.json, 'load', replace_after_read)
    assert load(directory, weights) is None
    monkeypatch.setattr(cache.json, 'load', original_load)
    assert load(directory, weights) == payload
