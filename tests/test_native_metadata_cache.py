"""Inference populates the fingerprinted header cache without an earlier show."""

import os

from tllama.helpers.metadata_cache import (
    build_model_file_fingerprint, load_metadata_cache, save_metadata_cache,
)

from test_native_fim import fim, body
from test_native_manager import native_manager
from test_native_public_api import public, request_body


async def test_reused_fim_worker_scans_header_once_and_weight_change_invalidates(fim, monkeypatch):
    manager, client, name, toml, weights, _ = fim
    scan = manager._get_model_metadata_sync
    scans = []
    def counted(path):
        scans.append(path)
        return scan(path)
    monkeypatch.setattr(manager, '_get_model_metadata_sync', counted)
    for stream in (False, True, False):
        response = await client.post('/api/generate', json=body(name, stream=stream))
        assert response.status_code == 200, response.text
    assert len(scans) == 1 and len(manager.test.ledger) == 1
    cached = load_metadata_cache(manager.metadata_cache_dir, weights)
    assert cached['fim']['tokens']['<|file_separator|>'] == 10

    # A definition-only change needs new sampling, not another GGUF scan.
    toml.write_text(toml.read_text().replace('temperature=0.4', 'temperature=0.8'))
    response = await client.post('/api/generate', json=body(name))
    assert response.status_code == 200 and len(scans) == 1

    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    response = await client.post('/api/generate', json=body(name))
    assert response.status_code == 200 and len(scans) == 2


async def test_failed_header_scan_is_not_cached_as_an_empty_success(public, monkeypatch):
    manager, client, name, _ = public
    scans = []
    def flaky(path):
        scans.append(path)
        if len(scans) == 1:
            raise OSError('temporary header read failure')
        return {'recommended_sampling':{'top_k':17}}
    monkeypatch.setattr(manager, '_get_model_metadata_sync', flaky)
    for _ in range(3):
        response = await client.post('/api/chat', json=request_body(name, '/api/chat'))
        assert response.status_code == 200
    assert len(scans) == 2
    cached = load_metadata_cache(manager.metadata_cache_dir, manager.local_models_dir/'a.gguf')
    assert cached == {'recommended_sampling':{'top_k':17}}


async def test_metadata_cache_write_failure_cannot_break_inference(fim, monkeypatch):
    manager, client, name, _, _, _ = fim
    blocked = manager.models_dir/'cache-is-a-file'
    blocked.write_text('not a directory')
    monkeypatch.setattr(manager, 'metadata_cache_dir', blocked)
    for _ in range(2):
        response = await client.post('/api/generate', json=body(name))
        assert response.status_code == 200 and response.json()['response'] == 'a * b'
    assert len(manager.test.ledger) == 1


def test_delayed_metadata_write_never_stamps_old_header_with_new_weight_fingerprint(tmp_path):
    weights = tmp_path/'m.gguf'
    weights.write_bytes(b'old weights')
    fingerprint = build_model_file_fingerprint(weights)
    weights.write_bytes(b'new, different weights')
    save_metadata_cache(tmp_path/'cache', 'm', weights, {'fim':'old inventory'},
                        fingerprint=fingerprint)
    assert load_metadata_cache(tmp_path/'cache', weights) is None
