"""HEAD parity and offline model information from real tiny GGUF headers."""

from datetime import datetime, timezone
import json
import os

import gguf
import httpx
import pytest

from tllama.main import app
from tllama.routers import ollama
from tllama.helpers.metadata_cache import load_metadata_cache, get_metadata_cache_path


@pytest.fixture
async def information(manager, monkeypatch):
    def gguf_file(name, arch='llama', **fields):
        path = manager.local_models_dir / name
        writer = gguf.GGUFWriter(str(path), arch=arch)
        for key, value in fields.items():
            if type(value) is bool:
                writer.add_bool(key, value)
            else:
                writer.add_string(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        return path

    weights = gguf_file('info.gguf', **{'general.license': 'apache-2.0',
                                      'general.license.link': 'https://example.test/license'})
    definition = manager.local_models_dir / 'info.toml'
    definition.write_text('[llm]\nmodel="Local/info.gguf"\n[system]\nprompt="TOML system"\n')
    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://info') as client:
        yield manager, client, weights, definition, gguf_file


@pytest.mark.parametrize('path', ['/api/version', '/api/tags'])
async def test_head_matches_get_headers_without_a_body_or_worker(information, path):
    manager, client, _, _, _ = information
    get = await client.get(path)
    head = await client.head(path)
    assert get.status_code == head.status_code == 200
    assert head.content == b''
    for header in ['content-type', 'content-length']:
        assert head.headers[header] == get.headers[header]
    assert not manager.native_snapshot()


async def test_show_reports_definition_system_license_and_timestamp_without_inference(information):
    manager, client, weights, definition, _ = information
    original = definition.read_text()
    os.utime(definition, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    first = (await client.post('/api/show', json={'model': 'info'})).json()
    assert first['system'] == 'TOML system' and first['license'] == 'apache-2.0'
    assert first['model_info']['general.license.link'] == 'https://example.test/license'
    assert datetime.fromisoformat(first['modified_at']) == datetime.fromtimestamp(1_700_000_000, timezone.utc)
    assert first['modelfile'] == original and first['projector_info'] == {}
    cached = load_metadata_cache(manager.metadata_cache_dir, weights)
    assert cached['metadata_raw']['general.license'] == 'apache-2.0'

    definition.write_text(original.replace('TOML system', 'Changed'))
    second = (await client.post('/api/show', json={'name': 'info'})).json()
    assert second['system'] == 'Changed' and second['modified_at'] != first['modified_at']
    assert load_metadata_cache(manager.metadata_cache_dir, weights) == cached
    assert not manager.native_snapshot()


async def test_show_reads_configured_projector_metadata_and_clears_it_after_definition_edit(information):
    manager, client, _, definition, write_gguf = information
    write_gguf('projector.gguf', 'clip', **{'general.name': 'Actual projector', 'clip.has_vision_encoder': True})
    original = definition.read_text()
    definition.write_text(original + '\n[mmproj]\nmodel="Local/projector.gguf"\n')
    first = (await client.post('/api/show', json={'model': 'info'})).json()
    assert first['projector_info']['general.name'] == 'Actual projector'
    assert first['projector_info']['general.architecture'] == 'clip'
    assert 'vision' in first['capabilities']
    definition.write_text(original)
    second = (await client.post('/api/show', json={'model': 'info'})).json()
    assert second['projector_info'] == {} and 'vision' not in second['capabilities']
    assert not manager.native_snapshot()


async def test_previous_header_cache_is_rebuilt_for_license_without_touching_weights(information):
    manager, client, weights, _, _ = information
    await client.post('/api/show', json={'model': 'info'})
    path = get_metadata_cache_path(manager.metadata_cache_dir, weights)
    old = json.loads(path.read_text())
    old['schema_version'] = 7
    old['metadata']['metadata_raw'].pop('general.license')
    path.write_text(json.dumps(old))
    manager._invalidate_metadata_cache_entry('info')
    response = await client.post('/api/show', json={'model': 'info'})
    assert response.status_code == 200 and response.json()['license'] == 'apache-2.0'
    assert json.loads(path.read_text())['schema_version'] == 8


async def test_missing_license_is_not_fabricated(information):
    manager, client, _, _, write_gguf = information
    write_gguf('unlicensed.gguf')
    (manager.local_models_dir / 'unlicensed.toml').write_text('[llm]\nmodel="Local/unlicensed.gguf"\n')
    response = await client.post('/api/show', json={'model': 'unlicensed'})
    assert response.status_code == 200 and response.json()['license'] == ''
