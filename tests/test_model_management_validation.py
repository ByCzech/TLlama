"""Malformed management inputs fail before filesystem/network side effects."""

from unittest.mock import AsyncMock

import httpx
import pytest

from tllama.main import app
from tllama.routers import ollama


@pytest.mark.parametrize('body', [b'[]', b'null', b'"model"', b'{', b'',
    b'{"model":12}', b'{"name":[]}', b'{"model":false}', b'{"name":{}}'])
async def test_invalid_delete_body_is_400_and_never_reaches_manager(monkeypatch, body):
    delete = AsyncMock(side_effect=AssertionError('invalid input reached deletion'))
    monkeypatch.setattr(ollama.model_manager, 'delete_model', delete)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.request('DELETE', '/api/delete', content=body,
                                        headers={'Content-Type': 'application/json'})
    assert response.status_code == 400, response.text
    assert isinstance(response.json()['error'], str)
    delete.assert_not_awaited()


@pytest.mark.parametrize('payload', [[], None, 'model', {'model':12}, {'name':[]},
    {'name':False}, {'runner':[],'model':'a'}])
async def test_invalid_show_body_never_reads_model_metadata(monkeypatch, payload):
    metadata = AsyncMock(side_effect=AssertionError('invalid input reached metadata'))
    monkeypatch.setattr(ollama.model_manager, 'get_model_metadata', metadata)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        # Explicit null is a JSON value, not HTTPX's missing-body shorthand.
        import json
        response = await client.post('/api/show', content=json.dumps(payload),
                                     headers={'Content-Type':'application/json'})
    assert response.status_code == 400 and isinstance(response.json()['error'], str)
    metadata.assert_not_awaited()


@pytest.mark.parametrize('payload,expected', [({'model':'a'}, 'a'), ({'name':'a'}, 'a'),
    ({'name':'a','model':'b'}, 'a'), ({'name':None,'model':'a','verbose':True}, 'a')])
async def test_show_preserves_aliases_and_ignored_extension_fields(monkeypatch, payload, expected):
    metadata = AsyncMock(return_value={'arch':'llama'})
    monkeypatch.setattr(ollama.model_manager, 'get_model_metadata', metadata)
    monkeypatch.setattr(ollama.model_manager, 'read_model_definition', lambda name: None)
    monkeypatch.setattr(ollama.model_manager, '_toml_fingerprint_for_reference', lambda name: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/show', json=payload)
    assert response.status_code == 200 and response.json()['details']['family'] == 'llama'
    metadata.assert_awaited_once_with(expected)


@pytest.mark.parametrize('payload,expected', [({'model':' a '}, 'a'), ({'name':' a '}, 'a'),
    ({'name':'a','model':'b'}, 'a'), ({'name':None,'model':'a','extra':True}, 'a')])
async def test_delete_preserves_alias_order_trimming_and_extension_fields(monkeypatch, payload, expected):
    result = {'deleted_path':'Local/a.toml', 'kept_model_file':None, 'cleanup':{}}
    delete = AsyncMock(return_value=result)
    monkeypatch.setattr(ollama.model_manager, 'delete_model', delete)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.request('DELETE', '/api/delete', json=payload)
    assert response.status_code == 200 and response.json()['deleted'] == result['deleted_path']
    delete.assert_awaited_once_with(expected)


@pytest.mark.parametrize('payload', [{}, {'name':None}, {'model':''}, {'name':' ', 'model':'valid'}])
async def test_empty_delete_keeps_existing_missing_name_error(monkeypatch, payload):
    delete = AsyncMock(side_effect=AssertionError('empty name reached deletion'))
    monkeypatch.setattr(ollama.model_manager, 'delete_model', delete)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.request('DELETE', '/api/delete', json=payload)
    assert response.status_code == 400 and response.json() == {'error':'Missing model name'}
    delete.assert_not_awaited()
