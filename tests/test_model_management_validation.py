"""Malformed management inputs fail before filesystem/network side effects."""

from unittest.mock import AsyncMock
from types import SimpleNamespace
import json

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


@pytest.mark.parametrize('body', [b'[]', b'null', b'{', b'', b'{"model":12}',
    b'{"name":[]}', b'{"model":"a","username":42}', b'{"model":"a","password":[]}',
    b'{"model":"a","stream":"false"}', b'{"model":"a","stream":0}',
    b'{"model":"a","runner":{}}'])
async def test_invalid_pull_never_resolves_or_contacts_huggingface(monkeypatch, body):
    from unittest.mock import Mock
    resolve = Mock(side_effect=AssertionError('invalid input reached repository resolution'))
    lookup = AsyncMock(side_effect=AssertionError('invalid input reached network'))
    monkeypatch.setattr(ollama.model_manager, 'resolve_hf_pull_target', resolve)
    monkeypatch.setattr(ollama.model_manager, 'fetch_hf_file_info', lookup)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', content=body, headers={'Content-Type':'application/json'})
    assert response.status_code == 400 and isinstance(response.json()['error'], str)
    assert '42' not in response.json()['error']
    resolve.assert_not_called()
    lookup.assert_not_awaited()


@pytest.mark.parametrize('stream', [False, True, None])
@pytest.mark.parametrize('alias', ['model', 'name', 'both'])
async def test_pull_preserves_alias_precedence_token_and_stream_modes(monkeypatch, stream, alias):
    from unittest.mock import Mock
    payload = {'stream':stream, 'username':None, 'password':' token ', 'extra':'ignored'}
    payload[alias if alias != 'both' else 'model'] = ' ns/repo/file.gguf '
    if alias == 'both':
        payload['name'] = 'other/repo/file.gguf'
    resolve = Mock(return_value={'repo_id':'ns/repo','filename':'file.gguf'})
    lookup = AsyncMock(return_value=SimpleNamespace(status='found',size=10))
    pull = AsyncMock(return_value='data/hash/weights.gguf')
    monkeypatch.setattr(ollama.model_manager, 'resolve_hf_pull_target', resolve)
    monkeypatch.setattr(ollama.model_manager, 'fetch_hf_file_info', lookup)
    monkeypatch.setattr(ollama.model_manager, 'pull_hf_file', pull)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', json=payload)
    assert response.status_code == 200
    if stream is False:
        assert response.json()['status'] == 'success'
    else:
        assert response.headers['content-type'].startswith('application/x-ndjson')
        assert json.loads(response.text.splitlines()[-1])['status'] == 'success'
    resolve.assert_called_once_with('ns/repo/file.gguf')
    assert lookup.await_args.kwargs['token'] == 'token'
    assert pull.await_args.kwargs['token'] == 'token'
