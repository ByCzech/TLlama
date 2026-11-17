"""Contract errors are actionable without translating registries or importing Modelfiles."""

from contextlib import aclosing
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from tllama.client import ClientError, TLlamaClient
from tllama.client_import import create_from_file
from tllama.main import app
from tllama.routers import ollama as routes


DEFINITION = '[llm]\nfrom="missing.gguf"\n'


@pytest.mark.parametrize('payload', [
    {'model': 'new', 'from': 'qwen3:8b'},
    {'model': 'new', 'modelfile': 'FROM qwen3:8b'},
    {'model': 'new', 'files': {'weights.gguf': 'sha256:' + 'a' * 64}},
    {'model': 'new', 'messages': [{'role': 'user', 'content': 'hello'}]},
    {'model': 'new', 'definition': DEFINITION, 'from': 'qwen3:8b'},
    {'model': 'new', 'definition': DEFINITION, 'stream': False},
    {'model': 'new'},
    {'model': 'new', 'definition': 42},
])
async def test_foreign_create_payload_is_400_before_repository_operation(monkeypatch, payload):
    create = Mock(side_effect=AssertionError('invalid body reached repository'))
    monkeypatch.setattr(routes, 'create_definition', create)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/create', json=payload)
    assert response.status_code == 400
    error = response.json()['error']
    assert 'definition' in error and 'TLlama TOML' in error and 'Ollama' in error
    create.assert_not_called()


async def test_modelfile_inside_definition_is_not_imported(manager, monkeypatch):
    monkeypatch.setattr(routes, 'model_manager', manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/create', json={'model': 'new', 'definition': 'FROM qwen3:8b\n'})
    assert response.status_code == 400
    error = response.json()['error']
    assert 'invalid TOML' in error and 'TLlama TOML' in error and 'Modelfile' in error
    assert not list(manager.models_dir.rglob('*.toml'))
    assert not (manager.models_dir / '.tllama/uploads').exists()
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('stream', [False, True, None])
@pytest.mark.parametrize('reference', ['qwen3:8b', 'library/qwen3:8b',
    'https://huggingface.co/ns/repo/blob/main/model.gguf',
    'https://huggingface.co/ns/repo/resolve/main/model.gguf', 'ns/repo/../model'])
async def test_non_hf_reference_fails_before_remote_lookup(manager, monkeypatch, stream, reference):
    monkeypatch.setattr(routes, 'model_manager', manager)
    lookup = AsyncMock(side_effect=AssertionError('invalid reference reached HF'))
    pull = AsyncMock(side_effect=AssertionError('invalid reference downloaded'))
    monkeypatch.setattr(manager, 'fetch_hf_file_info', lookup)
    monkeypatch.setattr(manager, 'pull_hf_file', pull)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', json={'model': reference, 'stream': stream})
    assert response.status_code == 400
    error = response.json()['error']
    assert 'Hugging Face' in error and 'Ollama registry' in error and 'namespace/repository' in error
    lookup.assert_not_awaited()
    pull.assert_not_awaited()
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('stream', [False, True, None])
async def test_missing_hf_file_keeps_404_and_explains_source(manager, monkeypatch, stream):
    monkeypatch.setattr(routes, 'model_manager', manager)
    monkeypatch.setattr(manager, 'fetch_hf_file_info', AsyncMock(return_value=SimpleNamespace(status='missing')))
    pull = AsyncMock(side_effect=AssertionError('known missing file downloaded'))
    monkeypatch.setattr(manager, 'pull_hf_file', pull)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', json={'model': 'ns/repo/file.gguf', 'stream': stream})
    assert response.status_code == 404
    assert "'file.gguf' not found in repository 'ns/repo'" in response.json()['error']
    assert 'Hugging Face' in response.json()['error']
    pull.assert_not_awaited()


@pytest.mark.parametrize('stream', [False, True, None])
@pytest.mark.parametrize('reference,filename', [('ns/repo/file', 'file.gguf'),
    ('ns/repo/sub/file.gguf', 'sub/file.gguf'), ('ns/repo/blob/main/file', 'blob/main/file.gguf')])
async def test_supported_pull_preserves_literal_paths_and_stream_shapes(manager, monkeypatch, stream, reference, filename):
    monkeypatch.setattr(routes, 'model_manager', manager)
    lookup = AsyncMock(return_value=SimpleNamespace(status='found', size=10))
    pull = AsyncMock(return_value='data/group/weights.gguf')
    monkeypatch.setattr(manager, 'fetch_hf_file_info', lookup)
    monkeypatch.setattr(manager, 'pull_hf_file', pull)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', json={'model': reference, 'stream': stream})
    assert response.status_code == 200
    result = response.json() if stream is False else json.loads(response.text.splitlines()[-1])
    assert result == {'status': 'success', 'path': 'data/group/weights.gguf'}
    assert lookup.await_args.kwargs['filename'] == filename
    assert pull.await_args.kwargs['repo_id'] == 'ns/repo'


@pytest.mark.parametrize('operation', ['create', 'check_create'])
@pytest.mark.parametrize('status,body,original', [
    (400, b'{"error":"neither from nor files provided"}', 'neither from nor files'),
    (200, b'{"error":"unsupported create request"}', 'unsupported create request'),
    (200, b'{"status":"reading"}\n{"error":"bad source"}\n', 'invalid JSON'),
    (200, b'{"status":"success"}', 'preflight'),
])
async def test_foreign_create_responses_keep_original_error_and_contract(operation, status, body, original):
    if operation == 'create' and original == 'preflight':
        body, original = b'{"status":"ready"}', 'creation'
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(status, content=body)
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError) as caught:
            await getattr(client, operation)('new', DEFINITION)
    error = str(caught.value)
    assert original in error and 'TLlama TOML' in error and 'TLlama daemon' in error
    assert len(calls) == 1 and calls[0][:2] == ('POST', '/api/create')
    assert calls[0][2]['definition'] == DEFINITION


async def test_failed_foreign_preflight_stops_before_hash_or_upload(tmp_path, monkeypatch):
    definition = tmp_path / 'model.toml'
    definition.write_text(DEFINITION)
    scan = Mock(side_effect=AssertionError('preflight failure must stop local scanning/hashing'))
    monkeypatch.setattr('tllama.client_import._groups', scan)
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(400, json={'error': 'neither from nor files provided'})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match='TLlama TOML'):
            await create_from_file(client, 'new', definition, as_json=True)
    assert calls == [('POST', '/api/create')]
    scan.assert_not_called()


@pytest.mark.parametrize('status,body,original', [
    (404, b'{"error":"pull model manifest: file does not exist"}', 'pull model manifest'),
    (200, b'{"status":"pulling manifest"}\n{"error":"remote rejected source"}\n', 'remote rejected source'),
    (200, b'{"status":"pulling manifest"}\n', 'without confirming success'),
    (200, b'not json\n', 'invalid NDJSON'),
])
async def test_foreign_pull_error_preserves_message_and_explains_both_sources(status, body, original):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(status, content=body)
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError) as caught:
            async with aclosing(client.pull('ns/repo/file.gguf')) as progress:
                async for _ in progress:
                    pass
    error = str(caught.value)
    assert original in error and 'Hugging Face' in error and 'Ollama registry model name' in error
    assert calls == [{'model': 'ns/repo/file.gguf', 'stream': True}]


async def test_client_can_forward_registry_reference_without_guessing_server():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b'{"status":"pulling manifest"}\n{"status":"success"}\n')
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        events = [event async for event in client.pull('qwen3:8b')]
    assert events == [{'status': 'pulling manifest'}, {'status': 'success'}]
    assert calls == [{'model': 'qwen3:8b', 'stream': True}]


async def test_server_hint_is_not_duplicated_by_client(manager, monkeypatch):
    monkeypatch.setattr(routes, 'model_manager', manager)
    async with TLlamaClient('http://test', transport=httpx.ASGITransport(app=app)) as client:
        with pytest.raises(ClientError) as caught:
            async for _ in client.pull('qwen3:8b'):
                pass
    assert str(caught.value).count('TLlama /api/pull') == 1


async def test_create_server_hint_is_not_duplicated_by_client(manager, monkeypatch):
    monkeypatch.setattr(routes, 'model_manager', manager)
    async with TLlamaClient('http://test', transport=httpx.ASGITransport(app=app)) as client:
        with pytest.raises(ClientError) as caught:
            # Missing upload group is a normal invalid TLlama create request.
            await client.create('new', DEFINITION)
    assert str(caught.value).count('TLlama /api/create') == 1
    assert 'Uploads must match exactly' in str(caught.value)


async def test_local_modelfile_error_names_toml_before_any_http(tmp_path):
    definition = tmp_path / 'Modelfile'
    definition.write_text('FROM qwen3:8b\n')
    request = Mock(side_effect=AssertionError('Modelfile must fail locally'))
    async with TLlamaClient(transport=httpx.MockTransport(request)) as client:
        with pytest.raises(ClientError) as caught:
            await create_from_file(client, 'new', definition, as_json=True)
    assert 'invalid TOML' in str(caught.value) and 'TLlama TOML' in str(caught.value)
    request.assert_not_called()


@pytest.mark.parametrize('operation', ['create', 'pull'])
@pytest.mark.parametrize('stream', [False, True])
async def test_ollama_python_sdk_receives_actionable_400(monkeypatch, operation, stream):
    # Ollama creates default clients at import; keep optional local tests proxy-independent.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(key, raising=False)
    sdk = pytest.importorskip('ollama')
    create = Mock(side_effect=AssertionError('SDK foreign create reached repository'))
    lookup = AsyncMock(side_effect=AssertionError('SDK registry tag reached HF'))
    monkeypatch.setattr(routes, 'create_definition', create)
    monkeypatch.setattr(routes.model_manager, 'fetch_hf_file_info', lookup)
    client = sdk.AsyncClient(host='http://test', transport=httpx.ASGITransport(app=app), trust_env=False)
    try:
        with pytest.raises(sdk.ResponseError) as caught:
            if operation == 'create':
                response = await client.create(model='new', from_='qwen3:8b', stream=stream)
            else:
                response = await client.pull(model='qwen3:8b', stream=stream)
            if stream:
                async with aclosing(response):
                    async for _ in response:
                        pass
        assert caught.value.status_code == 400
        assert ('TLlama TOML' if operation == 'create' else 'Hugging Face') in str(caught.value)
    finally:
        await client._client.aclose()
    create.assert_not_called()
    lookup.assert_not_awaited()
