"""Helper create probes support before preflight, local scanning or upload."""

import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from tllama.client import TLlamaClient, ClientError
from tllama.client_import import create_from_file
from tllama.helpers.model_toml import parse_model_toml
from tllama.main import app
from tllama.repository_check import check_repository
from tllama.repository_names import definition_path
from tllama.repository_prune import prune_repository
from tllama.routers import ollama
from test_model_create import gguf_file, create_client


def section(role, value, key='from'):
    return f'[{role}]\n{key} = {json.dumps(value)}\n'


@pytest.mark.parametrize('operation', ['check_create', 'create'])
@pytest.mark.parametrize('supported', [False, True])
@pytest.mark.parametrize('text', [section('mmproj', 'source.gguf'), section('draft', 'source.gguf'),
    section('mmproj', 'data/group/weights.gguf', 'model'), section('draft', 'data/group/weights.gguf', 'model'),
    section('llm', 'main.gguf') + section('draft', 'draft.gguf'),
    section('llm', 'data/main/weights.gguf', 'model') + section('draft', 'data/draft/weights.gguf', 'model')])
async def test_helper_create_requires_capability_even_for_definition_only_operations(operation, supported, text):
    calls = []
    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.method == 'GET':
            return httpx.Response(200, json={'models': [], 'server_features': ['repository_helpers'] if supported else []})
        return httpx.Response(200, json={'status': 'ready' if operation == 'check_create' else 'success'})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        if supported:
            await getattr(client, operation)('model', text, force=True)
        else:
            with pytest.raises(ClientError, match='repository_helpers'):
                await getattr(client, operation)('model', text, force=True)
    assert calls[0] == ('GET', '/api/tags', None)
    if supported:
        expected = {'model': 'model', 'definition': text, 'force': True}
        expected.update({'dry_run': True} if operation == 'check_create' else {'uploads': {}})
        assert calls[1:] == [('POST', '/api/create', expected)]
    else:
        assert len(calls) == 1


@pytest.mark.parametrize('operation', ['check_create', 'create'])
@pytest.mark.parametrize('projector', [False, True])
async def test_ordinary_main_and_projector_create_do_not_probe_or_change_payload(operation, projector):
    text = section('llm', 'source.gguf') + (section('mmproj', 'projector.gguf') if projector else '')
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'status': 'ready' if operation == 'check_create' else 'success'})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        await getattr(client, operation)('model', text)
    expected = {'model': 'model', 'definition': text, 'force': False}
    expected.update({'dry_run': True} if operation == 'check_create' else {'uploads': {}})
    assert calls == [('POST', '/api/create', expected)]


@pytest.mark.parametrize('text', [section('mmproj', 'missing.gguf'), section('draft', 'missing.gguf'),
    section('llm', 'main.gguf') + section('draft', 'missing.gguf')])
async def test_unsupported_server_stops_before_local_header_hash_or_upload(tmp_path, monkeypatch, text):
    path = tmp_path / 'definition.toml'
    path.write_text(text)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported operations must not inspect local GGUF files')
    monkeypatch.setattr('tllama.client_import._groups', forbidden)
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={'models': []})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match='repository_helpers'):
            await create_from_file(client, 'model', path, as_json=True)
    assert calls == [('GET', '/api/tags')]


@pytest.fixture
async def helper_client(manager, monkeypatch):
    # The production marker intentionally waits for the remaining list patch.
    # Simulate that advertised contract only here; mutations use the real app.
    monkeypatch.setattr(ollama, 'model_manager', manager)
    calls = []
    async with httpx.ASGITransport(app=app) as transport:
        async def handle(request):
            calls.append((request.method, request.url.path))
            if request.method == 'GET' and request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [], 'server_features': ['repository_helpers']})
            return await transport.handle_async_request(request)
        async with TLlamaClient('http://test', transport=httpx.MockTransport(handle)) as client:
            yield client, calls


@pytest.mark.parametrize('role, arch', [('mmproj', 'clip'), ('draft', 'dflash')])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('absolute', [False, True])
async def test_explicit_local_helper_import_registers_only_requested_toml_and_complete_data(
        helper_client, manager, tmp_path, monkeypatch, role, arch, split, absolute):
    client, calls = helper_client
    source = tmp_path / 'client files'
    monkeypatch.chdir(source.parent)
    names = [f'{role}-00001-of-00002.gguf', f'{role}-00002-of-00002.gguf'] if split else [f'{role}:Q8%.gguf']
    files = [gguf_file(source / name, arch=arch, shard=i if split else None, count=2 if split else None)
             for i, name in enumerate(names)]
    definition = tmp_path / 'definitions' / 'helper.toml'
    definition.parent.mkdir()
    local = str(files[0]) if absolute else files[0].relative_to(Path.cwd()).as_posix()
    definition.write_text('# Keep comment\n' + section(role, local))
    public_name = 'ByCzech/Helper:Q8%'
    assert await create_from_file(client, public_name, definition, as_json=True) == {'status': 'success', 'model': public_name}
    target = manager.models_dir / definition_path(public_name)
    assert list(manager.models_dir.rglob('*.toml')) == [target]
    spec = parse_model_toml(target.read_text())
    assert spec.model_type == role and set(spec.document) == {role}
    assert '# Keep comment' in target.read_text()
    installed = manager.repository.root / spec.primary_model
    members = sorted(installed.parent.glob('*.gguf'))
    assert [member.read_bytes() for member in members] == [file.read_bytes() for file in files]
    assert [member.with_suffix('.inf').read_text().splitlines() for member in members] == [[name] for name in names]
    assert check_repository(manager.repository)['ok']
    assert prune_repository(manager.repository, apply=True)['removed_groups'] == []
    assert calls[:2] == [('GET', '/api/tags'), ('POST', '/api/create')]
    assert calls[-2:] == [('GET', '/api/tags'), ('POST', '/api/create')]
    assert manager.native_snapshot() == []
    manager.delete_model_definition(public_name)
    assert not installed.parent.exists()


@pytest.mark.parametrize('independent_helpers', [False, True])
async def test_composite_import_creates_no_implicit_helpers_and_respects_explicit_ones(
        helper_client, manager, tmp_path, independent_helpers):
    client, calls = helper_client
    refs, paths = {}, {}
    for role, arch in [('llm', 'llama'), ('mmproj', 'clip'), ('draft', 'dflash')]:
        files = [gguf_file(tmp_path / f'{role}-{i+1:05d}-of-00002.gguf', arch=arch, shard=i, count=2) for i in range(2)]
        refs[role] = section(role, str(files[0]))
        if independent_helpers and role != 'llm':
            definition = tmp_path / (role + '.toml')
            definition.write_text(refs[role])
            await create_from_file(client, role, definition, as_json=True)
    definition = tmp_path / 'composite.toml'
    definition.write_text('# Composite\n' + ''.join(refs.values()) + '[sampling]\ntemperature=0.3\n')
    await create_from_file(client, 'main', definition, as_json=True)
    spec = parse_model_toml(manager.read_model_definition('main'))
    assert spec.sampling == {'temperature': 0.3} and 'spec_type' not in spec.runtime
    for role, model, source in spec.components:
        assert source is None
        paths[role] = manager.repository.root / model
    assert len(list(manager.models_dir.rglob('*.toml'))) == (3 if independent_helpers else 1)
    assert len(list((manager.models_dir / 'data').iterdir())) == 3
    assert sum(method == 'POST' and '/blobs/' in path for method, path in calls) == 6
    manager.delete_model_definition('main')
    assert not paths['llm'].parent.exists()
    assert all(path.is_file() is independent_helpers for role, path in paths.items() if role != 'llm')
    if independent_helpers:
        for role in ('mmproj', 'draft'):
            manager.delete_model_definition(role)
    assert not list((manager.models_dir / 'data').iterdir())


@pytest.mark.parametrize('broken', ['missing', 'missing-shard', 'continuation', 'hash-failure'])
async def test_bad_draft_preparation_never_starts_any_upload(helper_client, manager, tmp_path, monkeypatch, broken):
    client, calls = helper_client
    main = gguf_file(tmp_path / 'main.gguf')
    draft = tmp_path / 'draft.gguf'
    if broken in {'missing-shard', 'continuation'}:
        draft = gguf_file(tmp_path / 'draft-00001-of-00002.gguf', arch='dflash', shard=0, count=2)
        if broken == 'continuation':
            draft = gguf_file(tmp_path / 'draft-00002-of-00002.gguf', arch='dflash', shard=1, count=2)
    elif broken == 'hash-failure':
        draft = gguf_file(draft, arch='dflash')
        from tllama.client_import import _hash
        def failing(path, *, stop):
            if path == draft:
                raise ClientError('draft changed during hash')
            return _hash(path, stop=stop)
        monkeypatch.setattr('tllama.client_import._hash', failing)
    definition = tmp_path / 'definition.toml'
    definition.write_text(section('llm', str(main)) + section('draft', str(draft)))
    with pytest.raises(ClientError):
        await create_from_file(client, 'main', definition, as_json=True)
    assert calls == [('GET', '/api/tags'), ('POST', '/api/create')]
    assert not (manager.models_dir / '.tllama/uploads').exists()
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_helper_name_conflict_fails_before_source_work(helper_client, manager, tmp_path, monkeypatch):
    client, calls = helper_client
    original = '# Existing user definition\n'
    (manager.models_dir / definition_path('taken')).write_text(original)
    path = tmp_path / 'helper.toml'
    path.write_text(section('draft', 'missing.gguf'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Conflict must be reported before local scanning')
    monkeypatch.setattr('tllama.client_import._groups', forbidden)
    with pytest.raises(ClientError, match='409'):
        await create_from_file(client, 'taken', path, as_json=True)
    assert calls == [('GET', '/api/tags'), ('POST', '/api/create')]
    assert (manager.models_dir / definition_path('taken')).read_text() == original


async def test_current_server_without_marker_keeps_helper_import_gated(create_client, manager, tmp_path):
    definition = tmp_path / 'helper.toml'
    definition.write_text(section('draft', 'missing.gguf'))
    with pytest.raises(ClientError, match='repository_helpers'):
        await create_from_file(create_client, 'helper', definition, as_json=True)
    assert not list(manager.models_dir.rglob('*.toml'))


def test_definition_only_helper_import_needs_no_local_gguf_or_server_dependencies(tmp_path):
    definition = tmp_path / 'helper.toml'
    definition.write_text(section('draft', 'data/group/weights.gguf', 'model'))
    code = '''
import importlib.abc
import sys
class RefuseRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gguf', 'llama_cpp', 'fastapi', 'uvicorn'} or fullname in {
            'tllama.backend', 'tllama.main', 'tllama.worker', 'tllama.config'}:
            raise AssertionError('Runtime dependency imported: ' + fullname)
sys.meta_path.insert(0, RefuseRuntime())
import asyncio
import json
import httpx
from tllama.client import TLlamaClient
from tllama.client_import import create_from_file
calls = []
def handle(request):
    calls.append((request.method, request.url.path))
    if request.method == 'GET':
        data = {'models': [], 'server_features': ['repository_helpers']}
    else:
        data = {'status': 'ready' if json.loads(request.content).get('dry_run') else 'success'}
    return httpx.Response(200, json=data)
async def run():
    async with TLlamaClient(transport=httpx.MockTransport(handle)) as client:
        assert await create_from_file(client, 'helper', sys.argv[1], as_json=True) == {'status': 'success'}
    assert calls == [('GET', '/api/tags'), ('POST', '/api/create')] * 2
asyncio.run(run())
'''
    result = subprocess.run([sys.executable, '-c', code, str(definition)],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
