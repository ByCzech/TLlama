"""Copy real repository definitions through HTTP without loading or cloning weights."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tllama.main import app
from tllama.routers import ollama
from tllama.model_definitions import copy_definition
from tllama.helpers.model_toml import write_model_toml
from tllama.cli import main
from test_client_cli import cli_server, isolated_proxy_environment


@pytest.fixture
def definition(manager):
    manager.ensure_storage()
    (manager.local_models_dir / 'weights.gguf').write_bytes(b'weights')
    (manager.local_models_dir / 'projector.gguf').write_bytes(b'projector')
    text = '# Keep comments\n[llm]\nmodel = "Local/weights.gguf"\n[mmproj]\nmodel="Local/projector.gguf"\n[system]\nprompt="Czech"\n'
    (manager.local_models_dir / 'original.toml').write_text(text)
    return text


@pytest.mark.parametrize('force', [False, True])
async def test_copy_preserves_definition_and_weight_inodes(manager, definition, monkeypatch, force):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    target = manager.tllama_models_dir / 'collection' / 'copy.toml'
    if force:
        target.parent.mkdir(parents=True)
        target.write_text('previous definition')
    inode = (manager.local_models_dir / 'weights.gguf').stat().st_ino
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/copy', json={'source': 'original', 'destination': 'collection/copy', 'force': force})
    assert response.status_code == 200, response.text
    assert target.read_text() == definition
    assert (manager.local_models_dir / 'weights.gguf').stat().st_ino == inode
    assert len(list(manager.models_dir.rglob('*.gguf'))) == 2
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('source,destination,status', [
    ('missing', 'copy', 404), ('original', 'original', 400), ('original', '../bad', 400),
    ('original', '/absolute', 400), ('original', 'a//b', 400), ('original', 'a/b/c/d', 400),
    ('original', 'a\\b', 400), ('original', 'original ', 400),
])
async def test_copy_rejects_bad_names(manager, definition, monkeypatch, source, destination, status):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/copy', json={'source': source, 'destination': destination})
    assert response.status_code == status, response.text


async def test_copy_conflict_preserves_destination(manager, definition, monkeypatch):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    target = manager.local_models_dir / 'copy.toml'
    target.write_text('do not replace')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        result = await client.post('/api/copy', json={'source': 'original', 'destination': 'copy'})
    assert result.status_code == 409 and target.read_text() == 'do not replace'


def test_copy_cannot_escape_through_parent_symlink(manager, definition, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (manager.tllama_models_dir / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        copy_definition(manager, 'original', 'escape/copied', force=True)
    assert list(outside.iterdir()) == []


def test_atomic_no_clobber_for_concurrent_definitions(tmp_path):
    path = tmp_path / 'shared.toml'
    def write(text):
        try:
            write_model_toml(path, text)
            return text
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write, [str(n) for n in range(20)]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1 and path.read_text() == winners[0]
    assert not list(tmp_path.glob('*.tmp'))


def test_cli_copy_posts_controls(cli_server, capsys):
    cli_server['responses']['/api/copy'] = (200, {'status': 'success'})
    assert main(['cp', 'source', 'destination', '--force', '--json', '--host', cli_server['url']]) == 0
    assert cli_server['calls'] == [('POST', '/api/copy', {'source': 'source', 'destination': 'destination', 'force': True})]
    assert json.loads(capsys.readouterr().out) == {'status': 'success'}
