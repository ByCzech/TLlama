"""Exercise advertised helper management through the public CLI and owned TCP."""

import asyncio
import json
import os
from pathlib import Path
import socket
import sys

import httpx
import pytest
import uvicorn

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.main import app
from tllama.routers import ollama, openai
from test_model_create import gguf_file


@pytest.mark.parametrize('alias_root', [False, True])
async def test_cli_helper_create_discover_copy_show_and_remove_with_real_advertisement(
        manager, tmp_path, monkeypatch, alias_root):
    if alias_root:
        alias = tmp_path / 'models-alias'
        alias.symlink_to(manager.repository.root, target_is_directory=True)
        manager = ModelManager(BackendConfig(models_dir=str(alias)))
        manager.ensure_storage()
    monkeypatch.setattr(ollama, 'model_manager', manager)
    monkeypatch.setattr(openai, 'model_manager', manager)
    definitions = {}
    for role, arch in [('llm', 'llama'), ('mmproj', 'clip'), ('draft', 'dflash')]:
        files = [gguf_file(tmp_path / f'{role}-{i+1:05d}-of-00002.gguf', arch=arch, shard=i, count=2)
                 for i in range(2)]
        definition = tmp_path / (role + '.toml')
        definition.write_text(f'[{role}]\nfrom={json.dumps(str(files[0]))}\n')
        definitions[role] = definition
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    address = f'http://127.0.0.1:{listener.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith('_proxy')}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    async def command(*args):
        child = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tllama', *args,
            '--host', address, '--json', env=env, cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(child.communicate(), 15)
            assert child.returncode == 0, stderr.decode()
            return json.loads(stdout), stderr.decode()
        finally:
            if child.returncode is None:
                child.kill()
                await child.communicate()
    try:
        async def ready():
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError('Server stopped before startup')
                await asyncio.sleep(0.01)
        await asyncio.wait_for(ready(), 5)
        empty, _ = await command('list', '--all')
        assert empty == {'models': [], 'server_features': ['repository_helpers']}
        for role, name in [('llm', 'base'), ('mmproj', 'projector'), ('draft', 'draft')]:
            created, _ = await command('create', name, '--file', str(definitions[role]))
            assert created == {'status': 'success', 'model': name}
        copied, _ = await command('cp', 'base', 'composite', '--mmproj', 'projector', '--draft', 'draft')
        assert copied['status'] == 'success'
        default, _ = await command('list')
        assert {row['name'] for row in default['models']} == {'base', 'composite'}
        all_models, _ = await command('list', '--all')
        assert {row['name']: row['model_definition']['type'] for row in all_models['models']} == {
            'base': 'main', 'composite': 'main', 'projector': 'mmproj', 'draft': 'draft'}
        for role, name in [('mmproj', 'projector'), ('draft', 'draft')]:
            filtered, _ = await command('list', '--type', role)
            assert [row['name'] for row in filtered['models']] == [name]
        shown, _ = await command('show', 'composite')
        spec = parse_model_toml(shown['modelfile'])
        expected = {role: value for role, value, source in spec.components}
        assert shown['model_definition'] == {'type': 'main', 'artifacts': expected}
        assert spec.runtime == {}
        paths = {role: manager.repository.root / value for role, value in expected.items()}
        signatures = {path: path.stat().st_ino for file in paths.values() for path in file.parent.glob('*.gguf')}
        assert len(signatures) == 6
        async with httpx.AsyncClient(base_url=address, trust_env=False) as wire:
            head = await wire.head('/api/tags?type=all')
            assert head.status_code == 200 and head.content == b''
            models = (await wire.get('/v1/models')).json()
            assert {row['id'] for row in models['data']} == {'base', 'composite'}
            # Legacy clients send no feature fields and still use the same routes.
            legacy = await wire.post('/api/copy', json={'source': 'base', 'destination': 'legacy-copy'})
            assert legacy.status_code == 200, legacy.text
            legacy = await wire.request('DELETE', '/api/delete', json={'name': 'legacy-copy'})
            assert legacy.status_code == 200 and legacy.json()['status'] == 'success'
        assert all(path.stat().st_ino == inode for path, inode in signatures.items())
        removed, stderr = await command('rm', 'composite', '--remove-unused-helpers')
        cleanup = removed['response']['cleanup']
        assert cleanup['removed_helpers'] == ['draft', 'projector'] and cleanup['unused_helpers'] == []
        assert 'removed helper: draft' in stderr and 'removed helper: projector' in stderr
        assert paths['llm'].is_file() and not paths['mmproj'].parent.exists() and not paths['draft'].parent.exists()
        final, _ = await command('list', '--all')
        assert [row['name'] for row in final['models']] == ['base']
        assert manager.native_snapshot() == []
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        listener.close()
        await manager.shutdown()
