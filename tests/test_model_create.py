"""Create from client-side TOML, including Local imports and complete shard groups."""

import asyncio
from contextlib import aclosing
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import threading
import json
import os
import socket
import sys

import gguf
import httpx
import pytest
import uvicorn

from tllama.main import app
from tllama.routers import ollama
from tllama.client import TLlamaClient, ClientError
from tllama.client_import import create_from_file, _file_job, _hash, _chunks
from tllama.helpers.model_toml import parse_model_toml


def gguf_file(path, *, shard=None, count=None, arch='llama'):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), arch=arch)
    if shard is not None:
        writer.add_uint16('split.no', shard)
        writer.add_uint16('split.count', count)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


@pytest.fixture
async def create_client(manager, monkeypatch):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with TLlamaClient('http://test', transport=httpx.ASGITransport(app=app)) as client:
        yield client


async def test_definition_only_create_preserves_text_and_changes_no_weights(create_client, manager, tmp_path):
    weight = gguf_file(manager.local_models_dir / 'existing.gguf')
    signature = weight.stat()
    text = '# User comment\n[llm]\nmodel = "Local/existing.gguf"\n[system]\nprompt="Hello"\n'
    definition = tmp_path / 'client' / 'model.toml'
    definition.parent.mkdir()
    definition.write_text(text)
    result = await create_from_file(create_client, 'created', definition, as_json=True)
    assert result == {'status': 'success', 'model': 'created'}
    assert (manager.local_models_dir / 'created.toml').read_text() == text
    assert weight.stat().st_ino == signature.st_ino
    assert not (manager.local_models_dir / '.uploads').exists()
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('absolute', [False, True])
async def test_from_uploads_model_and_projector_into_server_local(create_client, manager, tmp_path, monkeypatch, absolute):
    source = tmp_path / 'client files'
    model = gguf_file(source / 'model.gguf')
    projector = gguf_file(source / 'projector.gguf', arch='clip')
    definition = tmp_path / 'definitions' / 'model.toml'
    gguf_file(definition.parent / model.name, arch='other-model')
    gguf_file(definition.parent / projector.name, arch='other-projector')
    monkeypatch.chdir(tmp_path if absolute else source)
    model_from = str(model) if absolute else model.name
    projector_from = str(projector) if absolute else projector.name
    text = f'# Keep comment\n[llm]\nfrom = "{model_from}" # weight comment\n[mmproj]\nfrom="{projector_from}"\n[system]\nprompt="Czech"\n'
    definition.write_text(text)
    count = []
    receive = ollama.receive_blob
    async def tracked(*args):
        count.append(args[1])
        return await receive(*args)
    monkeypatch.setattr(ollama, 'receive_blob', tracked)
    await create_from_file(create_client, 'collection/created', definition, as_json=True)
    stored = (manager.tllama_models_dir / 'collection' / 'created.toml').read_text()
    spec = parse_model_toml(stored)
    assert spec.llm_from is None and spec.mmproj_from is None
    assert spec.llm_model.startswith('Local/imports/') and spec.mmproj_model.startswith('Local/imports/')
    assert (manager.models_dir / spec.llm_model).read_bytes() == model.read_bytes()
    assert (manager.models_dir / spec.mmproj_model).read_bytes() == projector.read_bytes()
    assert '# Keep comment' in stored and '# weight comment' in stored
    assert spec.system_prompt == 'Czech' and definition.read_text() == text
    await create_from_file(create_client, 'another', definition, as_json=True)
    assert len(count) == 2  # HEAD avoids uploading identical files again.
    assert manager.native_snapshot() == []
    def no_rescan(*args, **kwargs):
        raise AssertionError('Import must warm digest/header caches for the final paths')
    monkeypatch.setattr(manager, '_compute_content_sha256', no_rescan)
    monkeypatch.setattr(manager, '_get_model_metadata_sync', no_rescan)
    listed = await manager.list_local_models_with_metadata()
    assert {row['id'] for row in listed} == {'collection/created', 'another'}


async def test_split_from_uploads_every_shard_and_preserves_names(create_client, manager, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = gguf_file(tmp_path / 'bundle-00001-of-00002.gguf', shard=0, count=2)
    second = gguf_file(tmp_path / 'bundle-00002-of-00002.gguf', shard=1, count=2)
    definition = tmp_path / 'split.toml'
    definition.write_text('[llm]\nfrom="bundle-00001-of-00002.gguf"\n')
    await create_from_file(create_client, 'split', definition, as_json=True)
    spec = parse_model_toml((manager.local_models_dir / 'split.toml').read_text())
    installed = manager.models_dir / spec.llm_model
    assert installed.name == first.name and (installed.parent / second.name).read_bytes() == second.read_bytes()
    assert len(list(installed.parent.glob('*.gguf'))) == 2


@pytest.mark.parametrize('broken', ['missing_shard', 'continuation', 'bad_projector', 'bad_name'])
async def test_bad_inputs_do_not_upload_anything(create_client, manager, tmp_path, monkeypatch, broken):
    monkeypatch.chdir(tmp_path)
    first = gguf_file(tmp_path / 'bundle-00001-of-00002.gguf', shard=0, count=2)
    if broken != 'missing_shard':
        second = gguf_file(tmp_path / 'bundle-00002-of-00002.gguf', shard=1, count=2)
    text = f'[llm]\nfrom="{first.name}"\n'
    if broken == 'continuation':
        text = f'[llm]\nfrom="{second.name}"\n'
    elif broken == 'bad_projector':
        text += '[mmproj]\nfrom="missing.gguf"\n'
    definition = tmp_path / 'model.toml'
    definition.write_text(text)
    with pytest.raises(ClientError):
        await create_from_file(create_client, '../bad' if broken == 'bad_name' else 'created', definition, as_json=True)
    assert not (manager.local_models_dir / '.uploads').exists()
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_relative_from_does_not_fall_back_to_toml_directory(create_client, manager, tmp_path, monkeypatch):
    gguf_file(tmp_path / 'definitions' / 'model.gguf')
    definition = tmp_path / 'definitions' / 'model.toml'
    definition.write_text('[llm]\nfrom="model.gguf"\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ClientError, match='Import source is not a file'):
        await create_from_file(create_client, 'created', definition, as_json=True)
    assert not (manager.local_models_dir / '.uploads').exists()
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_create_requires_force_and_does_not_replace_weights(create_client, manager, tmp_path):
    gguf_file(manager.local_models_dir / 'existing.gguf')
    definition = tmp_path / 'model.toml'
    definition.write_text('[llm]\nmodel="Local/existing.gguf"\n')
    await create_from_file(create_client, 'created', definition, as_json=True)
    definition.write_text(definition.read_text() + '[system]\nprompt="changed"\n')
    with pytest.raises(ClientError, match='409'):
        await create_from_file(create_client, 'created', definition, as_json=True)
    await create_from_file(create_client, 'created', definition, force=True, as_json=True)
    assert parse_model_toml((manager.local_models_dir / 'created.toml').read_text()).system_prompt == 'changed'


async def test_conflict_fails_before_hashing_or_upload(create_client, manager, tmp_path, monkeypatch):
    (manager.local_models_dir / 'existing.toml').write_text('existing user content')
    source = gguf_file(tmp_path / 'file.gguf')
    definition = tmp_path / 'model.toml'
    definition.write_text('[llm]\nfrom="file.gguf"\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('Hashing should not start after a failed preflight')
    monkeypatch.setattr('tllama.client_import._hash', forbidden)
    with pytest.raises(ClientError, match='409'):
        await create_from_file(create_client, 'existing', definition, as_json=True)
    assert not (manager.local_models_dir / '.uploads').exists()
    assert (manager.local_models_dir / 'existing.toml').read_text() == 'existing user content'


async def test_api_from_cannot_read_server_files(create_client, manager, tmp_path):
    secret = gguf_file(tmp_path / 'server-only.gguf')
    with pytest.raises(ClientError, match='400'):
        await create_client.create('created', '[llm]\nfrom=' + repr(str(secret)) + '\n')
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_api_rejects_unreadable_blob_before_definition(create_client, manager):
    payload = b'GGUF' + b'\0' * 128
    digest = 'sha256:' + sha256(payload).hexdigest()
    await create_client.upload_blob(digest, payload)
    with pytest.raises(ClientError, match='400'):
        await create_client.create('bad', '[llm]\nfrom="file.gguf"\n',
                                    uploads={'llm': [{'digest': digest, 'filename': 'file.gguf'}]})
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_changed_staged_blob_cannot_poison_import_digest(create_client, manager, tmp_path):
    from tllama.model_uploads import blob_path
    source = gguf_file(tmp_path / 'source.gguf')
    raw = source.read_bytes()
    digest = 'sha256:' + sha256(raw).hexdigest()
    await create_client.upload_blob(digest, raw)
    blob_path(manager, digest).write_bytes(raw + b'changed')
    with pytest.raises(ClientError, match='400'):
        await create_client.create('bad', '[llm]\nfrom="source.gguf"\n',
                                    uploads={'llm': [{'digest': digest, 'filename': 'source.gguf'}]})
    assert not list(manager.models_dir.rglob('*.toml'))


async def test_cancelling_file_work_joins_worker_before_return():
    entered = threading.Event()
    finished = threading.Event()
    def work(*, stop):
        entered.set()
        stop.wait(timeout=5)
        finished.set()
    task = asyncio.create_task(_file_job(work))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_changed_file_is_refused_before_upload(tmp_path):
    source = gguf_file(tmp_path / 'file.gguf')
    digest, signature = await _file_job(_hash, source)
    source.write_bytes(source.read_bytes() + b'changed')
    async with aclosing(_chunks(source, signature)) as chunks:
        with pytest.raises(ClientError, match='changed before upload'):
            await anext(chunks)


async def test_real_cli_uploads_from_over_tcp(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    source = gguf_file(tmp_path / 'client' / 'remote.gguf')
    definition = tmp_path / 'definitions' / 'model.toml'
    gguf_file(definition.parent / source.name, arch='other-model')
    definition.write_text('[llm]\nfrom="remote.gguf"\n')
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    address = f'http://127.0.0.1:{listener.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith('_proxy')}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    child = None
    try:
        async def ready():
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError('Server stopped before startup')
                await asyncio.sleep(0.01)
        await asyncio.wait_for(ready(), 5)
        child = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tllama', 'create', 'tcp-model',
            '--file', '../definitions/model.toml', '--host', address, '--json', env=env, cwd=source.parent,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 15)
        assert child.returncode == 0, stderr.decode()
        assert json.loads(stdout) == {'status': 'success', 'model': 'tcp-model'}
        assert stderr == b''
        spec = parse_model_toml((manager.local_models_dir / 'tcp-model.toml').read_text())
        assert (manager.models_dir / spec.llm_model).read_bytes() == source.read_bytes()
        assert manager.native_snapshot() == []
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.communicate()
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        listener.close()
