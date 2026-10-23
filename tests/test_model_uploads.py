"""Chunked uploads verify bytes before publication and clean up interrupted files."""

import asyncio
from hashlib import sha256

import httpx
import pytest
from starlette.requests import Request

from tllama.main import app
from tllama.routers import ollama
from tllama.client import TLlamaClient, ClientError
from tllama.model_uploads import blob_path, receive_blob


PAYLOAD = b'GGUF' + bytes(range(128))
DIGEST = 'sha256:' + sha256(PAYLOAD).hexdigest()


async def test_streamed_upload_is_verified_and_idempotent(manager, monkeypatch):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with TLlamaClient('http://test', transport=httpx.ASGITransport(app=app)) as client:
        assert not await client.blob_exists(DIGEST)
        async def content():
            for i in range(0, len(PAYLOAD), 3):
                yield PAYLOAD[i:i + 3]
        from contextlib import aclosing
        async with aclosing(content()) as chunks:
            await client.upload_blob(DIGEST, chunks)
        target = blob_path(manager, DIGEST)
        inode = target.stat().st_ino
        assert await client.blob_exists(DIGEST)
        await client.upload_blob(DIGEST, PAYLOAD)
        assert target.stat().st_ino == inode and target.read_bytes() == PAYLOAD
    assert not list(manager.models_dir.rglob('*.toml'))
    assert not list(target.parent.glob('.upload-*'))


@pytest.mark.parametrize('payload,digest,status', [
    (PAYLOAD, 'sha256:' + '0' * 64, 400), (b'bad' * 40, DIGEST, 400),
    (b'GGUF', 'sha256:' + sha256(b'GGUF').hexdigest(), 400),
    (PAYLOAD, 'not-a-digest', 400),
])
async def test_invalid_upload_never_publishes(manager, monkeypatch, payload, digest, status):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/blobs/' + digest, content=payload,
                                     headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == status
    assert not list(manager.models_dir.rglob('.upload-*'))
    assert not list((manager.local_models_dir / '.uploads').glob('[0-9a-f]' * 64))


async def test_upload_cancellation_drains_disk_work_and_removes_temporary(manager):
    entered = asyncio.Event()
    calls = 0
    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {'type': 'http.request', 'body': PAYLOAD[:30], 'more_body': True}
        entered.set()
        await asyncio.Event().wait()
    request = Request({'type': 'http', 'headers': []}, receive)
    task = asyncio.create_task(receive_blob(manager, DIGEST, request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not blob_path(manager, DIGEST).exists()
    assert not list((manager.local_models_dir / '.uploads').glob('.upload-*'))


async def test_upload_detects_disconnected_body(manager, monkeypatch):
    monkeypatch.setattr(ollama, 'model_manager', manager)
    count = 0
    async def receive():
        nonlocal count
        count += 1
        return ({'type': 'http.request', 'body': PAYLOAD[:30], 'more_body': True} if count == 1
                else {'type': 'http.disconnect'})
    request = Request({'type': 'http', 'headers': []}, receive)
    from starlette.requests import ClientDisconnect
    with pytest.raises(ClientDisconnect):
        await receive_blob(manager, DIGEST, request)
    assert not blob_path(manager, DIGEST).exists()


def test_staging_cannot_escape_by_symlink(manager, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (manager.local_models_dir / '.uploads').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        blob_path(manager, DIGEST)


@pytest.mark.parametrize('outcome', ['cancel', 'write_error'])
async def test_upload_closes_all_producer_layers_between_writes(outcome):
    import sys
    entered = asyncio.Event()
    closed = asyncio.Event()
    held = []
    hooks = sys.get_asyncgen_hooks()
    def firstiter(gen):
        held.append(gen)
        if hooks.firstiter:
            hooks.firstiter(gen)
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            assert await anext(request.stream.__aiter__()) == b'first'
            entered.set()
            if outcome == 'write_error':
                raise httpx.WriteError('write failed', request=request)
            await asyncio.Event().wait()
    async def content():
        try:
            yield b'first'
            yield b'second'
        finally:
            closed.set()
    sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=hooks.finalizer)
    try:
        async with TLlamaClient(transport=Transport()) as client:
            task = asyncio.create_task(client.upload_blob(DIGEST, content()))
            await entered.wait()
            if outcome == 'cancel':
                task.cancel()
            with pytest.raises(asyncio.CancelledError if outcome == 'cancel' else ClientError):
                await task
            assert closed.is_set()
            assert [g.ag_code.co_qualname for g in held if g.ag_frame is not None] == []
    finally:
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)
        for gen in held:
            if not gen.ag_running:
                await gen.aclose()
