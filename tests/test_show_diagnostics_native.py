"""Show leaves actual resident subprocesses and admitted leases untouched."""

import asyncio

from test_native_public_api import public
from test_native_manager import native_manager, records
import pytest


@pytest.mark.parametrize('busy', [False, True])
async def test_show_does_not_touch_resident_generation_or_lease(public, monkeypatch, busy):
    manager, client, name, _ = public
    monkeypatch.setattr(manager, '_get_model_metadata_sync', lambda _: {'arch': 'llama', 'template': '{{ Prompt }}'})
    if not busy:
        await manager.preload_native_model(name, keep_alive=-1)

    async def check():
        worker = manager.test.ledger[0]
        # Serialize against unrelated periodic telemetry. Show itself must
        # neither require this lock nor create any new worker request.
        async with worker._memory_lock:
            before = manager.native_snapshot()
            requests = records(manager.test.roots[0], 'requests')
            pid = worker.process.pid
            for _ in range(2):
                response = await asyncio.wait_for(client.post('/api/show', json={'model': name}), timeout=3)
                assert response.status_code == 200 and response.json()['template_source'] == 'tllama_toml'
            assert manager.native_snapshot() == before
            assert len(manager.test.ledger) == 1 and worker.process.pid == pid and worker.process.returncode is None
            assert records(manager.test.roots[0], 'requests') == requests

    if busy:
        async with manager.acquire_native_model(name, keep_alive=-1):
            await check()
    else:
        await check()
