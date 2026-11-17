"""Real owned TCP stub children verify device policy integration, not GPU loads."""

import asyncio
from dataclasses import replace

from test_native_manager import native_manager
from test_native_residency import wait_until


async def test_manager_device_limits_keep_other_native_pids_and_reap_before_replacement(native_manager, monkeypatch):
    manager = native_manager(limit=0, device_limits=[{'name':'Vulkan0','max_loaded_models':2}])
    factory = manager._native._factory
    monkeypatch.setattr(manager._native, '_factory', lambda spec: factory(replace(spec, transport='tcp')))
    for name, device in [('E','Vulkan0'), ('R','Vulkan0'), ('C','none'), ('A','Vulkan1'), ('B','Vulkan1')]:
        manager.test.define(name, extra=f'[runtime]\ndevice="{device}"\n')
    saved = {name: await manager.preload_native_model(name) for name in ['E','R','C']}
    pids = {name: next(row['pid'] for row in manager.native_snapshot() if row['model']==name) for name in saved}
    async with manager.acquire_native_model('A') as lease:
        incoming = asyncio.create_task(manager.preload_native_model('B'))
        await wait_until(lambda: manager._native._entries[lease.generation].retired)
        assert lease.worker.process.returncode is None and not incoming.done()
        assert (await lease.worker.completion({'prompt':'still active'}))['content']
    replacement = await incoming
    assert lease.worker.process.returncode is not None and not lease.worker.runtime_directory.exists()
    assert replacement != lease.generation
    for name, generation in saved.items():
        assert await manager.preload_native_model(name) == generation
        assert next(row['pid'] for row in manager.native_snapshot() if row['model']==name) == pids[name]
    assert len(manager.native_snapshot()) == 4
