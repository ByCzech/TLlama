"""ModelManager native leases exercise real subprocesses and loopback transport."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.residency import ModelUnloaded, ResidencyClosed
from tllama.worker import NativeWorker, WorkerExited, WorkerStartupError


@pytest.fixture
async def native_manager(tmp_path):
    executable = tmp_path / "native-stub"
    executable.write_text(f"#!{sys.executable}\n" + Path(__file__).with_name("native_worker_stub.py").read_text())
    executable.chmod(0o700)
    managers = []

    def make(*, limit=1, mode="normal", memory_report=None, **config_values):
        directory = tmp_path / str(len(managers))
        directory.mkdir()
        ledger, roots = [], []
        options = {"mode": mode}

        def factory(spec):
            root = directory / f"worker-{len(ledger)}"
            root.mkdir()
            if memory_report is not None:
                (root / "memory-report").write_text(json.dumps(memory_report))
            worker = NativeWorker(replace(spec, environment={**spec.environment, "STUB_ROOT": str(root),
                                                              "STUB_MODE": options["mode"]},
                                          terminate_timeout=0.2, kill_timeout=2))
            ledger.append(worker)
            roots.append(root)
            return worker

        config = BackendConfig(models_dir=str(directory / "models"), context_length=2048,
                               max_loaded_models=limit, runtime_overrides={"device": "none"},
                               **config_values)
        manager = ModelManager(config, native_worker_factory=factory,
                               native_launch_options={"executable": executable, "load_timeout": 5})
        manager.ensure_storage()

        def define(name="a", extra=""):
            path = manager.local_models_dir / f"{name}.gguf"
            path.write_bytes(b"fake weights")
            toml = manager.local_models_dir / f"{name}.toml"
            toml.write_text(f'[llm]\nmodel="Local/{name}.gguf"\n' + extra)
            return name, toml

        manager.test = SimpleNamespace(ledger=ledger, roots=roots, options=options, define=define)
        managers.append(manager)
        return manager

    yield make
    await asyncio.gather(*(manager.shutdown() for manager in managers))
    assert all(worker.process is None or worker.process.returncode is not None
               for manager in managers for worker in manager.test.ledger)


async def wait_until(predicate, timeout=3):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


def records(root, name):
    path = root / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def test_registered_model_resolves_and_concurrent_preloads_spawn_once(native_manager):
    manager = native_manager()
    name, _ = manager.test.define(extra='[runtime]\nn_ctx=4096\ntype_kv="q4_0"\nflash_attn=true')
    numbers = await asyncio.gather(*(manager.preload_native_model(name) for _ in range(5)))
    assert len(set(numbers)) == len(manager.test.ledger) == 1
    snapshot = manager.native_snapshot()[0]
    assert snapshot["requested_context"] == 4096 and snapshot["effective_context"] == 2048
    async with manager.acquire_native_model(name) as lease:
        assert lease.resolved.launch.type_k == lease.resolved.launch.type_v == "q4_0"
        assert (await lease.worker.chat({"messages": []}))["choices"]
    await manager.unload_native_model(name)
    await manager.wait_native_closed(numbers[0])
    worker = manager.test.ledger[0]
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


async def test_every_admitted_transport_waiter_keeps_lease_during_unload(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name) as first:
        async with first.worker.chat_stream({"stub": "hold"}) as events:
            await anext(events)

            async def queued():
                async with manager.acquire_native_model(name) as lease:
                    return await lease.worker.chat({})

            waiting = asyncio.create_task(queued())
            try:
                await wait_until(lambda: manager.native_snapshot()[0]["leases"] == 2)
                await manager.unload_native_model(name)
                assert first.worker.process.returncode is None
                assert manager.native_snapshot()[0]["state"] == "draining"
            finally:
                # Leave transport context first so the queued request can run.
                await events.aclose()
        assert (await waiting)["choices"]
        assert first.worker.process.returncode is None
    await manager.wait_native_closed(first.generation)
    assert records(manager.test.roots[0], "cancelled") == ["stream"]


async def test_cancel_queued_request_releases_reservation_and_does_not_lose_stream(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name) as lease:
        async with lease.worker.chat_stream({"stub": "hold"}) as events:
            await anext(events)

            async def queued():
                async with manager.acquire_native_model(name) as other:
                    await other.worker.chat({})

            pending = asyncio.create_task(queued())
            await wait_until(lambda: manager.native_snapshot()[0]["leases"] == 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert manager.native_snapshot()[0]["leases"] == 1
            await manager.unload_native_model(name)
            assert lease.worker.process.returncode is None
    await manager.wait_native_closed(lease.generation)


async def test_cancel_loading_client_does_not_cancel_owned_load_or_other_waiter(native_manager):
    manager = native_manager(mode="hang-load")
    name, _ = manager.test.define()
    first = asyncio.create_task(manager.preload_native_model(name))
    second = asyncio.create_task(manager.preload_native_model(name))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / "launch").exists())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert manager.test.ledger[0].process.returncode is None
    assert not second.done()
    await manager.unload_native_model(name)
    with pytest.raises(ModelUnloaded):
        await second
    await manager.wait_native_closed(1)
    assert len(manager.test.ledger) == 1


async def test_failed_native_start_is_reaped_and_a_new_request_can_recover(native_manager):
    manager = native_manager(mode="fatal")
    name, _ = manager.test.define()
    results = await asyncio.gather(*(manager.preload_native_model(name) for _ in range(3)), return_exceptions=True)
    assert all(isinstance(result, WorkerStartupError) for result in results)
    assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode == 23
    assert manager.native_snapshot() == []
    manager.test.options["mode"] = "normal"
    assert await manager.preload_native_model(name) == 2


@pytest.mark.parametrize("edit", ["touch", "sampling", "context", "weights"])
async def test_definition_change_creates_generation_and_old_release_is_safe(native_manager, edit):
    manager = native_manager(limit=2)
    name, path = manager.test.define()
    async with manager.acquire_native_model(name) as old:
        if edit == "touch":
            stamp = path.stat().st_mtime_ns
            os.utime(path, ns=(stamp + 1_000_000, stamp + 1_000_000))
        elif edit == "sampling":
            path.write_text(path.read_text() + "[sampling]\ntemperature=0.2")
        elif edit == "context":
            path.write_text(path.read_text() + "[runtime]\nn_ctx=4096")
        else:
            (manager.local_models_dir / "a.gguf").write_bytes(b"new fake weights")
        async with manager.acquire_native_model(name) as new:
            assert new.generation != old.generation and new.worker is not old.worker
            assert len(manager.native_snapshot()) == 2
    await manager.wait_native_closed(old.generation)
    assert new.worker.process.returncode is None
    assert [row["generation"] for row in manager.native_snapshot()] == [new.generation]


@pytest.mark.parametrize("invalid", ["deleted", "broken", "missing-weight"])
async def test_unreachable_definition_drains_but_cannot_kill_existing_lease(native_manager, invalid):
    manager = native_manager()
    name, path = manager.test.define()
    async with manager.acquire_native_model(name) as old:
        if invalid == "deleted":
            path.unlink()
        elif invalid == "broken":
            path.write_text("[llm\n")
        else:
            (manager.local_models_dir / "a.gguf").unlink()
        with pytest.raises((FileNotFoundError, ValueError)):
            await manager.preload_native_model(name)
        assert old.worker.process.returncode is None
        assert manager.native_snapshot()[0]["state"] == "draining"
    await manager.wait_native_closed(old.generation)


async def test_model_manager_janitor_uses_completion_based_native_ttl(native_manager):
    manager = native_manager(janitor_interval_seconds=0.01)
    name, _ = manager.test.define()
    clock = [100.0]
    manager._native._clock = lambda: clock[0]
    async with manager.acquire_native_model(name, keep_alive="1s") as lease:
        clock[0] += 100
        await asyncio.sleep(0.03)
        assert lease.worker.process.returncode is None
    assert manager.native_snapshot()[0]["ttl_remaining"] == 1
    clock[0] += 1
    await asyncio.wait_for(manager.wait_native_closed(lease.generation), 3)
    assert lease.worker.process.returncode is not None


async def test_active_crash_releases_capacity_and_is_not_replayed(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name) as old:
        os.kill(old.worker.pid, signal.SIGKILL)
        await old.worker.wait_for_exit()
        with pytest.raises(WorkerExited):
            await old.worker.chat({})
        await manager.wait_native_closed(old.generation)
        async with manager.acquire_native_model(name) as new:
            assert new.generation != old.generation
    assert len(manager.test.ledger) == 2
    assert not records(manager.test.roots[0], "requests")[-1]["path"].endswith("completions")


async def test_shutdown_cancels_start_and_does_not_publish_worker_afterwards(native_manager):
    manager = native_manager(mode="hang-load")
    name, _ = manager.test.define()
    pending = asyncio.create_task(manager.preload_native_model(name))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / "launch").exists())
    await manager.shutdown()
    with pytest.raises((ModelUnloaded, ResidencyClosed)):
        await pending
    assert manager.native_snapshot() == [] and manager.test.ledger[0].process.returncode is not None


async def test_shutdown_interrupts_active_http_and_late_lease_release_is_safe(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name) as lease:
        request = asyncio.create_task(lease.worker.chat({"stub": "hold"}))
        await wait_until(lambda: (manager.test.roots[0] / "started").exists())
        await manager.shutdown()
        with pytest.raises(RuntimeError):
            await request
    assert manager.native_snapshot() == []


async def test_restart_after_full_shutdown_does_not_reuse_generation_ids(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    old = await manager.preload_native_model(name)
    await manager.shutdown()
    await manager.start()
    new = await manager.preload_native_model(name)
    assert new > old
    await manager.wait_native_closed(old)
    assert manager.native_snapshot()[0]["generation"] == new


async def test_failed_janitor_cannot_skip_native_process_cleanup(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    await manager.preload_native_model(name)
    manager._janitor_task.cancel()
    await asyncio.gather(manager._janitor_task, return_exceptions=True)

    async def fail():
        raise RuntimeError("janitor failed")

    manager._janitor_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="janitor failed"):
        await manager.shutdown()
    assert manager.native_snapshot() == [] and manager.test.ledger[0].process.returncode is not None
    manager._shutdown_task = None  # Let fixture teardown repeat the already-completed cleanup.


@pytest.mark.parametrize("valid", [False, True])
async def test_managed_hardware_smoke_entry_point(native_manager, valid):
    manager = native_manager()
    name, _ = manager.test.define()
    repo = Path(__file__).resolve().parents[1]
    root = manager.models_dir.parent / "smoke-worker"
    root.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("TLLAMA_")}
    env.update(PYTHONPATH=str(repo / "src"), TLLAMA_MODELS=str(manager.models_dir),
               TLLAMA_RUNTIME_DEVICE="none", STUB_ROOT=str(root), STUB_MODE="normal",
               TLLAMA_RUNTIME_EXECUTABLE=str(manager._native_launch_options["executable"]))
    child = await asyncio.create_subprocess_exec(
        sys.executable, str(repo / "scripts" / "manager-smoke.py"), "--model", name if valid else "missing",
        "--load-timeout", "5",
        "--request-timeout", "2", cwd=repo, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), 10)
    finally:
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            await child.wait()
    stages = [json.loads(line) for line in stdout.splitlines()]
    if not valid:
        assert child.returncode == 1 and stages[-1]["stage"] == "error"
        assert not (root / "launch").exists()
        return
    assert child.returncode == 0, (stdout.decode(), stderr.decode())
    assert len(records(root, "launch")) == 1
    assert [stage["stage"] for stage in stages] == ["preloaded", "ready", "memory-ready", "chat",
        "memory-after-chat", "unload-during-stream", "memory-during-stream",
        "stream-finished", "worker-closed", "manager-closed"]
    measurements = [stage["telemetry"] for stage in stages if stage["stage"].startswith("memory-")]
    assert all(item["source"] == "native_load_log" and item["fallback_reason"] == "endpoint_not_supported"
               for item in measurements)
    assert stages[-2]["directory_removed"] is True and stages[-1]["residency"] == []
