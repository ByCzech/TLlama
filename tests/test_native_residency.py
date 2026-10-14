"""Policy races use controlled workers; socket/process cases are tested separately."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tllama.config import BackendConfig
from tllama.native_config import resolve_worker_config
from tllama.residency import (NativeResidency, ModelDefinition, ModelUnloaded,
                               ModelCapacityError, ModelCleanupError, ResidencyClosed)


class ControlledWorker:
    def __init__(self, spec, ledger):
        self.spec, self.ledger = spec, ledger
        self.pid = len(ledger) + 1
        self.state = "new"
        self.process = SimpleNamespace(returncode=None)
        self.props = {"default_generation_settings": {"n_ctx": spec.n_ctx or 4096}}
        self.diagnostics = "controlled worker"
        self.starting = asyncio.Event()
        self.allow_start = asyncio.Event()
        self.allow_start.set()
        self.allow_close = asyncio.Event()
        self.allow_close.set()
        self.exited = asyncio.Event()
        self.start_error = None
        self.close_error = None
        ledger.append(self)

    async def start(self):
        self.state = "starting"
        self.starting.set()
        await self.allow_start.wait()
        if self.start_error:
            raise self.start_error
        self.state = "ready"
        return self

    async def close(self):
        if self.close_error:
            raise self.close_error
        self.state = "closing"
        await self.allow_close.wait()
        self.state = "closed"
        if self.process.returncode is None:
            self.process.returncode = 0
        self.exited.set()

    async def wait_for_exit(self):
        await self.exited.wait()
        return self.process.returncode

    def crash(self):
        self.process.returncode = 71
        self.state = "failed"
        self.exited.set()


@pytest.fixture
async def pool():
    ledger, definitions = [], {}
    now = [100.0]
    options = {"hold_load": False, "hold_close": False, "load_error": None}
    config = BackendConfig(runtime_overrides={"device": "none"})

    async def resolve(name, ctx):
        value = definitions.get(name, 1)
        if isinstance(value, Exception):
            raise value
        resolved = resolve_worker_config(config, Path(f"/models/{name}.gguf"), num_ctx=ctx)
        return ModelDefinition(name, resolved, value)

    def factory(spec):
        worker = ControlledWorker(spec, ledger)
        if options["hold_load"]:
            worker.allow_start.clear()
        if options["hold_close"]:
            worker.allow_close.clear()
        worker.start_error = options["load_error"]
        return worker

    manager = NativeResidency(resolve, worker_factory=factory, clock=lambda: now[0])
    manager.test = SimpleNamespace(ledger=ledger, definitions=definitions, now=now, options=options)
    yield manager
    for worker in ledger:
        worker.allow_start.set()
        worker.allow_close.set()
        worker.close_error = None
    await manager.shutdown()


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


async def test_concurrent_loads_share_one_owned_task(pool):
    pool.test.options["hold_load"] = True

    async def use():
        async with pool.lease("a") as lease:
            return lease.generation

    callers = [asyncio.create_task(use()) for _ in range(5)]
    await wait_until(lambda: pool.snapshot() and pool.snapshot()[0]["leases"] == 5)
    pool.test.ledger[0].allow_start.set()
    assert len(set(await asyncio.gather(*callers))) == 1
    assert len(pool.test.ledger) == pool.occupied == 1


async def test_cancel_one_waiter_keeps_the_other_and_shared_load(pool):
    pool.test.options["hold_load"] = True
    first = asyncio.create_task(pool.preload("a"))
    second = asyncio.create_task(pool.preload("a"))
    await wait_until(lambda: pool.snapshot() and pool.snapshot()[0]["leases"] == 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert pool.snapshot()[0]["leases"] == 1
    pool.test.ledger[0].allow_start.set()
    await second
    assert len(pool.test.ledger) == 1


async def test_last_cancelled_waiter_leaves_manager_owned_load_to_finish(pool):
    pool.test.options["hold_load"] = True
    caller = asyncio.create_task(pool.preload("a", keep_alive_seconds=3))
    await wait_until(lambda: pool.test.ledger)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert pool.snapshot()[0]["leases"] == 0
    pool.test.ledger[0].allow_start.set()
    await wait_until(lambda: pool.snapshot()[0]["state"] == "ready")
    assert pool.snapshot()[0]["ttl_remaining"] == 3


async def test_cancel_after_admission_before_lease_delivery_releases_reference(pool, monkeypatch):
    entered, proceed = asyncio.Event(), asyncio.Event()
    forget = pool._forget_intent

    async def delayed(intent):
        entered.set()
        await proceed.wait()
        await forget(intent)

    monkeypatch.setattr(pool, "_forget_intent", delayed)
    caller = asyncio.create_task(pool.preload("a"))
    await entered.wait()
    assert pool.snapshot()[0]["leases"] == 1
    caller.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert pool.snapshot()[0]["leases"] == 0


async def test_failed_load_reaches_all_waiters_once_and_releases_capacity(pool):
    pool.test.options.update(hold_load=True, load_error=RuntimeError("bad kernels"))
    callers = [asyncio.create_task(pool.preload("a")) for _ in range(3)]
    await wait_until(lambda: pool.snapshot() and pool.snapshot()[0]["leases"] == 3)
    pool.test.ledger[0].allow_start.set()
    results = await asyncio.gather(*callers, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) and "bad kernels" in str(result) for result in results)
    assert len(pool.test.ledger) == 1 and pool.occupied == 0
    assert pool.failures[-1]["phase"] == "load"


async def test_explicit_unload_cancels_loading_and_notifies_every_waiter(pool):
    pool.test.options["hold_load"] = True
    callers = [asyncio.create_task(pool.preload("a")) for _ in range(2)]
    await wait_until(lambda: pool.snapshot() and pool.snapshot()[0]["leases"] == 2)
    assert await pool.unload("a") is True
    results = await asyncio.gather(*callers, return_exceptions=True)
    assert all(isinstance(result, ModelUnloaded) for result in results)
    await pool.wait_closed(1)
    assert pool.occupied == 0 and pool.test.ledger[0].state == "closed"


async def test_unload_while_definition_is_being_read_prevents_late_spawn(pool, monkeypatch):
    entered, proceed = asyncio.Event(), asyncio.Event()
    resolve = pool._resolve

    async def delayed(name, ctx):
        entered.set()
        await proceed.wait()
        return await resolve(name, ctx)

    monkeypatch.setattr(pool, "_resolve", delayed)
    caller = asyncio.create_task(pool.preload("a"))
    await entered.wait()
    assert await pool.unload("a") is True
    proceed.set()
    with pytest.raises(ModelUnloaded):
        await caller
    assert pool.test.ledger == [] and pool.occupied == 0


async def test_concurrent_slow_definition_reader_inherits_failed_load_without_retry(pool, monkeypatch):
    pool.test.options.update(hold_load=True, load_error=RuntimeError("bad kernels"))
    entered, proceed = asyncio.Event(), asyncio.Event()
    resolve = pool._resolve
    reads = [0]

    async def delayed(name, ctx):
        reads[0] += 1
        if reads[0] == 2:
            entered.set()
            await proceed.wait()
        return await resolve(name, ctx)

    monkeypatch.setattr(pool, "_resolve", delayed)
    first = asyncio.create_task(pool.preload("a"))
    await wait_until(lambda: pool.test.ledger)
    second = asyncio.create_task(pool.preload("a"))
    await entered.wait()
    pool.test.ledger[0].allow_start.set()
    with pytest.raises(RuntimeError, match="bad kernels"):
        await first
    proceed.set()
    with pytest.raises(RuntimeError, match="bad kernels"):
        await second
    assert len(pool.test.ledger) == 1
    pool.test.options.update(hold_load=False, load_error=None)
    await pool.preload("a")
    assert len(pool.test.ledger) == 2


async def test_unload_before_load_coroutine_gets_its_first_step_cleans_reservation(pool, monkeypatch):
    original = pool._load_entry

    async def late_load(entry):
        await asyncio.sleep(100)
        return await original(entry)

    monkeypatch.setattr(pool, "_load_entry", late_load)
    caller = asyncio.create_task(pool.preload("a"))
    await wait_until(lambda: pool.occupied == 1)
    await pool.unload("a")
    result = (await asyncio.gather(caller, return_exceptions=True))[0]
    assert isinstance(result, ModelUnloaded)
    await asyncio.wait_for(pool.wait_closed(1), 2)
    assert pool.test.ledger[0].state == "closed"


async def test_unload_active_lease_keeps_old_worker_alive_until_release(pool):
    async with pool.lease("a") as lease:
        assert await pool.unload("a") is True
        assert lease.worker.state == "ready"
        assert pool.snapshot()[0]["state"] == "draining"
        assert pool.snapshot()[0]["leases"] == 1
    await pool.wait_closed(lease.generation)
    assert lease.worker.state == "closed" and pool.occupied == 0


async def test_single_model_replacement_waits_for_lease_and_actual_close(pool):
    pool.test.options["hold_close"] = True
    async with pool.lease("a") as lease:
        other = asyncio.create_task(pool.preload("b"))
        await wait_until(lambda: pool.snapshot()[0]["state"] == "draining")
        assert len(pool.test.ledger) == 1 and not other.done()
    await wait_until(lambda: lease.worker.state == "closing")
    assert pool.occupied == 1 and len(pool.test.ledger) == 1
    lease.worker.allow_close.set()
    await other
    assert len(pool.test.ledger) == 2 and pool.occupied == 1


async def test_larger_limit_preserves_error_policy_for_ready_models(pool):
    pool.max_loaded_models = 2
    await pool.preload("a")
    await pool.preload("b")
    with pytest.raises(ModelCapacityError, match="limit reached"):
        await pool.preload("c")
    assert len(pool.test.ledger) == 2


async def test_draining_old_release_does_not_unload_new_generation_of_same_name(pool):
    pool.max_loaded_models = 2
    async with pool.lease("a") as old:
        pool.test.definitions["a"] = 2
        async with pool.lease("a") as new:
            assert old.generation != new.generation
            assert pool.occupied == 2
    await pool.wait_closed(old.generation)
    assert new.worker.state == "ready" and pool.occupied == 1


async def test_different_contexts_do_not_share_a_load_or_wrong_worker(pool):
    pool.max_loaded_models = 2
    async with pool.lease("a", num_ctx=2048) as one:
        async with pool.lease("a", num_ctx=4096) as two:
            assert one.worker.spec.n_ctx == 2048 and two.worker.spec.n_ctx == 4096
            assert one.generation != two.generation
    await pool.wait_closed(one.generation)


async def test_replaced_loading_snapshot_can_finish_for_admitted_caller(pool):
    pool.max_loaded_models = 2
    pool.test.options["hold_load"] = True
    first = asyncio.create_task(pool.preload("a"))
    await wait_until(lambda: pool.test.ledger)
    pool.test.definitions["a"] = 2
    second = asyncio.create_task(pool.preload("a"))
    await wait_until(lambda: len(pool.test.ledger) == 2)
    for worker in pool.test.ledger:
        worker.allow_start.set()
    numbers = await asyncio.gather(first, second)
    assert numbers == [1, 2]
    await pool.wait_closed(1)
    assert pool.test.ledger[1].state == "ready"


async def test_ttl_starts_after_last_lease_even_when_usage_exceeds_it(pool):
    async with pool.lease("a", keep_alive_seconds=5) as lease:
        pool.test.now[0] += 100
        await pool.expire()
        assert lease.worker.state == "ready" and pool.snapshot()[0]["ttl_remaining"] is None
    assert pool.snapshot()[0]["ttl_remaining"] == 5
    pool.test.now[0] += 4
    await pool.expire()
    assert pool.snapshot()[0]["state"] == "ready"
    pool.test.now[0] += 1
    await pool.expire()
    await pool.wait_closed(lease.generation)
    assert pool.occupied == 0


async def test_latest_admitted_keep_alive_wins_regardless_of_release_order(pool):
    async with pool.lease("a", keep_alive_seconds=5):
        async with pool.lease("a", keep_alive_seconds=None):
            pass
    pool.test.now[0] += 1000
    await pool.expire()
    assert pool.snapshot()[0]["keep_alive"] is None and pool.occupied == 1


async def test_zero_keep_alive_closes_after_usage_and_empty_preload_only_unloads(pool):
    async with pool.lease("a", keep_alive_seconds=0) as lease:
        assert lease.worker.state == "ready"
    await pool.wait_closed(lease.generation)
    assert await pool.preload("a", keep_alive_seconds=0) is None
    assert len(pool.test.ledger) == 1


async def test_invalid_definition_drains_existing_model_but_preserves_lease(pool):
    async with pool.lease("a") as lease:
        pool.test.definitions["a"] = FileNotFoundError("definition disappeared")
        with pytest.raises(FileNotFoundError):
            await pool.preload("a")
        assert lease.worker.state == "ready" and pool.snapshot()[0]["state"] == "draining"
    await pool.wait_closed(lease.generation)


async def test_crash_is_recorded_and_next_independent_request_can_reload(pool):
    async with pool.lease("a") as old:
        old.worker.crash()
        await wait_until(lambda: pool.failures)
        await pool.wait_closed(old.generation)
        assert pool.failures[-1]["phase"] == "exit"
        async with pool.lease("a") as new:
            assert new.generation != old.generation and new.worker is not old.worker
    assert len(pool.test.ledger) == 2


async def test_shutdown_cancels_load_and_capacity_waiter_without_new_spawn(pool):
    pool.test.options["hold_load"] = True
    first = asyncio.create_task(pool.preload("a"))
    second = asyncio.create_task(pool.preload("b"))
    await wait_until(lambda: pool.test.ledger)
    await pool.shutdown()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, (ResidencyClosed, ModelUnloaded)) for result in results)
    assert pool.occupied == 0 and len(pool.test.ledger) == 1
    with pytest.raises(ResidencyClosed):
        await pool.preload("a")


async def test_shutdown_forces_active_generation_closed_and_late_release_is_safe(pool):
    async with pool.lease("a") as lease:
        await pool.shutdown()
        assert lease.worker.state == "closed" and pool.occupied == 0
    assert pool.occupied == 0


async def test_cancelling_shutdown_waits_for_owned_cleanup(pool):
    pool.test.options["hold_close"] = True
    await pool.preload("a")
    task = asyncio.create_task(pool.shutdown())
    await wait_until(lambda: pool.test.ledger[0].state == "closing")
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    pool.test.ledger[0].allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.occupied == 0


async def test_cleanup_failure_cannot_free_capacity_by_removing_dict_entry(pool):
    number = await pool.preload("a")
    worker = pool.test.ledger[0]
    worker.close_error = RuntimeError("reap failed")
    await pool.unload("a")
    with pytest.raises(ModelCleanupError):
        await pool.wait_closed(number)
    with pytest.raises(ModelCleanupError):
        await pool.preload("b")
    assert pool.occupied == 1 and len(pool.test.ledger) == 1
    # Repair the controlled cleanup for fixture teardown (no real process here).
    worker.close_error = None
    await worker.close()
    async with pool._condition:
        pool._finish_closed(pool._entries[number])
