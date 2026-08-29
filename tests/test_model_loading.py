"""Loading a model must not stop the rest of the server.

The load is an await of tens of seconds on a large file. Holding a lock
across it meant that loading one model blocked listings, metadata reads and
work on models already resident. Moving it out introduces concurrency that
has to be pinned: two callers wanting the same model, a caller wanting a
different one with no room for it, and every way a load can fail.
"""

import asyncio

import pytest


class FakeLlama:
    def __init__(self, name, n_ctx=4096):
        self.name = name
        self._n_ctx = n_ctx

    def n_ctx(self):
        return self._n_ctx


@pytest.fixture
def loadable(manager, gguf_file, toml_file, monkeypatch):
    """Replace the blocking load with one whose timing the test controls."""
    state = {"started": [], "finished": [], "hold": None, "fail": None}

    def make(name):
        gguf_file(f"Local/{name}.gguf")
        toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

    def load(model_path, requested_n_ctx, virtual_spec=None):
        name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        state["started"].append(name)

        if state["hold"] is not None:
            state["hold"].wait(timeout=5)
        if state["fail"] is not None:
            raise state["fail"]

        state["finished"].append(name)
        return FakeLlama(name, requested_n_ctx or 4096), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)
    state["make"] = make
    return state


async def test_a_model_loads_and_is_returned(manager, loadable):
    loadable["make"]("alpha")

    llm = await manager.get_model("alpha")

    assert llm.name == "alpha"
    assert loadable["started"] == ["alpha"]


async def test_the_lock_is_free_while_a_model_loads(manager, loadable):
    """The whole point: unrelated work must not queue behind a load."""
    import threading

    loadable["make"]("alpha")
    loadable["hold"] = threading.Event()

    loading = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)

    assert not manager._models_lock.locked()
    assert "alpha" in manager._loading

    loadable["hold"].set()
    await loading


async def test_concurrent_requests_for_one_model_load_it_once(manager, loadable):
    loadable["make"]("alpha")

    models = await asyncio.gather(*(manager.get_model("alpha") for _ in range(5)))

    assert loadable["started"] == ["alpha"]
    assert {id(model) for model in models} == {id(models[0])}


async def test_a_second_model_waits_when_there_is_no_room(make_manager, gguf_file, toml_file, monkeypatch):
    """A single-model limit serialises rather than overshooting it."""
    import threading

    manager = make_manager(max_loaded_models=1)
    for name in ("alpha", "beta"):
        gguf_file(f"Local/{name}.gguf")
        toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

    resident = []
    hold = threading.Event()

    def load(model_path, requested_n_ctx, virtual_spec=None):
        name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        hold.wait(timeout=5)
        resident.append(len(manager.models))
        return FakeLlama(name), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)

    first = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(manager.get_model("beta"))
    await asyncio.sleep(0.05)

    hold.set()
    await asyncio.gather(first, second)

    assert len(manager.models) == 1
    assert list(manager.models) == ["beta"]


async def test_concurrent_loads_cannot_exceed_the_limit(make_manager, gguf_file, toml_file, monkeypatch):
    """Counting only resident models would let every caller pass the check."""
    import threading

    manager = make_manager(max_loaded_models=2)
    names = ["alpha", "beta", "gamma", "delta"]
    for name in names:
        gguf_file(f"Local/{name}.gguf")
        toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

    hold = threading.Event()
    peak = 0

    def load(model_path, requested_n_ctx, virtual_spec=None):
        nonlocal peak
        name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        peak = max(peak, len(manager.models) + len(manager._loading))
        hold.wait(timeout=5)
        return FakeLlama(name), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)

    tasks = [asyncio.create_task(manager.get_model(name)) for name in names]
    await asyncio.sleep(0.05)
    hold.set()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert peak <= 2
    assert len(manager.models) <= 2
    assert any(isinstance(result, RuntimeError) for result in results)


async def test_a_failed_load_is_reported_and_leaves_no_trace(manager, loadable):
    loadable["make"]("alpha")
    loadable["fail"] = RuntimeError("unable to load model")

    with pytest.raises(RuntimeError, match="unable to load model"):
        await manager.get_model("alpha")

    assert manager._loading == {}
    assert "alpha" not in manager.models


async def test_a_failed_load_does_not_strand_a_concurrent_waiter(manager, loadable):
    """Both callers have to hear about it, and the slot has to be released."""
    import threading

    loadable["make"]("alpha")
    loadable["hold"] = threading.Event()
    loadable["fail"] = RuntimeError("unable to load model")

    first = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)

    loadable["hold"].set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert manager._loading == {}


async def test_a_cancelled_loader_does_not_cancel_a_waiter(manager, loadable):
    """A client disconnecting mid-load must not take the others with it."""
    import threading

    loadable["make"]("alpha")
    loadable["hold"] = threading.Event()

    first = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    loadable["hold"].set()
    llm = await asyncio.wait_for(second, timeout=5)

    assert llm.name == "alpha"
    assert manager._loading == {}


async def test_an_absent_model_still_fails_immediately(manager, loadable):
    with pytest.raises(FileNotFoundError):
        await manager.get_model("no-such-model")

    assert manager._loading == {}
    assert loadable["started"] == []


async def test_a_resident_model_is_returned_without_loading_again(manager, loadable):
    loadable["make"]("alpha")
    first = await manager.get_model("alpha")

    second = await manager.get_model("alpha")

    assert second is first
    assert loadable["started"] == ["alpha"]


async def test_a_doomed_load_is_attempted_once_for_all_waiters(manager, loadable):
    """Repeating a load that just failed, once per waiter, helps nobody."""
    import threading

    loadable["make"]("alpha")
    loadable["hold"] = threading.Event()
    loadable["fail"] = RuntimeError("unable to load model")

    callers = [asyncio.create_task(manager.get_model("alpha")) for _ in range(4)]
    await asyncio.sleep(0.05)
    loadable["hold"].set()

    results = await asyncio.gather(*callers, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert loadable["started"] == ["alpha"]


async def test_a_failure_elsewhere_does_not_fail_this_request(make_manager, gguf_file, toml_file, monkeypatch):
    """Waiting for room is not the same as waiting for your own model."""
    import threading

    manager = make_manager(max_loaded_models=1)
    for name in ("alpha", "beta"):
        gguf_file(f"Local/{name}.gguf")
        toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

    hold = threading.Event()

    def load(model_path, requested_n_ctx, virtual_spec=None):
        name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        if name == "alpha":
            hold.wait(timeout=5)
            raise RuntimeError("unable to load model")
        return FakeLlama(name), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)

    failing = asyncio.create_task(manager.get_model("alpha"))
    await asyncio.sleep(0.05)
    unrelated = asyncio.create_task(manager.get_model("beta"))
    await asyncio.sleep(0.05)
    hold.set()

    with pytest.raises(RuntimeError):
        await failing

    llm = await asyncio.wait_for(unrelated, timeout=5)
    assert llm.name == "beta"
