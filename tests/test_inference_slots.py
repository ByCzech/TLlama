"""Generation on one model has to be serialised.

A llama_cpp.Llama owns a single context and a single KV cache, and neither
the constructor nor create_completion accepts a sequence id, so two
concurrent generations on one object corrupt each other. The upstream server
serialises every request globally for this reason; these tests pin the finer
per-model guarantee TLlama gives instead.
"""

import asyncio

import pytest


async def record_overlap(manager, model_name, ledger, hold=0.02):
    async with manager.acquire_inference_slot(model_name):
        ledger.append(("enter", model_name))
        await asyncio.sleep(hold)
        ledger.append(("leave", model_name))


def concurrency_peak(ledger, model_name):
    depth = peak = 0
    for event, name in ledger:
        if name != model_name:
            continue
        depth += 1 if event == "enter" else -1
        peak = max(peak, depth)
    return peak


async def test_two_generations_on_one_model_never_overlap(manager):
    ledger = []

    await asyncio.gather(*(record_overlap(manager, "m", ledger) for _ in range(4)))

    assert concurrency_peak(ledger, "m") == 1


async def test_different_models_are_not_serialised_against_each_other(manager):
    """Waiting on a model you did not ask for is the thing being avoided."""
    ledger = []

    await asyncio.gather(
        record_overlap(manager, "a", ledger),
        record_overlap(manager, "b", ledger),
    )

    interleaved = [name for _, name in ledger]

    assert interleaved != ["a", "a", "b", "b"]
    assert interleaved != ["b", "b", "a", "a"]


async def test_a_failed_generation_releases_the_slot(manager):
    with pytest.raises(RuntimeError):
        async with manager.acquire_inference_slot("m"):
            raise RuntimeError("llama_decode failed")

    await asyncio.wait_for(record_overlap(manager, "m", []), timeout=1.0)


async def test_a_cancelled_generation_releases_the_slot(manager):
    """A client disconnecting mid-stream must not strand the model."""
    started = asyncio.Event()

    async def hang():
        async with manager.acquire_inference_slot("m"):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(hang())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(record_overlap(manager, "m", []), timeout=1.0)


async def test_waiting_for_a_slot_does_not_block_the_event_loop(manager):
    """The wait has to be a suspension, not a stall."""
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    ticker = asyncio.create_task(tick())
    await asyncio.gather(*(record_overlap(manager, "m", [], hold=0.05) for _ in range(3)))
    ticker.cancel()

    assert ticks > 10


async def test_the_slot_registry_is_reused_not_rebuilt(manager):
    async with manager.acquire_inference_slot("m"):
        pass
    first = manager._inference_slots["m"]

    async with manager.acquire_inference_slot("m"):
        pass

    assert manager._inference_slots["m"] is first


def test_the_slot_count_is_one_and_says_why(manager):
    """When the binding layer gains sequence slots, this is what changes."""
    assert manager._inference_slots_for("m") == 1
    assert "n_seq_max" in manager._inference_slots_for.__doc__


async def test_more_slots_would_allow_more_concurrency(manager, monkeypatch):
    """The shape is a capacity, so raising it is the only change needed."""
    monkeypatch.setattr(manager, "_inference_slots_for", lambda name: 2)
    ledger = []

    await asyncio.gather(*(record_overlap(manager, "m", ledger) for _ in range(4)))

    assert concurrency_peak(ledger, "m") == 2


async def test_the_queue_is_first_in_first_out(manager):
    """Order of arrival decides order of service, not order of waking."""
    served = []

    async def arrive(number, delay):
        await asyncio.sleep(delay)
        async with manager.acquire_inference_slot("m"):
            served.append(number)
            await asyncio.sleep(0.02)

    # Zero takes the free slot; one through eight then queue in that order.
    await asyncio.gather(*(arrive(n, n * 0.002) for n in range(9)))

    assert served == list(range(9))


async def test_a_latecomer_cannot_take_a_slot_being_handed_over(manager):
    """The hand-off must not be a race a newly arriving request can win."""
    served = []
    queued_up = asyncio.Event()

    async def arrive(number):
        async with manager.acquire_inference_slot("m"):
            served.append(number)
            if number == 0:
                await queued_up.wait()
            await asyncio.sleep(0.02)

    holder = asyncio.create_task(arrive(0))
    await asyncio.sleep(0)
    waiting = [asyncio.create_task(arrive(n)) for n in (1, 2, 3)]
    await asyncio.sleep(0.01)

    latecomer = asyncio.create_task(arrive(99))
    await asyncio.sleep(0.01)
    queued_up.set()

    await asyncio.gather(holder, *waiting, latecomer)

    assert served == [0, 1, 2, 3, 99]
