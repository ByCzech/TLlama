"""Memory HTTP failures and slow reads must not interfere with inference."""

import asyncio
import json
import logging

import pytest

from tllama.worker import WorkerStartupError
from test_native_worker import factory, records
from test_native_memory_api import payload


def prepare(root, payload, status=200):
    payload["runtime_fingerprint"] = "b11552-ec2797836f"
    (root / "memory-report").write_text(json.dumps(payload))
    (root / "memory-status").write_text(str(status))


async def settle(worker):
    async def wait():
        while worker.memory_snapshot()["native"]["reason"] == "not_observed":
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("mode", ["wrong-build", "missing-build"])
async def test_unknown_version_warns_and_inference_still_works(factory, caplog, mode):
    worker, _ = factory(mode, memory_source="log")
    with caplog.at_level(logging.WARNING):
        async with worker:
            assert (await worker.chat({}))["choices"]
            report = worker.memory.snapshot()
            assert report["runtime_fingerprint"] == ("wrong" if mode == "wrong-build" else None)
            assert report["runtime_verified"] is False
    assert "Unverified native runtime fingerprint" in caplog.text


@pytest.mark.parametrize("mode", ["wrong-build", "missing-build"])
async def test_strict_version_policy_still_rejects_unknown_build(factory, mode):
    worker, _ = factory(mode, version_policy="strict")
    with pytest.raises(WorkerStartupError, match="Unsupported runtime fingerprint"):
        await worker.start()
    assert not worker.runtime_directory.exists() and worker.process.returncode is not None


async def test_memory_api_runtime_is_supported_in_strict_mode(factory):
    worker, _ = factory(version_policy="strict", environment={"STUB_BUILD": "b11552-d1570abd3d"})
    async with worker:
        assert worker.runtime_fingerprint == "b11552-d1570abd3d"
        assert worker.memory.snapshot()["runtime_fingerprint"] == worker.runtime_fingerprint
        assert worker.memory.snapshot()["status"] == "known"
        assert worker.memory.snapshot()["runtime_verified"] is True


async def test_v1_runtime_log_fallback_completes_when_endpoint_is_missing(factory):
    worker, root = factory(environment={"STUB_BUILD": "b11552-d1570abd3d"})
    async with worker:
        await settle(worker)
        snapshot = worker.memory_snapshot()
        assert snapshot["source"] == "native_load_log"
        assert snapshot["fallback_reason"] == "endpoint_not_supported"
        assert snapshot["measurement"]["report"]["status"] == "known"
        assert snapshot["measurement"]["report"]["allocated_bytes"] == int(16.5 * 1024**2)
        assert (await worker.chat({}))["choices"]


async def test_compatible_experimental_build_can_use_native_memory(factory, payload):
    worker, root = factory("wrong-build")
    prepare(root, payload)
    payload["runtime_fingerprint"] = "wrong"
    (root / "memory-report").write_text(json.dumps(payload))
    async with worker:
        await settle(worker)
        assert worker.memory_snapshot()["source"] == "native_memory_api"
        assert worker.memory_snapshot()["measurement"]["report"]["runtime_fingerprint"] == "wrong"
        assert (await worker.chat({}))["choices"]


async def test_authenticated_native_snapshot_and_log_comparison_are_separate(factory, payload):
    worker, root = factory(memory_compare=True)
    prepare(root, payload)
    async with worker:
        await settle(worker)
        snapshot = worker.memory_snapshot()
        assert snapshot["source"] == "native_memory_api"
        assert snapshot["measurement"]["totals"]["allocated_bytes"] == 1350
        assert snapshot["comparison"]["log"]["report"]["allocated_bytes"] == int(16.5 * 1024**2)
        request = next(r for r in records(root, "requests") if r["path"] == "/tllama/memory")
        assert request["body"] == {} and "authorization" in request["header_names"]
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("status", ["partial", "unavailable"])
async def test_incomplete_native_http_report_is_selected_without_log_substitution(factory, payload, status):
    worker, root = factory()
    payload["status"] = status
    prepare(root, payload)
    async with worker:
        await settle(worker)
        snapshot = worker.memory_snapshot()
        assert snapshot["source"] == "native_memory_api"
        assert snapshot["measurement"]["report"]["status"] == status
        assert snapshot["measurement"]["totals"]["allocated_bytes"] is None
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("status,reason", [
    (401, "authentication_failed"), (404, "endpoint_not_supported"),
    (503, "not_ready"), (500, "http_error"), (302, "http_error"),
])
@pytest.mark.parametrize("source", ["auto", "native"])
async def test_http_failure_never_kills_inference_and_404_alone_is_cached(factory, payload, status, reason, source):
    worker, root = factory(memory_source=source)
    prepare(root, payload, status)
    async with worker:
        await settle(worker)
        before = len(records(root, "requests"))
        snapshot = worker.memory_snapshot()
        assert snapshot["native"]["reason"] == reason
        assert snapshot["native"]["http_status"] == status
        assert snapshot["native"]["support"] == ("unsupported" if status == 404 else "unknown")
        assert snapshot["source"] == ("native_load_log" if source == "auto" else None)
        await worker.refresh_memory()
        assert len(records(root, "requests")) == before + (0 if status == 404 else 1)
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("failure,reason", [
    ("schema", "unsupported_schema"), ("invalid", "invalid_report"),
    ("json", "invalid_response"), ("identity", "runtime_fingerprint_mismatch"),
    ("oversized", "invalid_response"),
])
async def test_bad_native_reports_fall_back_without_replaying_requests(factory, payload, failure, reason):
    worker, root = factory()
    prepare(root, payload)
    if failure == "schema":
        payload["schema_version"] = 2
    elif failure == "invalid":
        payload["buffers"][0]["size_bytes"] = -1
    elif failure == "identity":
        payload["runtime_fingerprint"] = "another-process-build"
    (root / "memory-report").write_text("{" if failure == "json" else
                                        " " * (16 * 1024**2 + 1) if failure == "oversized" else json.dumps(payload))
    async with worker:
        await settle(worker)
        assert worker.memory_snapshot()["fallback_reason"] == reason
        assert (await worker.chat({}))["choices"]
        assert len([r for r in records(root, "requests") if r["path"] == "/v1/chat/completions"]) == 1


async def test_temporary_failure_discards_old_known_metrics_and_recovers(factory, payload):
    worker, root = factory(memory_source="native")
    prepare(root, payload)
    async with worker:
        await settle(worker)
        assert worker.memory_snapshot()["measurement"]["totals"]["allocated_bytes"] == 1350
        (root / "memory-status").write_text("503")
        await worker.refresh_memory()
        assert worker.memory_snapshot()["measurement"] is None
        (root / "memory-status").write_text("200")
        payload["snapshot_sequence"] += 1
        payload["buffers"][0]["size_bytes"] = 2000
        (root / "memory-report").write_text(json.dumps(payload))
        await worker.refresh_memory()
        assert worker.memory_snapshot()["measurement"]["totals"]["allocated_bytes"] == 2350


@pytest.mark.parametrize("compare", [False, True])
async def test_explicit_log_mode_respects_optional_comparison(factory, payload, compare):
    worker, root = factory(memory_source="log", memory_compare=compare)
    prepare(root, payload)
    async with worker:
        if compare:
            await settle(worker)
        await worker.refresh_memory()
        snapshot = worker.memory_snapshot()
        assert snapshot["source"] == "native_load_log"
        assert ("comparison" in snapshot) is compare
        assert any(r["path"] == "/tllama/memory" for r in records(root, "requests")) is compare


async def test_memory_reads_during_stream_do_not_wait_for_inference_slot(factory, payload):
    worker, root = factory()
    prepare(root, payload)
    async with worker:
        await settle(worker)
        async with worker.chat_stream({"stub": "hold"}) as events:
            await anext(events)
            await asyncio.wait_for(worker.refresh_memory(), 0.5)
            assert worker.memory_snapshot()["source"] == "native_memory_api"
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("behavior,reason", [("hang", "timeout"), ("disconnect", "transport_error")])
async def test_stalled_memory_does_not_block_start_or_inference(factory, payload, behavior, reason):
    worker, root = factory()
    prepare(root, payload)
    (root / "memory-behavior").write_text(behavior)
    async with worker:
        assert (await asyncio.wait_for(worker.chat({}), 0.5))["choices"]
        await settle(worker)
        assert worker.memory_snapshot()["fallback_reason"] == reason
    assert worker._memory_task.done() and worker._memory_client.is_closed


async def test_close_cancels_hanging_telemetry_and_reaps_child(factory, payload):
    worker, root = factory()
    prepare(root, payload)
    (root / "memory-behavior").write_text("hang")
    await worker.start()
    while not any(r["path"] == "/tllama/memory" for r in records(root, "requests")):
        await asyncio.sleep(0.01)
    await asyncio.wait_for(worker.close(), 0.5)
    assert worker._memory_task.done() and worker._memory_client.is_closed
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


async def test_unexpected_collector_error_does_not_break_inference(factory, payload, monkeypatch):
    worker, root = factory()
    prepare(root, payload)
    async def broken():
        raise RuntimeError("collector failed")
    monkeypatch.setattr(worker, "_fetch_memory", broken)
    async with worker:
        await settle(worker)
        assert worker.memory_snapshot()["fallback_reason"] == "collection_failed"
        assert (await worker.chat({}))["choices"]


async def test_native_metrics_survive_a_broken_log_collector(factory, payload, monkeypatch):
    worker, root = factory(memory_compare=True)
    prepare(root, payload)
    def broken(_):
        raise RuntimeError("parser failed")
    monkeypatch.setattr(worker.memory, "observe", broken)
    async with worker:
        await settle(worker)
        snapshot = worker.memory_snapshot()
        assert snapshot["source"] == "native_memory_api"
        assert snapshot["comparison"]["log"]["report"]["reason"] == "parser_error"
        assert (await worker.chat({}))["choices"]


async def test_optional_client_creation_failure_does_not_fail_start(factory, monkeypatch):
    import tllama.worker as module
    create = module.httpx.AsyncClient
    def client(**kwargs):
        if kwargs.get("limits").max_connections == 1:
            raise RuntimeError("optional client failed")
        return create(**kwargs)
    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    worker, _ = factory()
    async with worker:
        assert worker.memory_snapshot()["fallback_reason"] == "collection_failed"
        assert (await worker.chat({}))["choices"]


async def test_optional_client_close_failure_cannot_skip_process_cleanup(factory, payload, monkeypatch):
    worker, root = factory()
    prepare(root, payload)
    await worker.start()
    await settle(worker)
    close = worker._memory_client.aclose
    async def broken():
        await close()
        raise RuntimeError("optional close failed")
    monkeypatch.setattr(worker._memory_client, "aclose", broken)
    await worker.close()
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


async def test_polling_updates_allocations_without_an_inference_or_reporting_call(factory, payload):
    worker, root = factory()
    prepare(root, payload)
    async with worker:
        await settle(worker)
        payload["snapshot_sequence"] += 1
        payload["buffers"][0]["size_bytes"] = 2000
        (root / "memory-report").write_text(json.dumps(payload))
        async def changed():
            while worker.memory_snapshot()["measurement"]["totals"]["allocated_bytes"] != 2350:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(changed(), 2)
        assert not any(r["path"] == "/v1/chat/completions" for r in records(root, "requests"))
