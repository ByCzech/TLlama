"""Source selection uses independent snapshots and never fills native gaps."""

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload


def test_native_preferred_and_local_receipt_age_does_not_rewrite_native_age(payload):
    now = [10.0]
    telemetry = MemoryTelemetry(observe(), clock=lambda: now[0])
    telemetry.accept(NativeMemoryReport(payload))
    now[0] = 12.0
    snapshot = telemetry.snapshot()
    assert snapshot["source"] == "native_memory_api"
    assert snapshot["measurement"]["observation_age_ms"] == 2000
    assert snapshot["measurement"]["report"]["snapshot_age_ms"] == 23
    assert snapshot["measurement"]["totals"]["allocated_bytes"] == 1350
    assert "comparison" not in snapshot
    snapshot["measurement"]["report"]["buffers"].clear()
    assert telemetry.snapshot()["measurement"]["totals"]["allocated_bytes"] == 1350


@pytest.mark.parametrize("status", ["partial", "unavailable"])
def test_valid_incomplete_native_report_does_not_fall_back_to_complete_log(payload, status):
    payload["status"] = status
    telemetry = MemoryTelemetry(observe())
    telemetry.accept(NativeMemoryReport(payload))
    snapshot = telemetry.snapshot()
    assert snapshot["source"] == "native_memory_api" and snapshot["fallback_reason"] is None
    assert snapshot["measurement"]["totals"]["allocated_bytes"] is None


@pytest.mark.parametrize("mode,source", [("auto", "native_load_log"), ("native", None), ("log", "native_load_log")])
def test_failures_select_only_the_configured_fallback(payload, mode, source):
    telemetry = MemoryTelemetry(observe(), mode=mode)
    telemetry.accept(NativeMemoryReport(payload))
    telemetry.fail("not_ready", http_status=503)
    snapshot = telemetry.snapshot()
    assert snapshot["source"] == source
    assert snapshot["native"]["reason"] == "not_ready"
    assert snapshot["native"]["support"] == "supported"
    assert snapshot["fallback_reason"] == ("not_ready" if mode == "auto" else None)
    if mode == "native":
        assert snapshot["measurement"] is None
    else:
        assert snapshot["measurement"]["report"]["scope"] == "initial_load_allocations"


def test_comparison_keeps_both_scopes_and_sizes_without_combining(payload):
    telemetry = MemoryTelemetry(observe(), mode="log", compare=True)
    telemetry.accept(NativeMemoryReport(payload))
    snapshot = telemetry.snapshot()
    assert snapshot["source"] == "native_load_log"
    native, log = snapshot["comparison"]["native"], snapshot["comparison"]["log"]
    assert native["report"]["precision"] == "exact_bytes"
    assert log["report"]["precision"] == "rounded_mib"
    assert native["totals"]["allocated_bytes"] != log["report"]["allocated_bytes"]


def test_regressing_sequence_is_rejected_but_later_snapshot_recovers(payload):
    telemetry = MemoryTelemetry(observe())
    telemetry.accept(NativeMemoryReport(payload))
    payload["snapshot_sequence"] -= 1
    telemetry.accept(NativeMemoryReport(payload))
    assert telemetry.snapshot()["fallback_reason"] == "snapshot_sequence_regressed"
    payload["snapshot_sequence"] += 2
    telemetry.accept(NativeMemoryReport(payload))
    assert telemetry.snapshot()["source"] == "native_memory_api"


@pytest.mark.parametrize("reason,status,query_again", [
    ("endpoint_not_supported", 404, False), ("authentication_failed", 401, True),
    ("not_ready", 503, True), ("unsupported_schema", 200, True), ("timeout", None, True),
])
def test_only_missing_endpoint_is_negative_cached(reason, status, query_again):
    telemetry = MemoryTelemetry(observe())
    telemetry.fail(reason, http_status=status)
    assert telemetry.needs_native is query_again


def test_log_mode_does_not_query_native_without_comparison():
    assert not MemoryTelemetry(observe(), mode="log").needs_native
    assert MemoryTelemetry(observe(), mode="log", compare=True).needs_native
