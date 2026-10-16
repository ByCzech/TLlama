"""Memory API V1 contract tests using synthetic, explicitly sized inventories."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from tllama.native_memory_api import (
    MemoryReportError, NativeMemoryReport, UnsupportedMemorySchema,
)


@pytest.fixture
def payload():
    # Synthetic capacities, not measurements of any model or physical memory.
    return {
        "schema_version": 1,
        "runtime_fingerprint": "experimental-build-with-new-model-support",
        "scope": "backend_buffers", "precision": "exact_bytes", "status": "known",
        "missing_components": [], "model_state": "loaded",
        "snapshot_sequence": 17, "snapshot_stage": "after_request", "snapshot_age_ms": 23,
        "owners": [
            {"id": "model", "kind": "model", "roles": ["main", "mtp"]},
            {"id": "context", "kind": "context", "roles": ["main"]},
            {"id": "mtp", "kind": "context", "roles": ["mtp"]},
            {"id": "projector", "kind": "projector", "roles": ["vision"]},
        ],
        "devices": [
            {"id": "cpu", "backend": "CPU", "name": None, "type": "cpu"},
            {"id": "gpu0", "backend": "Vulkan", "name": "Same name", "type": "igpu"},
            {"id": "gpu1", "backend": "Vulkan", "name": "Same name", "type": "gpu"},
        ],
        "buffers": [
            {"id": "weights", "owners": [{"owner_id": "model", "component": "model_weights"}],
             "device_ids": ["gpu0"], "buffer_type": "GPU-sysmem", "memory_kind": "device",
             "backing": "allocated", "size_bytes": 1000},
            {"id": "shared", "owners": [{"owner_id": "context", "component": "compute"},
                                          {"owner_id": "mtp", "component": "compute"}],
             "device_ids": ["gpu0", "gpu1"], "buffer_type": None, "memory_kind": "device",
             "backing": "allocated", "size_bytes": 100},
            {"id": "input", "owners": [{"owner_id": "model", "component": "model_weights"}],
             "device_ids": ["cpu"], "buffer_type": "CPU_Mapped", "memory_kind": "host",
             "backing": "mmap", "size_bytes": 30},
            {"id": "staging", "owners": [{"owner_id": "context", "component": "output"}],
             "device_ids": ["gpu0"], "buffer_type": "Vulkan_Host", "memory_kind": "host_pinned",
             "backing": "allocated", "size_bytes": 20},
            {"id": "vision", "owners": [{"owner_id": "projector", "component": "projector_weights"}],
             "device_ids": ["gpu0"], "buffer_type": "Vulkan0", "memory_kind": "device",
             "backing": "allocated", "size_bytes": 200},
        ],
        "placement": {"status": "known", "missing_components": [], "layer_groups": [
            {"owner_id": "model", "component": "input_weights", "layer_indices": [],
             "location": "cpu", "device_ids": ["cpu"]},
            {"owner_id": "model", "component": "repeating_layers", "layer_indices": [0, 1],
             "location": "accelerator", "device_ids": ["gpu0"]},
            {"owner_id": "model", "component": "mtp_layers", "layer_indices": [0],
             "location": "mixed", "device_ids": ["cpu", "gpu0"]},
        ]},
    }


def test_capacities_are_counted_once_with_shared_owners_and_devices(payload):
    report = NativeMemoryReport(payload)
    totals = report.totals()
    assert totals.allocated_bytes == totals.known_bytes == 1350
    assert totals.device_bytes == totals.known_device_bytes == 1300
    assert totals.host_bytes == totals.known_host_bytes == 50
    assert totals.unclassified_bytes == totals.unsized_buffers == 0
    # Neither iGPU sysmem nor pinned host memory is reclassified by device type/name.
    assert report.snapshot() == payload
    assert "physical_residency_bytes" not in report.snapshot()


def test_distinct_buffers_are_not_deduplicated_by_name_or_size(payload):
    extra = deepcopy(payload["buffers"][0])
    extra["id"] = "other-weights"
    payload["buffers"].append(extra)
    assert NativeMemoryReport(payload).totals().allocated_bytes == 2350


def test_input_and_output_mutation_cannot_change_a_saved_snapshot(payload):
    expected = deepcopy(payload)
    report = NativeMemoryReport(payload)
    payload["buffers"][0]["size_bytes"] = 9999
    payload["owners"][0]["roles"].clear()
    returned = report.snapshot()
    returned["placement"]["layer_groups"][0]["device_ids"].clear()
    returned["buffers"].clear()
    assert report.snapshot() == expected
    with pytest.raises(FrozenInstanceError):
        report.totals().known_bytes = 0


def test_snapshot_ids_are_never_joined_across_requests(payload):
    first = NativeMemoryReport(payload)
    payload["snapshot_sequence"] += 1
    payload["buffers"][0]["size_bytes"] = 2000
    second = NativeMemoryReport(payload)
    assert first.totals().allocated_bytes == 1350
    assert second.totals().allocated_bytes == 2350
    assert second.snapshot()["snapshot_age_ms"] == 23


def test_partial_inventory_exposes_subtotals_without_zero_filling(payload):
    payload["status"] = "partial"
    payload["buffers"][0]["size_bytes"] = None
    payload["missing_components"] = [
        {"owner_id": "model", "component": "model_weights", "reason": "opaque_composite"}]
    report = NativeMemoryReport(payload)
    totals = report.totals()
    assert totals.allocated_bytes is totals.host_bytes is totals.device_bytes is None
    assert totals.known_bytes == 350 and totals.unsized_buffers == 1
    assert report.snapshot()["buffers"][0]["size_bytes"] is None


def test_missing_unenumerated_component_also_prevents_complete_total(payload):
    payload["status"] = "partial"
    payload["missing_components"] = [
        {"owner_id": None, "component": "future_component", "reason": "collector_not_implemented"}]
    totals = NativeMemoryReport(payload).totals()
    assert totals.known_bytes == 1350 and totals.unsized_buffers == 0
    assert totals.allocated_bytes is None


def test_unknown_memory_kind_keeps_total_but_not_an_invented_split(payload):
    payload["buffers"][0]["memory_kind"] = "unknown"
    totals = NativeMemoryReport(payload).totals()
    assert totals.allocated_bytes == 1350
    assert totals.host_bytes is totals.device_bytes is None
    assert (totals.known_host_bytes, totals.known_device_bytes, totals.unclassified_bytes) == (50, 300, 1000)


@pytest.mark.parametrize("state", ["sleeping", "unloaded"])
def test_known_empty_inventory_after_unload_is_zero(payload, state):
    payload.update(model_state=state, snapshot_stage="after_unload", owners=[], devices=[], buffers=[])
    payload["placement"]["layer_groups"] = []
    totals = NativeMemoryReport(payload).totals()
    assert totals.allocated_bytes == totals.host_bytes == totals.device_bytes == 0


def test_unavailable_inventory_does_not_masquerade_as_known_empty(payload):
    payload.update(status="unavailable", buffers=[])
    payload["missing_components"] = [
        {"owner_id": None, "component": "inventory", "reason": "collection_failed"}]
    totals = NativeMemoryReport(payload).totals()
    assert totals.known_bytes == 0 and totals.allocated_bytes is None


def test_inventory_and_placement_coverage_are_independent(payload):
    payload["placement"]["status"] = "partial"
    payload["placement"]["layer_groups"][1]["location"] = "unknown"
    payload["placement"]["missing_components"] = [
        {"owner_id": "model", "component": "repeating_layers", "reason": "opaque_distribution"}]
    report = NativeMemoryReport(payload)
    assert report.totals().allocated_bytes == 1350
    assert report.snapshot()["placement"]["status"] == "partial"


def test_embeddings_without_kv_and_loaded_inactive_adapters_are_legitimate(payload):
    payload["owners"].append({"id": "lora", "kind": "adapter", "roles": ["lora"]})
    extra = deepcopy(payload["buffers"][0])
    extra.update(id="adapter", owners=[{"owner_id": "lora", "component": "adapter_weights"}], size_bytes=42)
    payload["buffers"].append(extra)
    assert all(rel["component"] != "kv_cache" for b in payload["buffers"] for rel in b["owners"])
    assert NativeMemoryReport(payload).totals().allocated_bytes == 1392


def test_v1_additions_are_ignored_and_new_role_component_reason_names_survive(payload):
    expected = deepcopy(payload)
    payload["future_top"] = {"anything": True}
    payload["devices"][0]["future_device"] = 7
    payload["buffers"][0]["owners"][0]["future_relation"] = []
    payload["placement"]["future_placement"] = None
    assert NativeMemoryReport(payload).snapshot() == expected
    payload["owners"][0]["roles"].append("future_role")
    payload["buffers"][0]["owners"][0]["component"] = "future_component"
    assert NativeMemoryReport(payload).snapshot()["owners"][0]["roles"][-1] == "future_role"


def assign(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    key = int(parts[-1]) if isinstance(obj, list) else parts[-1]
    obj[key] = value


@pytest.mark.parametrize("path,value", [
    ("schema_version", True), ("schema_version", 1.0), ("schema_version", "1"),
    ("snapshot_sequence", 0), ("snapshot_sequence", -1), ("snapshot_sequence", True),
    ("snapshot_age_ms", -1), ("snapshot_age_ms", 0.5),
    ("buffers.0.size_bytes", True), ("buffers.0.size_bytes", -1),
    ("buffers.0.size_bytes", 1.0), ("buffers.0.size_bytes", "1000"),
    ("buffers.0.size_bytes", None), ("buffers.0.memory_kind", "vram"),
    ("buffers.0.backing", "imported"), ("buffers.0.device_ids", ["absent"]),
    ("buffers.0.device_ids", ["gpu0", "gpu0"]), ("buffers.0.id", ""),
    ("buffers.0.owners.0.owner_id", "absent"), ("buffers.0.owners.0.component", ""),
    ("devices.0.type", "apu"), ("devices.0.name", 1), ("devices.1.id", "cpu"),
    ("owners.1.id", "model"), ("owners.0.roles", ["main", "main"]),
    ("owners.0.kind", "weights"), ("buffers.1.id", "weights"),
    ("placement.layer_groups.0.owner_id", "absent"),
    ("placement.layer_groups.0.device_ids", ["absent"]),
    ("placement.layer_groups.0.layer_indices", [0]),
    ("placement.layer_groups.1.layer_indices", [0, 0]),
    ("placement.layer_groups.1.layer_indices", [True]),
    ("placement.layer_groups.1.location", "unknown"),
    ("scope", "physical_residency"), ("precision", "rounded_mib"),
    ("status", "complete"), ("model_state", "ready"), ("snapshot_stage", "after_token"),
    ("runtime_fingerprint", ""), ("owners", {}), ("placement", None),
])
def test_invalid_or_contradictory_reports_fail_explicitly(payload, path, value):
    assign(payload, path, value)
    with pytest.raises(MemoryReportError):
        NativeMemoryReport(payload)


@pytest.mark.parametrize("version", [0, 2, 1000])
def test_unsupported_schema_is_distinguishable_without_parsing_v1_fields(version):
    with pytest.raises(UnsupportedMemorySchema):
        NativeMemoryReport({"schema_version": version})


@pytest.mark.parametrize("path", ["scope", "buffers.0.size_bytes", "placement.status", "devices.0.backend"])
def test_missing_required_fields_are_not_silently_defaulted(payload, path):
    parts = path.split(".")
    obj = payload
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    del obj[parts[-1]]
    with pytest.raises(MemoryReportError, match="missing field"):
        NativeMemoryReport(payload)


@pytest.mark.parametrize("section", ["inventory", "placement"])
def test_known_coverage_cannot_hide_missing_components(payload, section):
    target = payload if section == "inventory" else payload["placement"]
    target["missing_components"] = [{"owner_id": None, "component": "compute", "reason": "size_unavailable"}]
    with pytest.raises(MemoryReportError, match="known coverage"):
        NativeMemoryReport(payload)


def test_missing_component_owner_must_resolve(payload):
    payload["status"] = "partial"
    payload["missing_components"] = [{"owner_id": "absent", "component": "compute", "reason": "size_unavailable"}]
    with pytest.raises(MemoryReportError, match="unresolved owner"):
        NativeMemoryReport(payload)


def test_duplicate_relationships_are_rejected_instead_of_recounted(payload):
    payload["buffers"][0]["owners"] *= 2
    with pytest.raises(MemoryReportError, match="duplicate"):
        NativeMemoryReport(payload)


def test_one_layer_cannot_have_conflicting_placement_groups(payload):
    group = deepcopy(payload["placement"]["layer_groups"][1])
    group.update(location="cpu", device_ids=["cpu"])
    payload["placement"]["layer_groups"].append(group)
    with pytest.raises(MemoryReportError, match="overlapping"):
        NativeMemoryReport(payload)


def test_zero_bytes_and_large_exact_integer_sizes_are_preserved(payload):
    payload["buffers"][0]["size_bytes"] = 2**63 + 123
    payload["buffers"][1]["size_bytes"] = 0
    assert NativeMemoryReport(payload).totals().allocated_bytes == 2**63 + 373


@pytest.mark.parametrize("payload", [None, [], "not JSON object", 1])
def test_nonobject_responses_fail_with_adapter_error(payload):
    with pytest.raises(MemoryReportError):
        NativeMemoryReport(payload)
