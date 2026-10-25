"""Occupied weight ranges are distinct from capacity, allocation and residency.

Synthetic byte counts follow the pinned native 0e9433bfd3 contract. They are
not a hardware qualification or a capture from the user's P100/MI50.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import random

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import public_memory, ollama_memory_sizes
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload, assign
from test_native_memory_locality import extended


def relationship(component, layer=None, owner="model"):
    return {"owner_id": owner, "component": component, "layer_index": layer}


def interval(name, buffer, offset, size, *relationships, devices=("gpu0",)):
    return {"id": name, "buffer_id": buffer, "offset_bytes": offset, "size_bytes": size,
            "assignment": {"status": "known", "device_ids": list(devices),
                           "sources": ["loader_selection"], "reasons": []},
            "relationships": list(relationships)}


@pytest.fixture
def weights(extended):
    extended["weight_storage"] = {
        "scope": "model_weights", "deduplication": "buffer_ranges", "status": "known",
        "missing_components": [], "intervals": [
            # Encoded block payload bytes (e.g. Q8_0/Q4_K), not dense elements.
            interval("w1", "weights", 64, 34, relationship("repeating_layers", 39)),
            interval("w2", "weights", 512, 144, relationship("repeating_layers", 40),
                     relationship("mtp_layers", 40)),
            # Input [0,8) and output [4,12) share four bytes, not two copies.
            interval("w3", "input", 0, 4, relationship("input_weights"), devices=("cpu",)),
            interval("w4", "input", 4, 4, relationship("input_weights"),
                     relationship("output_weights"), devices=("cpu",)),
            interval("w5", "input", 8, 4, relationship("output_weights"), devices=("cpu",)),
        ],
    }
    return extended


def test_payload_padding_and_tied_roles_are_counted_once(weights):
    report = NativeMemoryReport(weights)
    assert report.snapshot() == weights
    totals = report.weight_totals()
    assert asdict(totals) == {
        "status": "known", "payload_bytes": 190, "cpu_bytes": 12, "accelerator_bytes": 178,
        "known_bytes": 190, "known_cpu_bytes": 12, "known_accelerator_bytes": 178,
        "unclassified_bytes": 0, "interval_count": 5,
    }
    assert report.totals().allocated_bytes == 1350
    assert report.totals().device_bytes == 1300
    assert report.weight_totals(components={"input_weights"}).payload_bytes == 8
    assert report.weight_totals(components={"output_weights"}).payload_bytes == 8
    assert report.weight_totals(components={"input_weights", "output_weights"}).payload_bytes == 12
    assert report.weight_totals(components={"output_weights", "repeating_layers", "mtp_layers"}).payload_bytes == 186
    assert report.weight_totals(components={"repeating_layers"}).payload_bytes == 178
    assert report.weight_totals(components={"mtp_layers"}).payload_bytes == 144
    with pytest.raises(FrozenInstanceError):
        totals.known_bytes = 0


@pytest.mark.parametrize("backing", ["mmap", "allocated"])
def test_two_sparse_megabytes_in_a_gigabyte_span_have_payload_parity(weights, backing):
    mib = 1024**2
    weights["buffers"][0].update(backing=backing, size_bytes=1024*mib)
    weights["weight_storage"]["intervals"] = [
        interval("first", "weights", 0, mib, relationship("repeating_layers", 0)),
        interval("last", "weights", 1023*mib, mib, relationship("output_weights")),
    ]
    report = NativeMemoryReport(weights)
    assert report.weight_totals().payload_bytes == 2*mib
    assert report.totals().allocated_bytes == 1024*mib + 350


def test_provider_memory_kind_and_locality_do_not_override_weight_assignment(weights):
    weights["buffers"][0].update(memory_kind="host_pinned", device_ids=["gpu1"])
    weights["buffers"][2].update(device_ids=["gpu0"], buffer_type="CUDA_Host")
    report = NativeMemoryReport(weights)
    assert (report.weight_totals().cpu_bytes, report.weight_totals().accelerator_bytes) == (12, 178)
    assert report.totals().host_bytes == 1050
    assert report.snapshot()["allocation_locality"] == weights["allocation_locality"]


def test_main_mtp_role_filters_use_union_and_native_layer_indices(weights):
    report = NativeMemoryReport(weights)
    for roles in ({"main"}, {"mtp"}, {"main", "mtp"}):
        assert report.weight_totals(roles=roles).payload_bytes == 190
    assert report.weight_totals(owner_ids=(x for x in ["model"])).payload_bytes == 190
    for selectors in ({"owner_ids": []}, {"roles": []}, {"components": []}, {"roles": ["absent"]}):
        assert report.weight_totals(**selectors).payload_bytes == 0
    assert report.snapshot()["weight_storage"]["intervals"][1]["relationships"][-1]["layer_index"] == 40


def test_independent_draft_buffer_is_not_deduplicated_by_same_shape_or_name(weights):
    weights["owners"].append({"id": "draft", "kind": "model", "roles": ["draft"]})
    copy = deepcopy(weights["buffers"][0])
    copy.update(id="draft-buffer", owners=[{"owner_id": "draft", "component": "model_weights"}])
    weights["buffers"].append(copy)
    weights["weight_storage"]["intervals"].append(
        interval("draft", "draft-buffer", 64, 34, relationship("repeating_layers", 39, "draft"), devices=("cpu",)))
    report = NativeMemoryReport(weights)
    assert report.weight_totals().payload_bytes == 224
    assert report.weight_totals(roles={"main", "mtp"}).payload_bytes == 190
    assert report.weight_totals(roles={"draft"}).payload_bytes == 34
    assert report.weight_totals(roles={"main", "draft"}).cpu_bytes == 46


def test_shared_buffer_relationship_filters_match_owner_and_component_together(weights):
    weights["owners"].append({"id": "draft", "kind": "model", "roles": ["draft"]})
    weights["buffers"][0]["owners"].append({"owner_id": "draft", "component": "model_weights"})
    weights["weight_storage"]["intervals"][0]["relationships"].append(relationship("output_weights", owner="draft"))
    report = NativeMemoryReport(weights)
    assert report.weight_totals().payload_bytes == 190
    assert report.weight_totals(roles={"draft"}).payload_bytes == 34
    assert report.weight_totals(roles={"draft", "main"}).payload_bytes == 190
    assert report.weight_totals(owner_ids={"model"}, components={"output_weights"}).payload_bytes == 8
    assert report.weight_totals(owner_ids={"draft"}, components={"repeating_layers"}).payload_bytes == 0


@pytest.mark.parametrize("status,devices,reasons", [
    ("unavailable", [], ["assignment_conflict"]),
    ("unavailable", [], ["assignment_unknown"]),
    ("partial", ["gpu0"], ["assignment_unknown"]),
])
def test_partial_assignment_keeps_exact_geometry_without_inventing_a_split(weights, status, devices, reasons):
    section = weights["weight_storage"]
    section["status"] = "partial"
    section["missing_components"] = [{"owner_id": "model", "component": "repeating_layers",
                                      "layer_index": 39, "reason": reasons[0]}]
    section["intervals"][0]["assignment"].update(status=status, device_ids=devices, reasons=reasons)
    report = NativeMemoryReport(weights)
    assert report.snapshot()["weight_storage"] == section
    totals = report.weight_totals()
    assert totals.payload_bytes is totals.cpu_bytes is totals.accelerator_bytes is None
    assert (totals.known_bytes, totals.known_cpu_bytes, totals.known_accelerator_bytes, totals.unclassified_bytes) == (190, 12, 144, 34)
    assert report.totals().allocated_bytes == 1350
    # A filter cannot silently erase global coverage gaps.
    assert report.weight_totals(components={"output_weights"}).payload_bytes is None


@pytest.mark.parametrize("device_type,devices,classified", [
    ("igpu", ["gpu0"], True), ("gpu", ["gpu0", "gpu1"], True),
    ("accel", ["gpu0"], True), ("unknown", ["gpu0"], False),
    ("meta", ["gpu0"], False), ("gpu", ["gpu0", "cpu"], False),
])
def test_multi_device_interval_is_counted_once_and_ambiguous_bytes_are_unclassified(weights, device_type, devices, classified):
    weights["devices"][1]["type"] = device_type
    for item in weights["weight_storage"]["intervals"][:2]:
        item["assignment"]["device_ids"] = devices
    totals = NativeMemoryReport(weights).weight_totals()
    assert totals.payload_bytes == 190
    if classified:
        assert (totals.cpu_bytes, totals.accelerator_bytes, totals.unclassified_bytes) == (12, 178, 0)
    else:
        assert totals.cpu_bytes is totals.accelerator_bytes is None
        assert (totals.known_cpu_bytes, totals.unclassified_bytes) == (12, 178)


def test_partial_missing_geometry_and_nullable_owner_layer_are_retained(weights):
    section = weights["weight_storage"]
    section.update(status="partial", missing_components=[
        {"owner_id": None, "component": "future_component", "layer_index": None, "reason": "future_reason"}])
    totals = NativeMemoryReport(weights).weight_totals()
    assert totals.payload_bytes is None and totals.known_bytes == 190


def test_absence_is_unsupported_and_unavailable_is_not_zero(weights):
    older = deepcopy(weights)
    del older["weight_storage"]
    assert NativeMemoryReport(older).weight_totals() is None
    weights["weight_storage"].update(status="unavailable", intervals=[], missing_components=[
        {"owner_id": "model", "component": "model_weights", "layer_index": None, "reason": "not_inspected"}])
    totals = NativeMemoryReport(weights).weight_totals()
    assert totals.payload_bytes is None and totals.known_bytes == 0


@pytest.mark.parametrize("state", ["sleeping", "unloaded"])
def test_proved_unload_is_known_empty(weights, state):
    weights.update(model_state=state, snapshot_stage="after_unload", owners=[], devices=[], buffers=[])
    weights["placement"].update(status="known", missing_components=[], layer_groups=[])
    weights["weight_storage"]["intervals"] = []
    weights["allocation_locality"].update(status="known", missing_components=[], allocations=[], bindings=[])
    totals = NativeMemoryReport(weights).weight_totals()
    assert totals.payload_bytes == totals.cpu_bytes == totals.accelerator_bytes == 0


def test_unknown_buffer_capacity_does_not_erase_proven_ranges(weights):
    weights["status"] = "partial"
    weights["buffers"][0]["size_bytes"] = None
    report = NativeMemoryReport(weights)
    assert report.totals().allocated_bytes is None
    assert report.weight_totals().payload_bytes == 190


@pytest.mark.parametrize("path,value", [
    ("scope", "physical_residency"), ("deduplication", "tensor_names"), ("status", "complete"),
    ("intervals", None), ("intervals.0.id", ""), ("intervals.1.id", "w1"),
    ("intervals.0.buffer_id", "missing"), ("intervals.0.offset_bytes", True),
    ("intervals.0.offset_bytes", -1), ("intervals.0.offset_bytes", 1.5),
    ("intervals.0.offset_bytes", 2**64), ("intervals.0.offset_bytes", 2**64-1),
    ("intervals.0.size_bytes", 0), ("intervals.0.size_bytes", None),
    ("intervals.0.size_bytes", False), ("intervals.0.size_bytes", -1),
    ("intervals.0.size_bytes", "34"), ("intervals.0.size_bytes", 2**64),
    ("intervals.0.size_bytes", 937), ("intervals.1.offset_bytes", 97),
    ("intervals.0.relationships", []), ("intervals.0.relationships.0.owner_id", "missing"),
    ("intervals.0.relationships.0.owner_id", "context"),
    ("intervals.0.relationships.0.component", ""),
    ("intervals.0.relationships.0.layer_index", None),
    ("intervals.0.relationships.0.layer_index", True),
    ("intervals.0.relationships.0.layer_index", -1),
    ("intervals.2.relationships.0.layer_index", 0),
    ("intervals.0.assignment", None), ("intervals.0.assignment.device_ids", ["missing"]),
    ("intervals.0.assignment.device_ids", ["gpu0", "gpu0"]),
    ("intervals.0.assignment.device_ids", []),
    ("intervals.0.assignment.sources", ["future_source"]),
    ("intervals.0.assignment.sources", ["native_buffer", "native_buffer"]),
    ("intervals.0.assignment.status", "partial"), ("intervals.0.assignment.reasons", ["assignment_conflict"]),
    ("missing_components", [{"owner_id": None, "component": "model_weights", "layer_index": None, "reason": "gap"}]),
])
def test_invalid_weight_extension_withdraws_only_weight_evidence(weights, path, value):
    assign(weights["weight_storage"], path, value)
    report = NativeMemoryReport(weights)
    snapshot = report.snapshot()
    assert report.totals().allocated_bytes == 1350
    assert snapshot["placement"] == weights["placement"]
    assert snapshot["allocation_locality"] == weights["allocation_locality"]
    assert snapshot["weight_storage"]["status"] == "unavailable"
    assert snapshot["weight_storage"]["intervals"] == []
    assert snapshot["extension_errors"][-1]["path"] == "report.weight_storage"
    assert report.weight_totals().payload_bytes is None


@pytest.mark.parametrize("path", ["scope", "status", "deduplication", "missing_components", "intervals",
    "intervals.0.id", "intervals.0.buffer_id", "intervals.0.offset_bytes", "intervals.0.size_bytes",
    "intervals.0.relationships", "intervals.0.relationships.0.layer_index", "intervals.0.assignment"])
def test_missing_required_weight_fields_are_not_defaulted(weights, path):
    parts = path.split(".")
    obj = weights["weight_storage"]
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    del obj[parts[-1]]
    assert NativeMemoryReport(weights).snapshot()["weight_storage"]["status"] == "unavailable"


@pytest.mark.parametrize("case", ["missing_owner", "nonmodel_missing", "missing_layer", "missing_duplicate",
                                 "relationship_duplicate", "buffer_owner", "unsorted", "unavailable_intervals",
                                 "unavailable_no_reason"])
def test_semantic_contradictions_are_isolated(weights, case):
    section = weights["weight_storage"]
    if case.startswith("missing") or case == "nonmodel_missing":
        section.update(status="partial", missing_components=[
            {"owner_id": "model", "component": "repeating_layers", "layer_index": 39, "reason": "gap"}])
        if case == "missing_owner":
            section["missing_components"][0]["owner_id"] = "absent"
        elif case == "nonmodel_missing":
            section["missing_components"][0]["owner_id"] = "projector"
        elif case == "missing_layer":
            section["missing_components"][0]["layer_index"] = False
        else:
            section["missing_components"] *= 2
    elif case == "relationship_duplicate":
        section["intervals"][0]["relationships"] *= 2
    elif case == "buffer_owner":
        section["intervals"][0]["buffer_id"] = "shared"
    elif case == "unsorted":
        section["intervals"][:2] = reversed(section["intervals"][:2])
    else:
        section["status"] = "unavailable"
        if case == "unavailable_no_reason":
            section["intervals"] = []
    snapshot = NativeMemoryReport(weights).snapshot()
    assert snapshot["weight_storage"]["status"] == "unavailable"
    assert snapshot["extension_errors"][-1]["path"] == "report.weight_storage"


def test_forward_fields_and_names_survive_without_inferring_semantics(weights):
    weights["owners"][0]["roles"].append("future_role")
    section = weights["weight_storage"]
    item = section["intervals"][0]
    item["relationships"][0]["component"] = "future_component"
    expected = deepcopy(weights)
    section["future"] = True
    section["tensor_diagnostics"] = {"not_parsed_until_diagnostic_patch": True}
    item["private_address"] = "must_not_escape"
    item["relationships"][0]["future"] = 42
    item["assignment"]["future"] = 123
    report = NativeMemoryReport(weights)
    assert report.snapshot() == expected
    assert report.weight_totals().payload_bytes == 190  # geometry needs no role guess
    assert report.weight_totals(roles={"future_role"}).payload_bytes == 190
    assert report.weight_totals(components={"repeating_layers"}).payload_bytes == 144


def test_uint64_boundary_is_exact_without_allocating_declared_storage(weights):
    maximum = 2**64-1
    weights["buffers"][0]["size_bytes"] = maximum
    weights["weight_storage"]["intervals"] = [
        interval("last-byte", "weights", maximum-1, 1, relationship("other_weights"))]
    report = NativeMemoryReport(weights)
    assert report.weight_totals().payload_bytes == 1
    assert report.snapshot()["weight_storage"]["intervals"][0]["offset_bytes"] == maximum-1


def test_randomized_role_selections_match_independent_byte_set_oracle(weights):
    rng = random.Random(114)
    names = ["input_weights", "output_weights", "other_weights"]
    for _ in range(100):
        # A tiny independent byte map models arbitrary overlap. Publish its
        # role-boundary partition as native does; oracle sums selected bytes.
        visits = [(rng.randrange(32), rng.randrange(1, 9), rng.choice(names)) for _ in range(12)]
        byte_roles = {byte: {role for start, size, role in visits if start <= byte < start+size}
                      for byte in range(40)}
        intervals = []
        byte = 0
        while byte < 40:
            roles = byte_roles[byte]
            end = byte+1
            while end < 40 and byte_roles[end] == roles:
                end += 1
            if roles:
                intervals.append(interval(str(byte), "weights", byte, end-byte,
                    *(relationship(role) for role in sorted(roles))))
            byte = end
        weights["weight_storage"]["intervals"] = intervals
        report = NativeMemoryReport(weights)
        for mask in range(8):
            selection = {name for index, name in enumerate(names) if mask & (1 << index)}
            expected = sum(bool(roles & selection) for roles in byte_roles.values())
            assert report.weight_totals(components=selection).payload_bytes == expected


def test_public_cached_detail_keeps_weight_totals_separate_from_ps_sizes(weights):
    telemetry = MemoryTelemetry(observe(), compare=True)
    telemetry.accept(NativeMemoryReport(weights))
    original = telemetry.snapshot()
    public = public_memory(original)
    assert public["weight_storage_totals"]["payload_bytes"] == 190
    assert public["allocated_bytes"] == 1350
    assert public["comparison"]["native"]["weight_storage_totals"]["known_bytes"] == 190
    assert "weight_storage_totals" not in public["comparison"]["log"]["report"]
    # The displayed metric replaces weight spans; detailed capacity is intact.
    assert ollama_memory_sizes(public) == (510, 478)
    weights["weight_storage"]["intervals"][0]["size_bytes"] = 1
    public["weight_storage"]["intervals"].clear()
    public["weight_storage_totals"]["known_bytes"] = 0
    assert original["measurement"]["report"]["weight_storage"]["intervals"][0]["size_bytes"] == 34
    assert telemetry.snapshot()["measurement"]["weight_storage_totals"]["known_bytes"] == 190


def test_independent_extension_errors_do_not_discard_each_others_evidence(weights):
    weights["allocation_locality"]["bindings"][0]["allocation_id"] = "missing"
    report = NativeMemoryReport(weights)
    assert report.snapshot()["allocation_locality"]["status"] == "unavailable"
    assert report.weight_totals().payload_bytes == 190
    weights["weight_storage"]["intervals"][0]["buffer_id"] = "missing"
    report = NativeMemoryReport(weights)
    assert len(report.snapshot()["extension_errors"]) == 2
    assert report.totals().allocated_bytes == 1350


def test_lifecycle_withdraws_previous_weight_evidence_and_never_joins_snapshot_ids(weights):
    telemetry = MemoryTelemetry(observe(), compare=True)
    telemetry.accept(NativeMemoryReport(weights))
    first = telemetry.snapshot()
    weights["snapshot_sequence"] += 1
    weights["weight_storage"]["intervals"][0]["size_bytes"] = 1
    telemetry.accept(NativeMemoryReport(weights))
    assert first["measurement"]["weight_storage_totals"]["payload_bytes"] == 190
    assert telemetry.snapshot()["measurement"]["weight_storage_totals"]["payload_bytes"] == 157
    # Same sequence-independent IDs are only resolved in the new snapshot.
    weights["snapshot_sequence"] += 1
    weights["weight_storage"]["intervals"][0]["buffer_id"] = "previous-buffer"
    telemetry.accept(NativeMemoryReport(weights))
    current = telemetry.snapshot()
    assert current["measurement"]["weight_storage_totals"]["payload_bytes"] is None
    assert current["measurement"]["totals"]["allocated_bytes"] == 1350
    telemetry.fail("request_failed", http_status=503)
    current = telemetry.snapshot()
    assert current["comparison"]["native"] is None
    assert "weight_storage" not in public_memory(current)
    del weights["weight_storage"]  # older V1/replacement has no inherited extension
    weights["snapshot_sequence"] += 1
    telemetry.accept(NativeMemoryReport(weights))
    assert "weight_storage_totals" not in public_memory(telemetry.snapshot())
