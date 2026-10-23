"""Validate memory API V1 snapshots without HTTP, worker or public API policy.

Sizes are backend-buffer capacities/spans, including mmap, not physical RAM/VRAM
residency. IDs are local to one snapshot. No data is joined across snapshots.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


class MemoryReportError(ValueError):
    """The response is not a consistent memory API report."""


class UnsupportedMemorySchema(MemoryReportError):
    """The report uses a schema version this adapter does not understand."""


def _object(value, path):
    if not isinstance(value, dict):
        raise MemoryReportError(f"{path}: expected an object")
    return value


def _field(obj, key, path, validate):
    if key not in obj:
        raise MemoryReportError(f"{path}.{key}: missing field")
    return validate(obj[key], f"{path}.{key}")


def _string(value, path):
    if not isinstance(value, str) or not value:
        raise MemoryReportError(f"{path}: expected a nonempty string")
    return value


def _nullable_string(value, path):
    if value is not None and not isinstance(value, str):
        raise MemoryReportError(f"{path}: expected a string or null")
    return value


def _integer(value, path):
    if type(value) is not int or value < 0:
        raise MemoryReportError(f"{path}: expected a nonnegative integer")
    return value


def _enum(*choices):
    def validate(value, path):
        if not isinstance(value, str) or value not in choices:
            raise MemoryReportError(f"{path}: unsupported value")
        return value
    return validate


def _array(validate):
    def parse(value, path):
        if not isinstance(value, list):
            raise MemoryReportError(f"{path}: expected an array")
        return [validate(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return parse


def _record(**fields):
    def parse(value, path):
        obj = _object(value, path)
        # Added fields are ignored, including at nested levels, within V1.
        return {key: _field(obj, key, path, validate) for key, validate in fields.items()}
    return parse


_status = _enum("known", "partial", "unavailable")
_missing = _array(_record(owner_id=_nullable_string, component=_string, reason=_string))
_relationship = _record(owner_id=_string, component=_string)
_owner = _record(id=_string, kind=_enum("model", "context", "projector", "adapter"),
                 roles=_array(_string))
_device = _record(id=_string, backend=_nullable_string, name=_nullable_string,
                  type=_enum("cpu", "gpu", "igpu", "accel", "meta", "unknown"))
_buffer = _record(id=_string, owners=_array(_relationship), device_ids=_array(_string),
                  buffer_type=_nullable_string,
                  memory_kind=_enum("host", "host_pinned", "device", "unknown"),
                  backing=_enum("allocated", "mmap", "unknown"),
                  size_bytes=lambda value, path: None if value is None else _integer(value, path))
_group = _record(owner_id=_string, component=_string, layer_indices=_array(_integer),
                 location=_enum("cpu", "accelerator", "mixed", "unknown"),
                 device_ids=_array(_string))
_placement = _record(status=_status, missing_components=_missing, layer_groups=_array(_group))
_report = _record(
    schema_version=_integer, runtime_fingerprint=_string,
    scope=_enum("backend_buffers"), precision=_enum("exact_bytes"), status=_status,
    missing_components=_missing,
    model_state=_enum("loading", "loaded", "unloading", "unloaded", "sleeping", "failed"),
    snapshot_sequence=_integer,
    snapshot_stage=_enum("before_load", "after_load", "after_batch", "after_projector",
                         "after_request", "after_reconfigure", "before_unload", "after_unload",
                         "after_load_failure"),
    snapshot_age_ms=_integer, owners=_array(_owner), devices=_array(_device),
    buffers=_array(_buffer), placement=_placement,
)


def _unique(values, path):
    if len(values) != len(set(values)):
        raise MemoryReportError(f"{path}: duplicate entry")
    return set(values)


def _references(values, available, path):
    if not _unique(values, path) <= available:
        raise MemoryReportError(f"{path}: unresolved reference")


def _coverage(section, path, owners):
    for missing in section["missing_components"]:
        if missing["owner_id"] is not None and missing["owner_id"] not in owners:
            raise MemoryReportError(f"{path}.missing_components: unresolved owner")
    if section["status"] == "known" and section["missing_components"]:
        raise MemoryReportError(f"{path}: known coverage has missing components")


def _unsigned(value, path):
    value = _integer(value, path)
    if value > 2**64 - 1:
        raise MemoryReportError(f"{path}: unsigned integer overflow")
    return value


def _nullable_unsigned(value, path):
    return None if value is None else _unsigned(value, path)


def _native_properties(value, path):
    obj = _object(value, path)
    names = {"vulkan.memory_type_index", "vulkan.heap_index",
             "vulkan.memory_property_flags", "vulkan.heap_flags", "level_zero.memory_ordinal"}
    # Never expose unrecognized native handles/tokens. Zero is known evidence.
    return {key: _unsigned(obj[key], f"{path}.{key}") for key in obj if key in names}


_assignment = _record(status=_status, device_ids=_array(_string),
    sources=_array(_enum("loader_selection", "loader_override", "loader_mmap",
                         "loader_lazy", "native_buffer")), reasons=_array(_string))
_allocation = _record(id=_string, device_ids=_array(_string),
    kind=_enum("device", "host", "host_pinned", "managed", "shared", "system", "unknown"),
    origin=_enum("allocated", "imported", "registered", "unknown"), native_api=_nullable_string,
    size_bytes=_nullable_unsigned, locality=_enum("device_local", "non_device_local", "unknown"),
    native_properties=_native_properties)
_binding = _record(buffer_id=_string, allocation_id=_string,
    buffer_offset_bytes=_nullable_unsigned, allocation_offset_bytes=_nullable_unsigned,
    size_bytes=_nullable_unsigned)
_locality = _record(status=_status, scope=_enum("backend_buffer_allocations"),
    physical_aliasing=_enum("unknown"),
    missing_components=_array(_record(buffer_id=_nullable_string, reason=_string)),
    allocations=_array(_allocation), bindings=_array(_binding))


def _parse_assignment(value, path, devices):
    result = _assignment(value, path)
    _references(result["device_ids"], devices, f"{path}.device_ids")
    for key in ("sources", "reasons"):
        _unique(result[key], f"{path}.{key}")
    if result["status"] == "known" and (result["reasons"] or not result["device_ids"]):
        raise MemoryReportError(f"{path}: known assignment lacks complete evidence")
    return result


def _parse_locality(value, path, data, devices):
    result = _locality(value, path)
    buffers = {buffer["id"]: buffer for buffer in data["buffers"]}
    allocations = result["allocations"]
    _unique([allocation["id"] for allocation in allocations], f"{path}.allocations")
    allocations = {allocation["id"]: allocation for allocation in allocations}
    for allocation in allocations.values():
        _references(allocation["device_ids"], devices, f"{path}.allocations.device_ids")
    missing = result["missing_components"]
    _unique([(item["buffer_id"], item["reason"]) for item in missing], f"{path}.missing_components")
    if any(item["buffer_id"] is not None and item["buffer_id"] not in buffers for item in missing):
        raise MemoryReportError(f"{path}.missing_components: unresolved buffer")
    if result["status"] == "known" and missing:
        raise MemoryReportError(f"{path}: known coverage has missing components")
    _unique([tuple(binding.values()) for binding in result["bindings"]], f"{path}.bindings")
    for binding in result["bindings"]:
        if binding["buffer_id"] not in buffers or binding["allocation_id"] not in allocations:
            raise MemoryReportError(f"{path}.bindings: unresolved reference")
        length = binding["size_bytes"]
        for key, span in (("buffer_offset_bytes", buffers[binding["buffer_id"]]["size_bytes"]),
                          ("allocation_offset_bytes", allocations[binding["allocation_id"]]["size_bytes"])):
            offset = binding[key]
            end = offset + length if offset is not None and length is not None else None
            if end is not None and end > 2**64 - 1:
                raise MemoryReportError(f"{path}.bindings: range overflow")
            if span is not None and any(v is not None and v > span for v in (offset, length, end)):
                raise MemoryReportError(f"{path}.bindings: range exceeds span")
    # Coverage of relationships does not imply known sizes, locality or aliases.
    return result


def _extensions(obj, data, devices):
    """Optional metadata failures leave the valid V1 capacity inventory usable.

    Sanitize only the failed section, retain an explicit diagnostic, and never
    silently turn malformed assignment evidence into an older provider guess.
    """
    errors = []

    def parse(value, path, validate, unavailable):
        try:
            return validate(value, path)
        except MemoryReportError as exc:
            errors.append({"path": path, "reason": str(exc)})
            return unavailable

    for index, group in enumerate(obj["placement"]["layer_groups"]):
        if "assignment" in group:
            path = f"report.placement.layer_groups[{index}].assignment"
            data["placement"]["layer_groups"][index]["assignment"] = parse(
                group["assignment"], path, lambda v, p: _parse_assignment(v, p, devices),
                {"status": "unavailable", "device_ids": [], "sources": [], "reasons": ["invalid_metadata"]})
    if "allocation_locality" in obj:
        data["allocation_locality"] = parse(obj["allocation_locality"], "report.allocation_locality",
            lambda v, p: _parse_locality(v, p, data, devices),
            {"status": "unavailable", "scope": "backend_buffer_allocations", "physical_aliasing": "unknown",
             "missing_components": [{"buffer_id": None, "reason": "invalid_metadata"}],
             "allocations": [], "bindings": []})
    if errors:
        data["extension_errors"] = errors


@dataclass(frozen=True)
class MemoryTotals:
    """Inventory totals and explicitly incomplete subtotals, all in bytes.

    Complete totals are null when coverage is incomplete. Unknown memory kinds
    also make the host/device split null, even if every buffer size is known.
    Subtotals never imply completeness and do not allocate multi-device buffers
    among devices.
    """

    allocated_bytes: int | None
    host_bytes: int | None
    device_bytes: int | None
    known_bytes: int
    known_host_bytes: int
    known_device_bytes: int
    unclassified_bytes: int
    unsized_buffers: int


class NativeMemoryReport:
    """Own one validated snapshot; preserve native identity and coverage.

    Build identity is informational here. Compatibility policy, transport,
    receipt time and source selection belong to the worker integration.
    """

    def __init__(self, payload):
        obj = _object(payload, "report")
        version = _field(obj, "schema_version", "report", _integer)
        if version != 1:
            raise UnsupportedMemorySchema(f"Unsupported memory schema version: {version}")
        data = _report(obj, "report")
        if data["snapshot_sequence"] == 0:
            raise MemoryReportError("report.snapshot_sequence: expected a positive integer")
        owners = _unique([owner["id"] for owner in data["owners"]], "report.owners")
        devices = _unique([device["id"] for device in data["devices"]], "report.devices")
        _unique([buffer["id"] for buffer in data["buffers"]], "report.buffers")
        for owner in data["owners"]:
            _unique(owner["roles"], "report.owners.roles")
        _coverage(data, "report", owners)
        for buffer in data["buffers"]:
            _references(buffer["device_ids"], devices, "report.buffers.device_ids")
            relations = [(rel["owner_id"], rel["component"]) for rel in buffer["owners"]]
            _unique(relations, "report.buffers.owners")
            if any(owner not in owners for owner, _ in relations):
                raise MemoryReportError("report.buffers.owners: unresolved owner")
            if data["status"] == "known" and buffer["size_bytes"] is None:
                raise MemoryReportError("report.buffers: known coverage has an unsized buffer")
        placement = data["placement"]
        _coverage(placement, "report.placement", owners)
        layers = set()
        for group in placement["layer_groups"]:
            if group["owner_id"] not in owners:
                raise MemoryReportError("report.placement.layer_groups: unresolved owner")
            _references(group["device_ids"], devices, "report.placement.device_ids")
            _unique(group["layer_indices"], "report.placement.layer_indices")
            if placement["status"] == "known" and group["location"] == "unknown":
                raise MemoryReportError("report.placement: known coverage has unknown placement")
            identity = (group["owner_id"], group["component"])
            if group["component"] in {"input_weights", "output_weights", "other_weights"}:
                if group["layer_indices"]:
                    raise MemoryReportError("report.placement: non-layer group has layer indices")
            for index in group["layer_indices"]:
                key = (*identity, index)
                if key in layers:
                    raise MemoryReportError("report.placement: overlapping layer groups")
                layers.add(key)
        _extensions(obj, data, devices)
        self._data = data

    def snapshot(self) -> dict:
        """Return an independent copy of the understood V1 fields."""
        return deepcopy(self._data)

    def totals(self) -> MemoryTotals:
        """Sum each inventory leaf once, independently of owners and placement."""
        host = device = unclassified = unsized = 0
        for buffer in self._data["buffers"]:
            size = buffer["size_bytes"]
            if size is None:
                unsized += 1
            elif buffer["memory_kind"] in {"host", "host_pinned"}:
                host += size
            elif buffer["memory_kind"] == "device":
                device += size
            else:
                unclassified += size
        known = host + device + unclassified
        complete = self._data["status"] == "known"
        classified = all(buffer["memory_kind"] != "unknown" for buffer in self._data["buffers"])
        return MemoryTotals(known if complete else None,
                            host if complete and classified else None,
                            device if complete and classified else None,
                            known, host, device, unclassified, unsized)
