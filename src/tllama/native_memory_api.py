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
