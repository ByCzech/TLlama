"""Validate memory API V1 snapshots without HTTP, worker or public API policy.

Buffer sizes are capacities/spans, including mmap. Optional weight_storage
describes occupied weight bytes separately. Neither measures physical RAM/VRAM
residency. IDs are local to one snapshot. No data is joined across snapshots.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re


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


def _boolean(value, path):
    if type(value) is not bool:
        raise MemoryReportError(f"{path}: expected a boolean")
    return value


def _native_properties(value, path):
    obj = _object(value, path)
    names = {"vulkan.memory_type_index", "vulkan.heap_index",
             "vulkan.memory_property_flags", "vulkan.heap_flags", "level_zero.memory_ordinal"}
    integrated = {"cuda.device_integrated", "hip.device_integrated",
                  "level_zero.device_integrated"}
    # Never expose unrecognized native handles/tokens. Zero is known evidence.
    result = {}
    for key in obj:
        if key in names or key in integrated:
            item_path = f"{path}.{key}"
            number = _unsigned(obj[key], item_path)
            if key in integrated and number > 1:
                raise MemoryReportError(f"{item_path}: expected unsigned 0 or 1")
            result[key] = number
    return result


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


_weight_relationship = _record(owner_id=_string, component=_string,
                               layer_index=_nullable_unsigned)
_weight_missing = _array(_record(owner_id=_nullable_string, component=_string,
                                layer_index=_nullable_unsigned, reason=_string))
_weight_interval = _record(id=_string, buffer_id=_string, offset_bytes=_unsigned,
                          size_bytes=_unsigned, relationships=_array(_weight_relationship))
_weight_storage = _record(scope=_enum("model_weights"), deduplication=_enum("buffer_ranges"),
                         status=_status, missing_components=_weight_missing,
                         intervals=_array(_weight_interval))


def _parse_weight_storage(value, path, data, devices):
    result = _weight_storage(value, path)
    owners = {owner["id"]: owner for owner in data["owners"]}
    buffers = {buffer["id"]: buffer for buffer in data["buffers"]}
    _coverage(result, path, owners)
    missing = result["missing_components"]
    _unique([tuple(item.values()) for item in missing], f"{path}.missing_components")
    if any(item["owner_id"] is not None and owners[item["owner_id"]]["kind"] != "model"
           for item in missing):
        raise MemoryReportError(f"{path}.missing_components: owner outside model_weights scope")
    intervals = result["intervals"]
    _unique([interval["id"] for interval in intervals], f"{path}.intervals")
    if result["status"] == "unavailable" and (intervals or not missing):
        raise MemoryReportError(f"{path}: unavailable storage needs a reason and no intervals")
    ends = {}
    for index, (interval, raw) in enumerate(zip(intervals, value["intervals"])):
        item_path = f"{path}.intervals[{index}]"
        buffer = buffers.get(interval["buffer_id"])
        if buffer is None:
            raise MemoryReportError(f"{item_path}: unresolved buffer")
        start, size = interval["offset_bytes"], interval["size_bytes"]
        end = _unsigned(start + size, f"{item_path}.end_bytes")
        if size == 0 or (buffer["size_bytes"] is not None and end > buffer["size_bytes"]):
            raise MemoryReportError(f"{item_path}: empty range or range exceeds buffer span")
        if start < ends.get(interval["buffer_id"], 0):
            raise MemoryReportError(f"{item_path}: unordered or overlapping buffer ranges")
        ends[interval["buffer_id"]] = end
        relationships = interval["relationships"]
        if not relationships:
            raise MemoryReportError(f"{item_path}: missing relationships")
        _unique([tuple(rel.values()) for rel in relationships], f"{item_path}.relationships")
        for rel in relationships:
            owner = owners.get(rel["owner_id"])
            if owner is None or owner["kind"] != "model":
                raise MemoryReportError(f"{item_path}.relationships: unresolved model owner")
            if not any(r["owner_id"] == rel["owner_id"] and r["component"] == "model_weights"
                       for r in buffer["owners"]):
                raise MemoryReportError(f"{item_path}.relationships: owner does not own weight buffer")
            component, layer = rel["component"], rel["layer_index"]
            if (component in {"repeating_layers", "mtp_layers"} and layer is None
                    or component in {"input_weights", "output_weights", "other_weights"} and layer is not None):
                raise MemoryReportError(f"{item_path}.relationships: inconsistent native layer index")
        interval["assignment"] = _field(raw, "assignment", item_path,
            lambda v, p: _parse_assignment(v, p, devices))
        if result["status"] == "known" and interval["assignment"]["status"] != "known":
            raise MemoryReportError(f"{item_path}: known coverage has incomplete assignment")
    return result


_tensor_range = _record(offset_bytes=_unsigned, size_bytes=_unsigned)
_tensor_record = _record(id=_string, tensor_name=_nullable_string, owner_id=_string,
    component=_nullable_string, layer_index=_nullable_unsigned, buffer_id=_nullable_string,
    storage_status=_enum("known", "unavailable"), storage_reason=_nullable_string,
    size_bytes=_nullable_unsigned, ranges_complete=_boolean, ranges=_array(_tensor_range))
_tensor_diagnostics = _record(status=_status, record_limit=_unsigned, range_limit=_unsigned,
    truncated=_boolean, name_encoding=_enum("byte_escape"), missing_components=_weight_missing,
    records=_array(_tensor_record))
_escaped_name = re.compile(r'(?:[\x20-\x5b\x5d-\x7e]|\\\\|\\x[0-9a-fA-F]{2})*')


def _parse_tensor_diagnostics(value, path, data, devices):
    obj = _object(value, path)
    # Bound enumeration before copying the arrays, including the global range budget.
    limit = _field(obj, "record_limit", path, _unsigned)
    range_limit = _field(obj, "range_limit", path, _unsigned)
    if not 1 <= limit <= 4096 or not 1 <= range_limit <= 16384:
        raise MemoryReportError(f"{path}: invalid diagnostic limits")
    raw_records = obj.get("records")
    if not isinstance(raw_records, list) or len(raw_records) > limit:
        raise MemoryReportError(f"{path}: diagnostic record limit exceeded")
    range_count = 0
    for raw in raw_records:
        ranges = _object(raw, path).get("ranges")
        if not isinstance(ranges, list):
            raise MemoryReportError(f"{path}: expected diagnostic ranges")
        range_count += len(ranges)
        if range_count > range_limit:
            raise MemoryReportError(f"{path}: diagnostic range limit exceeded")
    result = _tensor_diagnostics(obj, path)
    owners = {o["id"]: o for o in data["owners"]}
    buffers = {b["id"]: b for b in data["buffers"]}
    _coverage(result, path, owners)
    missing = result["missing_components"]
    _unique([tuple(item.values()) for item in missing], f"{path}.missing_components")
    if any(item["owner_id"] is not None and owners[item["owner_id"]]["kind"] != "model" for item in missing):
        raise MemoryReportError(f"{path}: diagnostic missing owner outside model scope")
    reasons = {item["reason"] for item in missing}
    limited = bool(reasons & {"diagnostic_record_limit", "diagnostic_range_limit"})
    if result["truncated"] != limited or result["status"] == "known" and limited:
        raise MemoryReportError(f"{path}: inconsistent truncation evidence")
    if result["status"] == "unavailable" and (result["records"] or not missing):
        raise MemoryReportError(f"{path}: unavailable diagnostics need a reason and no records")
    _unique([record["id"] for record in result["records"]], f"{path}.records")
    for index, (record, raw) in enumerate(zip(result["records"], raw_records)):
        item_path = f"{path}.records[{index}]"
        owner = owners.get(record["owner_id"])
        if owner is None or owner["kind"] != "model":
            raise MemoryReportError(f"{item_path}: unresolved model owner")
        name = record["tensor_name"]
        if name is not None and _escaped_name.fullmatch(name) is None:
            raise MemoryReportError(f"{item_path}: invalid byte_escape name")
        component, layer = record["component"], record["layer_index"]
        if (component in {"repeating_layers", "mtp_layers"} and layer is None
                or component in {"input_weights", "output_weights", "other_weights"} and layer is not None):
            raise MemoryReportError(f"{item_path}: inconsistent native layer index")
        buffer = buffers.get(record["buffer_id"])
        if record["buffer_id"] is not None and buffer is None:
            raise MemoryReportError(f"{item_path}: unresolved buffer")
        if buffer is not None and {"owner_id": record["owner_id"], "component": "model_weights"} not in buffer["owners"]:
            raise MemoryReportError(f"{item_path}: owner does not own weight buffer")
        record["assignment"] = _field(raw, "assignment", item_path,
            lambda v, p: _parse_assignment(v, p, devices))
        if record["storage_status"] == "unavailable":
            if (not record["storage_reason"] or record["size_bytes"] is not None
                    or record["ranges"] or record["ranges_complete"]):
                raise MemoryReportError(f"{item_path}: inconsistent unavailable storage")
        else:
            if buffer is None or record["size_bytes"] is None or record["storage_reason"] is not None:
                raise MemoryReportError(f"{item_path}: known storage lacks geometry")
            if buffer["size_bytes"] is not None and record["size_bytes"] > buffer["size_bytes"]:
                raise MemoryReportError(f"{item_path}: diagnostic payload exceeds buffer span")
            end = total = 0
            for span in record["ranges"]:
                start, size = span["offset_bytes"], span["size_bytes"]
                next_end = _unsigned(start + size, item_path)
                if (not size or start < end
                        or buffer["size_bytes"] is not None and next_end > buffer["size_bytes"]):
                    raise MemoryReportError(f"{item_path}: invalid diagnostic range")
                end = next_end
                total = _unsigned(total + size, item_path)
            if (total > record["size_bytes"] or record["ranges_complete"] and total != record["size_bytes"]
                    or not record["ranges_complete"] and "diagnostic_range_limit" not in reasons):
                raise MemoryReportError(f"{item_path}: inconsistent diagnostic payload size")
        if result["status"] == "known" and (record["storage_status"] != "known"
                or record["assignment"]["status"] != "known" or component is None or not record["ranges_complete"]):
            raise MemoryReportError(f"{item_path}: known diagnostics lack complete evidence")
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
    if "weight_storage" in obj:
        data["weight_storage"] = parse(obj["weight_storage"], "report.weight_storage",
            lambda v, p: _parse_weight_storage(v, p, data, devices),
            {"status": "unavailable", "scope": "model_weights", "deduplication": "buffer_ranges",
             "missing_components": [{"owner_id": None, "component": "model_weights",
                                     "layer_index": None, "reason": "invalid_metadata"}],
             "intervals": []})
        raw_storage = obj["weight_storage"]
        if isinstance(raw_storage, dict) and "tensor_diagnostics" in raw_storage:
            path = "report.weight_storage.tensor_diagnostics"
            data["weight_storage"]["tensor_diagnostics"] = parse(raw_storage["tensor_diagnostics"], path,
                lambda v, p: _parse_tensor_diagnostics(v, p, data, devices),
                {"status": "unavailable", "record_limit": None, "range_limit": None,
                 "truncated": False, "name_encoding": "byte_escape",
                 "missing_components": [{"owner_id": None, "component": "model_weights",
                                         "layer_index": None, "reason": "invalid_metadata"}], "records": []})
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


@dataclass(frozen=True)
class WeightStorageTotals:
    """Unique occupied weight bytes, never additive with buffer/allocation totals.

    Complete totals require known coverage of the entire native weight scope,
    even for a filtered selection. Known subtotals preserve partial geometry.
    CPU/accelerator means assignment, not buffer kind or physical residency.
    An interval spanning CPU and GPU is unclassified, never split by a guess.
    """

    status: str
    payload_bytes: int | None
    cpu_bytes: int | None
    accelerator_bytes: int | None
    known_bytes: int
    known_cpu_bytes: int
    known_accelerator_bytes: int
    unclassified_bytes: int
    interval_count: int


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

    def weight_totals(self, *, owner_ids=None, roles=None, components=None) -> WeightStorageTotals | None:
        """Select relationships, then count each matching interval exactly once.

        Selectors are optional iterables of exact IDs/names; empty selects none.
        Role names are opaque, including future roles. Components are matched
        on the same relationship as the selected owner. Owner/role subtotals
        can overlap and must not be added. Absent extension returns None.
        """
        storage = self._data.get("weight_storage")
        if storage is None:
            return None
        owner_ids = None if owner_ids is None else set(owner_ids)
        owners = {owner["id"] for owner in self._data["owners"] if owner["kind"] == "model"
                  and (owner_ids is None or owner["id"] in owner_ids)}
        if roles is not None:
            roles = set(roles)
            owners &= {owner["id"] for owner in self._data["owners"] if roles.intersection(owner["roles"])}
        components = None if components is None else set(components)
        device_types = {device["id"]: device["type"] for device in self._data["devices"]}
        cpu = accelerator = unclassified = count = 0
        for interval in storage["intervals"]:
            if not any(rel["owner_id"] in owners
                       and (components is None or rel["component"] in components)
                       for rel in interval["relationships"]):
                continue
            count += 1
            size, assignment = interval["size_bytes"], interval["assignment"]
            types = {device_types[device] for device in assignment["device_ids"]}
            if assignment["status"] != "known" or not types:
                unclassified += size
            elif types == {"cpu"}:
                cpu += size
            elif types <= {"gpu", "igpu", "accel"}:
                accelerator += size
            else:
                unclassified += size
        known = cpu + accelerator + unclassified
        complete = storage["status"] == "known"
        classified = complete and unclassified == 0
        return WeightStorageTotals(storage["status"], known if complete else None,
                                   cpu if classified else None, accelerator if classified else None,
                                   known, cpu, accelerator, unclassified, count)
