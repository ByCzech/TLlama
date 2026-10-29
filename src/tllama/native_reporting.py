"""Ollama process reporting from immutable native generation snapshots."""

from copy import deepcopy

from tllama.helpers.runner import RUNNER
from tllama.native_memory import buffer_location


def _log_memory_sizes(memory):
    """Keep ps processor placement separate from the complete buffer footprint."""
    total, device = memory['allocated_bytes'], memory['device_bytes']
    offload = memory.get('layer_offload')
    if (memory.get('runtime_verified') is False or memory['status'] != 'known'
            or total is None or device is None or not offload
            or offload['offloaded_layers'] != offload['total_layers']):
        return total, device
    buffers = memory['buffers']
    if not any(kinds.get('model', 0) > 0 and buffer_location(backend) == 'device'
               for backend, kinds in buffers.items()):
        return total, device
    # Ollama reports full model-layer offload as 100% GPU, including when
    # projector compute, state or staging still allocates host buffers.
    # Keep those real allocations in the independent detailed inventory.
    return device, device


def _legacy_native_memory_sizes(memory):
    """Return display sizes and their basis, without changing buffer totals.

    This is an Ollama compatibility summary of storage/placement, not an
    execution percentage or a measurement of physically resident RAM/VRAM.
    """
    total, device = memory['allocated_bytes'], memory['device_bytes']
    if memory['status'] != 'known' or total is None:
        return None, None, 'incomplete_inventory'
    if device is None:
        return None, None, 'unknown_memory_kind'
    if not memory['buffers'] and total == 0:
        return 0, 0, 'empty_inventory'
    placement = memory['placement']
    if placement['status'] != 'known':
        return None, None, 'incomplete_placement'
    models = {owner['id'] for owner in memory['owners'] if owner['kind'] == 'model'}
    groups = {owner: [] for owner in models}
    components = {'input_weights', 'output_weights', 'repeating_layers', 'mtp_layers', 'other_weights'}
    for group in placement['layer_groups']:
        if group['owner_id'] not in groups or group['component'] not in components:
            return None, None, 'unsupported_placement'
        if group['location'] == 'unknown':
            return None, None, 'incomplete_placement'
        assignment = group.get('assignment')
        if assignment is not None:
            if assignment['status'] != 'known':
                return None, None, 'incomplete_assignment'
            types = {d['type'] for d in memory['devices'] if d['id'] in assignment['device_ids']}
            if types <= {'cpu', 'gpu', 'igpu', 'accel'}:
                location = ('cpu' if types == {'cpu'} else 'mixed' if 'cpu' in types else 'accelerator')
                if location != group['location']:
                    return None, None, 'inconsistent_assignment'
        groups[group['owner_id']].append(group)
    if not groups or any(not values for values in groups.values()):
        return None, None, 'incomplete_placement'
    noninput = {owner: [group for group in values if group['component'] != 'input_weights']
                for owner, values in groups.items()}
    full = all(values and all(group['location'] == 'accelerator' for group in values)
               for values in noninput.values())
    if not full:
        return total, device, 'buffer_totals'
    if not any(buffer['memory_kind'] == 'device' and buffer['size_bytes'] > 0
               and any(rel['owner_id'] in models and rel['component'] == 'model_weights'
                       for rel in buffer['owners']) for buffer in memory['buffers']):
        return None, None, 'inconsistent_placement'
    # Match Ollama's full-offload summary, not an allocation ratio. The full
    # inventory (including CPU projector compute and shared buffers) is kept
    # verbatim; policy_excluded_bytes explains the compatibility-only difference.
    return device, device, 'full_accelerator_weights'


def _weight_memory_summary(memory):
    """Replace pure weight-buffer capacities with their unique used ranges.

    Weight CPU/GPU bytes follow checked assignment. Other buffers keep their
    existing memory-kind accounting. A shared weight/nonweight leaf cannot be
    decomposed from capacity minus payload: that remainder can contain holes.
    """
    result = {'size': None, 'size_vram': None, 'basis': 'native_weight_storage',
              'metric': 'weight_payload_and_nonweight_buffers', 'precision': 'exact_bytes',
              'reason': None, 'weight_capacity_reduction_bytes': None,
              'policy_excluded_bytes': None, 'excluded_input_weight_bytes': None,
              'excluded_nonweight_host_bytes': None}

    def unavailable(reason):
        result['reason'] = reason
        return result

    if memory['status'] != 'known' or memory['allocated_bytes'] is None:
        return unavailable('incomplete_inventory')
    storage = memory['weight_storage']
    if storage['status'] != 'known':
        return unavailable('incomplete_weight_storage')
    models = {owner['id'] for owner in memory['owners'] if owner['kind'] == 'model'}
    devices = {device['id']: device['type'] for device in memory['devices']}
    components = {'input_weights', 'output_weights', 'repeating_layers', 'mtp_layers', 'other_weights'}
    cpu_weights = gpu_weights = 0
    noninput_models = set()
    cpu_noninput = False
    payload_by_buffer = {}
    for interval in storage['intervals']:
        relations = interval['relationships']
        if any(rel['component'] not in components for rel in relations):
            return unavailable('unsupported_weight_component')
        assignment = interval['assignment']
        types = {devices[device] for device in assignment['device_ids']}
        if assignment['status'] != 'known' or not types:
            return unavailable('incomplete_weight_assignment')
        cpu = types == {'cpu'}
        if not cpu and not types <= {'gpu', 'igpu', 'accel'}:
            return unavailable('unclassified_weight_assignment')
        size = interval['size_bytes']
        if cpu:
            cpu_weights += size
        else:
            gpu_weights += size
        noninput = {rel['owner_id'] for rel in relations if rel['component'] != 'input_weights'}
        noninput_models.update(noninput)
        cpu_noninput |= cpu and bool(noninput)
        key = interval['buffer_id']
        payload_by_buffer[key] = payload_by_buffer.get(key, 0) + size
    full = bool(models) and noninput_models == models and not cpu_noninput
    host_other = device_other = reduction = 0
    for buffer in memory['buffers']:
        owners = buffer['owners']
        is_weight = [rel['owner_id'] in models and rel['component'] == 'model_weights' for rel in owners]
        if any(is_weight):
            if not all(is_weight):
                return unavailable('shared_weight_nonweight_buffer')
            reduction += buffer['size_bytes'] - payload_by_buffer.get(buffer['id'], 0)
        elif buffer['memory_kind'] in {'host', 'host_pinned'}:
            host_other += buffer['size_bytes']
        elif buffer['memory_kind'] == 'device':
            device_other += buffer['size_bytes']
        else:
            return unavailable('unknown_nonweight_memory_kind')
    excluded_input = cpu_weights if full else 0
    excluded_other = host_other if full else 0
    device = gpu_weights + device_other
    total = cpu_weights + gpu_weights + host_other + device_other
    empty = not memory['buffers'] and memory['allocated_bytes'] == 0
    result.update(size=total - excluded_input - excluded_other, size_vram=device,
                  reason='empty_inventory' if empty else
                         'full_accelerator_weights' if full else 'weight_payload_and_buffer_totals',
                  weight_capacity_reduction_bytes=reduction,
                  policy_excluded_bytes=excluded_input + excluded_other,
                  excluded_input_weight_bytes=excluded_input, excluded_nonweight_host_bytes=excluded_other)
    return result


def ollama_memory_summary(memory):
    """Return sizes with explicit metric, precision and reconciliation terms.

    Only an absent weight extension selects legacy capacity accounting. An
    incomplete or malformed present extension never falls back to mmap spans.
    """
    basis = None
    if memory['source'] == 'native_memory_api':
        if 'weight_storage' in memory:
            return _weight_memory_summary(memory)
        size, device, reason = _legacy_native_memory_sizes(memory)
        basis = 'native_weight_placement'
    elif memory['source'] == 'native_load_log':
        total, device = _log_memory_sizes(memory)
        size, device = (total, device) if total is not None and device is not None else (None, None)
        reason = memory['reason']
        basis = ('verified_log_offload' if size is not None and size != memory['allocated_bytes']
                 else 'load_buffer_totals')
    else:
        size = device = None
        reason = memory['reason']
    allocated = memory['allocated_bytes']
    return {'size': size, 'size_vram': device, 'basis': basis, 'reason': reason,
            'metric': memory['scope'], 'precision': memory['precision'],
            'weight_capacity_reduction_bytes': 0,
            'policy_excluded_bytes': allocated - size if allocated is not None and size is not None else None}


def ollama_memory_sizes(memory):
    """Select the display policy for one complete, unmixed measurement source."""
    result = ollama_memory_summary(memory)
    return result['size'], result['size_vram']


def public_memory(telemetry):
    """Expose the selected inventory, source diagnostics and independent ages."""
    measurement = telemetry['measurement']
    if measurement is None:
        memory = {'source':None, 'status':'unavailable', 'reason':telemetry['native']['reason'],
                  'runtime_fingerprint':telemetry['runtime_fingerprint'],
                  'scope':None, 'precision':None, 'buffers':None,
                  'allocated_bytes':None, 'device_bytes':None, 'host_bytes':None}
    else:
        memory = deepcopy(measurement['report'])
        if telemetry['source'] == 'native_memory_api':
            memory.update(measurement['totals'])
            if 'weight_storage_totals' in measurement:
                memory['weight_storage_totals'] = deepcopy(measurement['weight_storage_totals'])
            memory.update(source='native_memory_api', reason=None if memory['status'] == 'known'
                          else 'incomplete_native_inventory')
    memory.update(units='bytes', physical_residency_bytes=None,
                  mode=telemetry['mode'], fallback_reason=telemetry['fallback_reason'],
                  native=deepcopy(telemetry['native']))
    elapsed = measurement.get('observation_age_ms') if measurement is not None else None
    memory['observation_age_ms'] = elapsed
    native_age = memory.get('snapshot_age_ms')
    memory['estimated_snapshot_age_ms'] = (native_age + elapsed
                                         if native_age is not None and elapsed is not None else None)
    if 'comparison' in telemetry:
        memory['comparison'] = deepcopy(telemetry['comparison'])
    return memory


def ollama_processes(generations, *, detailed=False):
    models = []
    for entry in generations:
        metadata = entry['metadata']
        if detailed:
            memory = public_memory(entry['memory'])
            compatibility = ollama_memory_summary(memory)
            size, size_vram = compatibility['size'], compatibility['size_vram']
            memory['ollama_compatibility'] = compatibility
        else:
            size, size_vram = entry['memory_sizes']
        family = metadata.get('arch', 'unknown')
        parameter_size = metadata.get('parameter_size')
        if not parameter_size:
            params = metadata.get('params')
            try:
                parameter_size = f'{round(int(params) / 1e9)}b' if int(params) > 0 else 'unknown'
            except (TypeError, ValueError, OverflowError):
                parameter_size = 'unknown'
        models.append({
            'name':entry['model'], 'model':entry['model'], 'runner':RUNNER,
            'size':size, 'size_vram':size_vram,
            'digest':entry['file_info'].get('content_sha256', ''),
            'context_length':entry['context_length'], 'expires_at':entry['expires_at'],
            'details':{'parent_model':'', 'format':'gguf', 'family':family, 'families':[family],
                       'parameter_size':parameter_size, 'quantization_level':metadata.get('bits', 'unknown')},
        })
        if detailed:
            models[-1].update(generation=entry['generation'], state=entry['state'], pid=entry['pid'],
                             leases=entry['leases'], retirement_reason=entry['retirement_reason'], memory=memory)
    return {'models':models}
