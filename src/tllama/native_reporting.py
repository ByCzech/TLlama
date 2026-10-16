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
    # The pinned runtime always keeps input tensors on the host. Once every
    # layer is offloaded, host model buffers cannot describe CPU layers.
    # Output buffers hold results; *_Host compute buffers are device staging.
    # Preserve CPU compute, KV/recurrent state, and projector allocations.
    excluded = sum(kinds.get('model', 0) + kinds.get('output', 0)
                   + (kinds.get('compute', 0) if backend.upper().endswith('_HOST') else 0)
                   for backend, kinds in buffers.items() if buffer_location(backend) == 'host')
    return total - excluded, device


def _native_memory_sizes(memory):
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
    host_input = {owner for owner, values in groups.items()
                  if any(group['component'] == 'input_weights' and group['location'] in {'cpu', 'mixed'}
                         for group in values)}
    contexts = {owner['id'] for owner in memory['owners'] if owner['kind'] == 'context'}

    def helper(rel, kind):
        component, owner = rel['component'], rel['owner_id']
        return ((component == 'model_weights' and owner in host_input)
                or (owner in contexts and (component == 'output'
                                          or component == 'compute' and kind == 'host_pinned')))

    # Every relationship must be removable: a shared leaf must not hide CPU
    # state, projector/adapter weights or other host computation through one
    # otherwise removable owner. Provider names and requested layers are unused.
    excluded = sum(buffer['size_bytes'] for buffer in memory['buffers']
                   if buffer['memory_kind'] in {'host', 'host_pinned'} and buffer['owners']
                   and all(helper(rel, buffer['memory_kind']) for rel in buffer['owners']))
    return total - excluded, device, 'full_accelerator_weights'


def ollama_memory_sizes(memory):
    """Select the display policy for one complete, unmixed measurement source."""
    if memory['source'] == 'native_memory_api':
        return _native_memory_sizes(memory)[:2]
    if memory['source'] == 'native_load_log':
        total, device = _log_memory_sizes(memory)
        return (total, device) if total is not None and device is not None else (None, None)
    return None, None


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


def ollama_processes(generations):
    models = []
    for entry in generations:
        metadata, memory = entry['metadata'], public_memory(entry['memory'])
        if memory['source'] == 'native_memory_api':
            size, size_vram, reason = _native_memory_sizes(memory)
            basis = 'native_weight_placement'
        else:
            size, size_vram = ollama_memory_sizes(memory)
            reason = memory['reason']
            basis = None
            if memory['source'] == 'native_load_log':
                basis = ('verified_log_offload' if size is not None and size != memory['allocated_bytes']
                         else 'load_buffer_totals')
        total = memory['allocated_bytes']
        memory['ollama_compatibility'] = {
            'size':size, 'size_vram':size_vram, 'basis':basis, 'reason':reason,
            'excluded_host_bytes':total - size if total is not None and size is not None else None,
        }
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
            # Native ownership and metric provenance are TLlama extensions.
            'generation':entry['generation'], 'state':entry['state'], 'pid':entry['pid'],
            'leases':entry['leases'], 'retirement_reason':entry['retirement_reason'], 'memory':memory,
        })
    return {'models':models}
