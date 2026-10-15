"""Ollama process reporting from immutable native generation snapshots."""

from tllama.helpers.runner import RUNNER
from tllama.native_memory import buffer_location


def ollama_memory_sizes(memory):
    """Keep ps processor placement separate from the complete buffer footprint."""
    total, device = memory['allocated_bytes'], memory['device_bytes']
    offload = memory.get('layer_offload')
    if (memory['status'] != 'known' or total is None or device is None or not offload
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


def ollama_processes(generations):
    models = []
    for entry in generations:
        metadata, memory = entry['metadata'], entry['memory']
        size, size_vram = ollama_memory_sizes(memory)
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
