"""Ollama process reporting from immutable native generation snapshots."""

from tllama.helpers.runner import RUNNER


def ollama_processes(generations):
    models = []
    for entry in generations:
        metadata, memory = entry['metadata'], entry['memory']
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
            'size':memory['allocated_bytes'], 'size_vram':memory['device_bytes'],
            'digest':entry['file_info'].get('content_sha256', ''),
            'context_length':entry['context_length'], 'expires_at':entry['expires_at'],
            'details':{'parent_model':'', 'format':'gguf', 'family':family, 'families':[family],
                       'parameter_size':parameter_size, 'quantization_level':metadata.get('bits', 'unknown')},
            # Native ownership and metric provenance are TLlama extensions.
            'generation':entry['generation'], 'state':entry['state'], 'pid':entry['pid'],
            'leases':entry['leases'], 'retirement_reason':entry['retirement_reason'], 'memory':memory,
        })
    return {'models':models}
