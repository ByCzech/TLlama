"""Portable public model IDs: weights, optional projector and TOML deltas.

No server configuration, environment, filesystem paths or request options may
enter this module. Component SHA-256 values are computed/cached by the caller.
"""

from hashlib import sha256
import json
import math
import struct
from collections.abc import Mapping

from tllama.helpers.common import build_sampling_kwargs
from tllama.native_config import builtin_runtime_defaults
from tllama.native_params import RUNTIME_TYPES, SAMPLING_TYPES, kv_type_name, MTP_DRAFT_N_MAX_DEFAULT


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _normal(value):
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        # Integer-valued TOML floats and integers have one numeric spelling.
        return int(value) if value == int(value) else float(value)
    return str(value)


def _runtime_defaults(metadata):
    defaults = builtin_runtime_defaults()
    # Defaults omitted from TLlama's argv are pinned common_params defaults
    # (b11850), not values discovered from this machine or its native binary.
    defaults.update(n_gpu_layers=None, device=None, fit=True, fit_target_mib=1024,
                    split_mode=1, main_gpu=0, tensor_split=None, seed=-1,
                    n_threads=-1, n_threads_batch=-1, rope_scaling_type=-1,
                    rope_freq_base=10000, rope_freq_scale=1, yarn_ext_factor=0,
                    yarn_attn_factor=1, yarn_beta_fast=32, yarn_beta_slow=1,
                    yarn_orig_ctx=0, offload_kqv=True, op_offload=True,
                    swa_full=False, no_perf=False, kv_overrides={},
                    lora_scale=1, spm_infill=False, spec_type='none', numa=0)
    defaults['n_ctx'] = metadata.get('context_length') or defaults['n_ctx']
    defaults['yarn_orig_ctx'] = defaults['n_ctx']
    raw = metadata.get('metadata_raw') or {}
    arch = metadata.get('arch', '')
    for name, key in {'rope_freq_base': 'rope.freq_base',
                      'yarn_orig_ctx': 'rope.scaling.original_context_length'}.items():
        if f'{arch}.{key}' in raw:
            defaults[name] = raw[f'{arch}.{key}']
    scaling = raw.get(f'{arch}.rope.scaling.type', 'linear')
    defaults['rope_scaling_type'] = {'none': 0, 'linear': 1, 'yarn': 2}.get(scaling, -1)
    defaults['yarn_ext_factor'] = 1 if scaling == 'yarn' else 0
    factor = raw.get(f'{arch}.rope.scaling.factor', raw.get(f'{arch}.rope.scale_linear'))
    if isinstance(factor, (int, float)) and factor > 0:
        defaults['rope_freq_scale'] = 1 / factor
    pooling = raw.get(f'{arch}.pooling_type', (metadata.get('embedding') or {}).get('pooling_type'))
    if pooling in (1, 2, 3):
        defaults['pooling_type'] = pooling
    return defaults


def _typed(value, annotation):
    if annotation == 'float':
        try:
            native = struct.unpack('f', struct.pack('f', value))[0]
            return native if math.isfinite(native) else value
        except OverflowError:
            return value
    if annotation == 'List[float]':
        return [_typed(item, 'float') for item in value]
    return value


def parameter_changes(spec, metadata, *, adapter_digest=None):
    """Only TOML changes against TLlama -> GGUF, in canonical parameter names."""
    changes = {}
    baseline = _runtime_defaults(metadata)
    runtime = dict(spec.runtime)
    for side in ('type_k', 'type_v'):
        if side in runtime or 'type_kv' in runtime:
            runtime[side] = kv_type_name(runtime.get(side, runtime.get('type_kv')))
    runtime.pop('type_kv', None)
    # Native sentinel spellings mean the same as inheriting GGUF/defaults.
    sentinels = {'n_ctx': 0, 'pooling_type': -1, 'rope_scaling_type': -1,
                 'rope_freq_base': 0, 'rope_freq_scale': 0, 'yarn_orig_ctx': 0}
    for key, value in runtime.items():
        if key == 'lora_path':
            if adapter_digest:
                changes['runtime.lora'] = adapter_digest
            continue
        if key in ('n_threads', 'n_threads_batch') and value <= 0:
            value = -1
        if key == 'seed' and value == 2**32 - 1:
            value = -1
        if key in sentinels and value == sentinels[key]:
            value = baseline[key]
        if key.startswith('yarn_') and key != 'yarn_orig_ctx' and value < 0:
            value = baseline[key]
        if key == 'numa':
            value = int(value)
        if key == 'draft_n_max':
            if runtime.get('spec_type', 'none') != 'draft-mtp' or value == MTP_DRAFT_N_MAX_DEFAULT:
                continue
        value = _typed(value, RUNTIME_TYPES.get(key))
        default = baseline.get(key)
        if default is not None:
            default = _typed(default, RUNTIME_TYPES.get(key))
        if value != default:
            if key == 'kv_overrides':
                value = {name: {'type': 'bool' if isinstance(item, bool) else
                                'int' if isinstance(item, int) else
                                'float' if isinstance(item, float) else 'str',
                                'value': _typed(item, 'float') if isinstance(item, float) else item}
                         for name, item in value.items()}
            changes['runtime.' + key] = value

    # Pass only the GGUF layer: metadata exposed by /show can also carry TOML
    # or server-derived fields, which must not contaminate this comparison.
    sampling_metadata = {'recommended_sampling': metadata.get('recommended_sampling') or {}}
    before = build_sampling_kwargs({}, sampling_metadata)
    after = build_sampling_kwargs({}, dict(sampling_metadata, sampling_defaults=spec.sampling,
                                           stop_defaults=spec.stop))
    for key, value in after.items():
        if key == 'seed':
            # Unset request seeds inherit the worker. Random has two native
            # spellings; an explicit random seed only changes a fixed worker.
            if value == 2**32 - 1:
                value = -1
            if value == -1 and runtime.get('seed', -1) in (-1, 2**32 - 1):
                continue
        value = _typed(value, SAMPLING_TYPES.get(key))
        default = before.get(key, [] if key == 'stop' else None)
        if default is not None:
            default = _typed(default, SAMPLING_TYPES.get(key))
        if value != default:
            changes['sampling.' + key] = value
    if spec.system_prompt not in (None, ''):
        changes['system.prompt'] = spec.system_prompt
    template = metadata.get('template') or ''
    tool_template = metadata.get('tool_template') or template
    if spec.template and (spec.template != template or tool_template != template):
        changes['template.jinja'] = spec.template
    if spec.fim_order != 'auto':
        changes['fim.order'] = spec.fim_order
    for key, value in spec.mmproj_context.items():
        if key in ('image_min_tokens', 'image_max_tokens') and value <= 0:
            value = -1
        if value != {'warmup': True, 'image_min_tokens': -1, 'image_max_tokens': -1,
                     'batch_max_tokens': 1024}[key]:
            changes['mmproj.' + key] = value
    return _normal(changes)


def file_group_digest(digests):
    """Single-file groups preserve their existing raw SHA-256."""
    if not digests or any(not value for value in digests):
        return ''
    if len(digests) == 1:
        return digests[0]
    return sha256(canonical_json({'format': 'tllama-gguf-shards-v1', 'shards': digests})).hexdigest()


def model_digest(llm, projector='', changes=None):
    if not llm:
        return ''
    parts = {'format': 'tllama-model-id-v1', 'llm': llm}
    if projector:
        parts['mmproj'] = projector
    if changes:
        parts['parameters'] = sha256(canonical_json(changes)).hexdigest()
    if len(parts) == 2:
        return llm
    return sha256(canonical_json(parts)).hexdigest()
