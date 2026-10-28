"""TLlama configuration vocabulary; embedded MTP audited in runtime-audit-070.md.

These are TLlama contracts, not a reflection of a Python binding or every
option the native CLI happens to expose. Update with the compatibility matrix.
"""

from __future__ import annotations

import math
from typing import Any, Mapping
from types import MappingProxyType


RUNTIME_TYPES = {
    "device": "str", "n_ctx": "int", "n_gpu_layers": "int",
    "fit_target_mib": "int",
    "split_mode": "int", "main_gpu": "int", "tensor_split": "List[float]",
    "use_mmap": "bool", "use_mlock": "bool", "kv_overrides": "Dict",
    "seed": "int", "n_batch": "int", "n_ubatch": "int",
    "n_threads": "int", "n_threads_batch": "int", "rope_scaling_type": "int",
    "rope_freq_base": "float", "rope_freq_scale": "float",
    "yarn_ext_factor": "float", "yarn_attn_factor": "float",
    "yarn_beta_fast": "float", "yarn_beta_slow": "float", "yarn_orig_ctx": "int",
    "offload_kqv": "bool", "flash_attn": "bool", "op_offload": "bool",
    "swa_full": "bool", "no_perf": "bool", "last_n_tokens_size": "int",
    "lora_path": "str", "lora_scale": "float", "numa": "Union[bool,int]",
    "type_k": "Union[str,int]", "type_v": "Union[str,int]", "type_kv": "Union[str,int]",
    "spm_infill": "bool",
    "spec_type": "str", "draft_n_max": "int",
    "embedding": "bool", "pooling_type": "int", "context_shift": "bool",
}

MTP_DRAFT_N_MAX_DEFAULT = 3

SAMPLING_TYPES = {
    "temperature": "float", "top_p": "float", "top_k": "int", "min_p": "float",
    "typical_p": "float", "presence_penalty": "float", "frequency_penalty": "float",
    "repeat_penalty": "float", "mirostat_mode": "int", "mirostat_tau": "float",
    "mirostat_eta": "float", "seed": "Optional[int]", "max_tokens": "Optional[int]",
    "stop": "List[str]", "logit_bias": "Dict",
}

# Pinned common/arg.cpp::kv_cache_types and ggml/include/ggml.h. This intentionally
# refuses weight quantization types that the server cannot use for its KV cache.
KV_TYPES = {"f32": 0, "f16": 1, "bf16": 30, "q8_0": 8, "q4_0": 2,
            "q4_1": 3, "iq4_nl": 20, "q5_0": 6, "q5_1": 7}
SPLIT_MODES = {0: "none", 1: "layer", 2: "row"}
ROPE_MODES = {-1: None, 0: "none", 1: "linear", 2: "yarn"}
NUMA_MODES = {0: None, 1: "distribute", 2: "isolate", 3: "numactl"}
POOLING_TYPES = {-1: None, 1: "mean", 2: "cls", 3: "last"}


def freeze_values(value):
    """Copy nested configuration so frozen launch contracts cannot be mutated."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_values(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_values(item) for item in value)
    return value


def kv_type_name(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("KV cache type must be a string or int, not bool")
    if not isinstance(value, (str, int)):
        raise ValueError(f"KV cache type must be a string or int, got {type(value).__name__}")
    if isinstance(value, str) and value.lower() in KV_TYPES:
        return value.lower()
    if isinstance(value, int) and not isinstance(value, bool):
        for name, enum in KV_TYPES.items():
            if enum == value:
                return name
    hint = f" (GGML_TYPE_{value.upper()})" if isinstance(value, str) else ""
    raise ValueError(f"unsupported native KV cache type {value!r}{hint}; expected {', '.join(KV_TYPES)}")


def device_problem(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\0" in value:
        return "device must be a nonempty native selector"
    parts = value.split(",")
    if any(not part or part.strip() != part for part in parts):
        return "device must contain nonempty selectors without surrounding whitespace"
    if "none" in parts and len(parts) != 1:
        return "device 'none' cannot be combined with GPU selectors"
    return None


def _type_problem(name: str, value: Any, annotation: str) -> str | None:
    if annotation.startswith("Optional["):
        if value is None:
            return None
        annotation = annotation[9:-1]
    if annotation == "bool":
        ok = type(value) is bool
    elif annotation == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif annotation == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif annotation == "str":
        ok = isinstance(value, str) and "\0" not in value
    elif annotation == "Union[bool,int]":
        ok = isinstance(value, (bool, int))
    elif annotation == "Union[str,int]":
        ok = isinstance(value, (str, int)) and not isinstance(value, bool)
    elif annotation == "List[float]":
        ok = isinstance(value, (list, tuple)) and all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in value
        )
    elif annotation == "List[str]":
        ok = isinstance(value, list) and all(isinstance(v, str) and "\0" not in v for v in value)
    elif annotation == "Dict":
        ok = isinstance(value, Mapping)
    else:
        raise AssertionError(annotation)
    if not ok:
        if annotation == "List[str]":
            return f"'{name}' must be an array of strings, got {type(value).__name__}"
        return f"'{name}' must be {annotation}, got {type(value).__name__}: {value!r}"
    return None


def runtime_problem(name: str, value: Any) -> str | None:
    if name == "model_path":
        return "'model_path' cannot be set here: the model path comes from [llm]"
    annotation = RUNTIME_TYPES.get(name)
    if annotation is None:
        return f"'{name}' is not a runtime parameter TLlama applies"
    problem = _type_problem(name, value, annotation)
    if problem:
        return problem
    if annotation == "int" and name != "seed" and not -(2**31) <= value < 2**31:
        return f"'{name}' must fit the native int32 CLI parameter"
    if name == "seed" and not -1 <= value < 2**32:
        return "'seed' must be -1 (random) or uint32"
    if name == "device":
        return device_problem(value)
    if name == "pooling_type" and value not in POOLING_TYPES:
        return "'pooling_type' must be -1 (model default), 1 (mean), 2 (cls) or 3 (last); token vectors and reranking are separate tasks"
    if name == "spec_type" and value not in {"none", "draft-mtp"}:
        return "'spec_type' must be none or draft-mtp; other speculative strategies are not integrated"
    if name == "draft_n_max" and value < 0:
        return "'draft_n_max' must be nonnegative; use spec_type=none to disable speculation"
    if name.startswith("type_"):
        try:
            kv_type_name(value)
        except ValueError as exc:
            return f"'{name}': {exc}"
    if name in ("n_ctx", "main_gpu", "yarn_orig_ctx") and value < 0:
        return f"'{name}' must be nonnegative"
    if name in ("n_batch", "n_ubatch") and value <= 0:
        return f"'{name}' must be positive"
    if name == "n_gpu_layers" and value < -1:
        return "'n_gpu_layers' must be -1 (all) or nonnegative"
    if name == "fit_target_mib" and value < 0:
        return "'fit_target_mib' must be nonnegative (MiB per device)"
    if name in ("split_mode", "rope_scaling_type", "numa"):
        allowed = {"split_mode": SPLIT_MODES, "rope_scaling_type": ROPE_MODES, "numa": NUMA_MODES}[name]
        if value not in allowed:
            return f"'{name}' must be one of {', '.join(map(str, allowed))}; native parser has no mapping for {value!r}"
    if name == "tensor_split" and (not value or any(v < 0 for v in value) or not any(value)):
        return "'tensor_split' needs nonnegative proportions with a positive total"
    if name == "last_n_tokens_size" and value < -1:
        return "'last_n_tokens_size' must be -1 or nonnegative"
    if name == "lora_path" and not value:
        return "'lora_path' must not be empty"
    if name == "kv_overrides":
        for key, item in value.items():
            if (not isinstance(key, str) or not key or any(c in key for c in "=\0\r\n")
                    or len(key.encode()) > 127):
                return "'kv_overrides' keys must be nonempty names up to 127 UTF-8 bytes without '=' or control delimiters"
            if isinstance(item, str):
                if "\0" in item or len(item.encode()) > 127:
                    return "'kv_overrides' string values must fit 127 UTF-8 bytes without NUL"
            elif type(item) is bool:
                pass
            elif isinstance(item, int):
                if not -(2**63) <= item < 2**63:
                    return "'kv_overrides' integers must fit int64"
            elif isinstance(item, float) and math.isfinite(item):
                pass
            else:
                return "'kv_overrides' values must be bool, int64, finite float or string"
    return None


def sampling_problem(name: str, value: Any) -> str | None:
    if name == "tfs_z":
        return "'tfs_z' was removed: the native runtime has no tail-free sampler equivalent"
    annotation = SAMPLING_TYPES.get(name)
    if annotation is None:
        return f"'{name}' is not a sampling parameter TLlama applies"
    problem = _type_problem(name, value, annotation)
    if problem:
        return problem
    if name == "seed" and value is not None and not -1 <= value < 2**32:
        return "'seed' must be -1 (random) or uint32"
    if annotation in ("int", "Optional[int]") and name != "seed" and value is not None and not -(2**31) <= value < 2**31:
        return f"'{name}' must fit the native int32 request parameter"
    if name == "temperature" and value < 0:
        return "'temperature' must be nonnegative; native negative values would become greedy"
    if name == "mirostat_mode" and value not in (0, 1, 2):
        return f"'mirostat_mode' must be one of 0, 1, 2, got {value}; any other value turns mirostat off without saying so"
    if name == "logit_bias":
        for token, bias in value.items():
            if (type(token) is not int or token < 0 or isinstance(bias, bool)
                    or not isinstance(bias, (int, float)) or not math.isfinite(bias)):
                return "'logit_bias' must map nonnegative integer token IDs to finite numbers"
    return None
