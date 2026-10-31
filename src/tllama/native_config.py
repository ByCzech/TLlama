"""Resolve TLlama configuration into worker launches and native sampling.

The caller resolves model/projector storage paths. This module performs no
inference, metadata probing, file writes or process-global environment changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tllama.config import BackendConfig, ConfigError
from tllama.helpers.common import build_sampling_kwargs
from tllama.helpers.fim import FIM_ORDERS
from tllama.helpers.model_toml import VirtualModelSpec
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides
from tllama.native_cli import RUNTIME_OPTION_NAMES
from tllama.native_params import freeze_values, kv_type_name, runtime_problem, MTP_DRAFT_N_MAX_DEFAULT
from tllama.worker_config import WorkerLaunchSpec
from tllama.decision_contract import decision_type


_DEFAULTS = {"n_gpu_layers": -1, "flash_attn": False, "n_batch": 512, "n_ubatch": 512,
             "use_mmap": True, "use_mlock": False, "last_n_tokens_size": 64,
             "type_k": "f16", "type_v": "f16", "embedding": False, "pooling_type": -1,
             "context_shift": False}


def builtin_runtime_defaults():
    """Copy TLlama's fixed launch defaults, without server/environment layers."""
    return dict(_DEFAULTS, n_ctx=BackendConfig.__dataclass_fields__['context_length'].default)


@dataclass(frozen=True)
class ResolvedWorkerConfig:
    launch: WorkerLaunchSpec
    runtime: Mapping
    sources: Mapping[str, str] = field(compare=False)
    global_sampling: Mapping
    model_sampling: Mapping | None
    model_stop: tuple[str, ...] | None
    system_prompt: str | None
    fim_order: str = "auto"


def _plain(value):
    """Copy TOML scalar subclasses/containers into ordinary JSON-compatible data."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def resolve_worker_config(
    config: BackendConfig, model: Path, virtual_spec: VirtualModelSpec | None = None, *,
    num_ctx: int | None = None, mmproj: Path | None = None,
    environment: Mapping[str, str] | None = None, executable: Path | None = None,
    alias: str = "tllama-worker", load_timeout: float = 600,
    request_timeout: float | None = None,
    metadata: Mapping | None = None, runtime_overrides: Mapping | None = None,
) -> ResolvedWorkerConfig:
    if virtual_spec is not None and (not isinstance(virtual_spec.fim_order, str)
                                     or virtual_spec.fim_order not in FIM_ORDERS):
        raise ConfigError("Invalid FIM order in model configuration")
    runtime = dict(_DEFAULTS)
    sources = {key: "baseline" for key in runtime}
    runtime["n_ctx"] = config.context_length
    sources["n_ctx"] = "global"

    def overlay(values, origin):
        for key, value in values.items():
            problem = runtime_problem(key, value)
            if problem:
                raise ConfigError(f"{origin} runtime: {problem}")
            if key == "type_kv":
                continue
            runtime[key] = _plain(value)
            sources[key] = origin
        # Expand each layer independently. A TOML shorthand must override
        # both global sides; explicit sides only override their own shorthand.
        for side in ("type_k", "type_v"):
            if side in values or "type_kv" in values:
                runtime[side] = kv_type_name(values.get(side, values.get("type_kv")))
                sources[side] = origin

    overlay(coerce_runtime_overrides(config.runtime_overrides), "global")
    global_kv = {key: value for key, value in {
        "type_kv": config.kv_cache_type, "type_k": config.k_cache_type, "type_v": config.v_cache_type,
    }.items() if value is not None}
    overlay(global_kv, "global")
    if virtual_spec is not None:
        overlay(virtual_spec.runtime, "TOML")
    if num_ctx is not None:
        overlay({"n_ctx": num_ctx}, "request")
    overlay(runtime_overrides or {}, "request")
    for name in ("n_threads", "n_threads_batch"):
        if name in (runtime_overrides or {}) and runtime_overrides[name] <= 0:
            runtime.pop(name, None)
            sources.pop(name, None)
    if (runtime_overrides or {}).get("n_gpu_layers") == 0:
        runtime["device"], sources["device"] = "none", "request CPU placement"
        for flag in ("op_offload", "offload_kqv"):
            runtime[flag], sources[flag] = False, "request CPU placement"
    if (runtime_overrides or {}).get("context_shift") is True and mmproj is not None:
        raise ConfigError("num_keep context shifting is unsupported by the native multimodal runtime")
    if runtime.get("spec_type") == "draft-mtp" and "draft_n_max" not in runtime:
        runtime["draft_n_max"] = MTP_DRAFT_N_MAX_DEFAULT
        sources["draft_n_max"] = "MTP default"
    if (runtime_overrides or {}).get("n_gpu_layers") == -1:
        runtime["n_gpu_layers"] = None
    device = runtime.get("device")
    if device is None:
        sources["device"] = "backend priority"
        runtime["device"] = None
    if (sources["n_gpu_layers"] == "baseline" and runtime.get("fit") is not False
            and (device is None or runtime.get("fit") is True)):
        runtime["n_gpu_layers"], sources["n_gpu_layers"] = None, "native fit"
    if device == "none":
        if sources["n_gpu_layers"] != "baseline" and runtime["n_gpu_layers"] not in (0, None):
            raise ConfigError("CPU device 'none' conflicts with explicit n_gpu_layers; use 0")
        for flag in ("op_offload", "offload_kqv"):
            if runtime.get(flag) is True:
                raise ConfigError(f"CPU device 'none' conflicts with explicit {flag}=true")
            runtime[flag], sources[flag] = False, "CPU placement"
        runtime["n_gpu_layers"], sources["n_gpu_layers"] = 0, "CPU placement"
    if "lora_path" in runtime:
        root = Path(config.models_dir).expanduser().resolve()
        path = Path(runtime["lora_path"])
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        # Match the model-store boundary for configuration paths. Callers
        # must not expose arbitrary native local-file access through TOML.
        if not path.is_relative_to(root):
            raise ConfigError("lora_path must resolve inside the model store")
        runtime["lora_path"] = str(path)
    if virtual_spec is not None and virtual_spec.mmproj_model is not None and mmproj is None:
        raise ConfigError("Resolve the TOML projector path before constructing its worker")
    if virtual_spec is not None and virtual_spec.mmproj_from is not None and mmproj is None:
        raise ConfigError("Import and resolve the TOML projector before constructing its worker")
    try:
        decision = decision_type(metadata or {})
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if (runtime["embedding"] or decision is not None) and virtual_spec is not None:
        if (virtual_spec.sampling or virtual_spec.stop or virtual_spec.system_prompt is not None
                or virtual_spec.fim_order != "auto"):
            raise ConfigError("Embedding/Decision workers do not accept model sampling, system prompts or FIM settings")
    sampling = coerce_sampling_overrides(config.sampling_overrides)
    fit = runtime.get("fit")
    if device == "none":
        fit = False
    elif device is not None and fit is None:
        fit = runtime["n_gpu_layers"] is None
    try:
        launch = WorkerLaunchSpec(
            model=model, executable=config.runtime_executable if executable is None else executable,
            device=device, alias=alias, mmproj=mmproj,
            backend_priority=config.backend_priority,
            version_policy=config.runtime_version_policy,
            memory_source=config.memory_source, memory_compare=config.memory_compare,
            n_ctx=int(runtime["n_ctx"]), fit=fit,
            n_gpu_layers=None if runtime["n_gpu_layers"] is None else int(runtime["n_gpu_layers"]),
            type_k=runtime["type_k"], type_v=runtime["type_v"], flash_attn=runtime["flash_attn"],
            embedding=runtime["embedding"], pooling_type=runtime["pooling_type"],
            decision_type=decision,
            environment=environment or {}, load_timeout=load_timeout, request_timeout=request_timeout,
            runtime_options={key: value for key, value in runtime.items() if key in RUNTIME_OPTION_NAMES},
            mmproj_options=_plain(virtual_spec.mmproj_context) if virtual_spec else {},
            template=virtual_spec.template if virtual_spec else None,
        )
    except ValueError as exc:
        raise ConfigError(f"Native launch configuration: {exc}") from exc
    return ResolvedWorkerConfig(launch, freeze_values(runtime), MappingProxyType(sources),
                                freeze_values(sampling), freeze_values(virtual_spec.sampling) if virtual_spec else None,
                                tuple(virtual_spec.stop) if virtual_spec else None,
                                virtual_spec.system_prompt if virtual_spec else None,
                                virtual_spec.fim_order if virtual_spec else "auto")


def build_native_sampling(opts: dict, metadata_info: dict | None = None, *,
                          resolved: ResolvedWorkerConfig | None = None,
                          global_sampling: dict | None = None) -> dict:
    """Preserve TLlama precedence, then translate its sampler vocabulary."""
    if resolved is not None:
        if global_sampling is not None:
            raise ValueError("global_sampling is already carried by resolved configuration")
        global_sampling = resolved.global_sampling
        metadata_info = dict(metadata_info or {})
        if resolved.model_sampling is not None:
            metadata_info["sampling_defaults"] = _plain(resolved.model_sampling)
            metadata_info["stop_defaults"] = list(resolved.model_stop)
    values = build_sampling_kwargs(opts, metadata_info, global_sampling)
    limit = values.pop("max_tokens")
    values["n_predict"] = -1 if limit is None else limit
    values["mirostat"] = values.pop("mirostat_mode")
    values["top_k"] = max(0, values["top_k"])
    if values.get("seed") is None:
        values.pop("seed", None)  # Let the explicitly configured worker seed apply.
    if "logit_bias" in values:
        values["logit_bias"] = {str(token): bias for token, bias in values["logit_bias"].items()}
    values["repeat_last_n"] = opts.get("repeat_last_n", resolved.runtime["last_n_tokens_size"] if resolved else 64)
    if "num_keep" in opts:
        values["n_keep"] = opts["num_keep"]
    return values
