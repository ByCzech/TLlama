"""Private native worker launch parameters, independent of inference bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Any

from tllama.native_cli import runtime_cli_args, mmproj_cli_args
from tllama.native_params import POOLING_TYPES, freeze_values, kv_type_name
from tllama.decision_contract import SUPPORTED_DECISION_RUNTIME_FINGERPRINTS, DECISION_TYPES
from tllama.backend_policy import DEFAULT_BACKEND_PRIORITY, backend_priority


# Reviewed 0.7.0 captures and limitations: docs/runtime-audit-070.md.
RUNTIME_FINGERPRINT = "b11850-0e9433bfd3"
SUPPORTED_RUNTIME_FINGERPRINTS = frozenset({RUNTIME_FINGERPRINT, "b11552-ec2797836f",
                                         "b11552-d1570abd3d", "b11552-3144c167ef"})
DEFAULT_EXECUTABLE = Path("/usr/lib/tllama/server/tllama-server")
_OWNED_ENV = {"GGML_BACKEND_PATH", "MTMD_BACKEND_DEVICE", "LLAMA_API_KEY"}


def _owned_environment(name: str) -> bool:
    return name.startswith(("LLAMA_ARG_", "AIP_")) or name in _OWNED_ENV


@dataclass(frozen=True)
class WorkerLaunchSpec:
    """Resolved load configuration for one private native process."""

    model: Path
    device: str | None
    alias: str = "tllama-worker"
    executable: Path = DEFAULT_EXECUTABLE
    mmproj: Path | None = None
    n_ctx: int = 2048
    n_gpu_layers: int | None = None
    backend_priority: tuple[str, ...] = DEFAULT_BACKEND_PRIORITY
    # Unset preserves the policy: automatic selection fits unset layer counts;
    # a direct explicit-device launch defaults to fitting off.
    fit: bool | None = None
    type_k: str | None = None
    type_v: str | None = None
    flash_attn: bool | None = None
    embedding: bool = False
    pooling_type: int = -1
    decision_type: str | None = None
    transport: str = "uds"
    port: int = 0
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    mmproj_options: Mapping[str, Any] = field(default_factory=dict)
    template: str | None = None
    runtime_fingerprint: str = RUNTIME_FINGERPRINT
    version_policy: str = "warn"
    memory_source: str = "auto"
    memory_compare: bool = False
    memory_tensor_diagnostics: int = 0
    load_timeout: float = 600.0
    request_timeout: float | None = None
    terminate_timeout: float = 5.0
    kill_timeout: float = 5.0
    log_limit_bytes: int = 4 * 1024 * 1024
    # An explicit diagnostic pin, including a pin to the default build. This
    # is independent of per-feature audit lists and never changes them.
    expected_runtime_fingerprint: str | None = None

    @property
    def accepted_runtime_fingerprints(self) -> frozenset[str]:
        """Select evidence for this mode; enforcement uses the shared policy."""
        if self.expected_runtime_fingerprint is not None:
            return frozenset({self.expected_runtime_fingerprint})
        if self.decision_type is not None:
            return SUPPORTED_DECISION_RUNTIME_FINGERPRINTS
        # Preserve the existing custom launch-field behavior for callers that
        # predate explicit pins. Normal production launches use the audit set.
        if self.runtime_fingerprint != RUNTIME_FINGERPRINT:
            return frozenset({self.runtime_fingerprint})
        return SUPPORTED_RUNTIME_FINGERPRINTS

    def __post_init__(self):
        object.__setattr__(self, "backend_priority", backend_priority(self.backend_priority))
        if self.fit is not None and type(self.fit) is not bool:
            raise ValueError("fit must be a boolean or unset")
        if self.version_policy not in ("warn", "strict"):
            raise ValueError("version_policy must be warn or strict")
        if self.memory_source not in ("auto", "native", "log"):
            raise ValueError("memory_source must be auto, native or log")
        if type(self.memory_compare) is not bool:
            raise ValueError("memory_compare must be a boolean")
        if type(self.memory_tensor_diagnostics) is not int or not 0 <= self.memory_tensor_diagnostics <= 4096:
            raise ValueError("memory_tensor_diagnostics must be an integer in 0..4096")
        if self.transport not in ("uds", "tcp"):
            raise ValueError("transport must be uds or tcp")
        if self.transport == "uds" and self.port != 0:
            raise ValueError("port requires transport=tcp")
        for name in ("model", "executable", "mmproj"):
            value = getattr(self, name)
            if value is None and name == "mmproj":
                continue
            path = Path(value)
            if not path.is_absolute() or "\0" in str(path):
                raise ValueError(f"{name} must be an absolute path")
            object.__setattr__(self, name, path)

        for name in ("device", "alias", "runtime_fingerprint"):
            value = getattr(self, name)
            if name == "device" and value is None:
                continue
            if not isinstance(value, str) or not value or "\0" in value:
                raise ValueError(f"{name} must be a nonempty string")
        expected = self.expected_runtime_fingerprint
        if expected is not None and (not isinstance(expected, str) or not expected.strip()
                                     or expected != expected.strip() or "\0" in expected):
            raise ValueError("expected_runtime_fingerprint must be a nonblank identity")
        devices = self.device.split(",") if self.device is not None else []
        if any(not part or part.strip() != part for part in devices):
            raise ValueError("device must contain nonempty server device selectors")
        if "none" in devices and len(devices) != 1:
            raise ValueError("device 'none' cannot be combined with other devices")
        if "," in self.alias:
            raise ValueError("alias must name exactly one model")

        for name, minimum, maximum in (("n_ctx", 0, None), ("port", 0, 65535),
                                       ("log_limit_bytes", 1024, None)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"Invalid {name}: {value!r}")
        if self.n_gpu_layers is not None:
            if type(self.n_gpu_layers) is not int or self.n_gpu_layers < -1:
                raise ValueError("n_gpu_layers must be -1 (all), a nonnegative integer, or unset")
        if self.device == "none" and self.n_gpu_layers not in (None, 0):
            raise ValueError("CPU placement requires n_gpu_layers=0 or unset")
        if self.flash_attn is not None and type(self.flash_attn) is not bool:
            raise ValueError("flash_attn must be a boolean or unset")
        for name in ("type_k", "type_v"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value or "\0" in value):
                raise ValueError(f"{name} must be a KV cache type name or unset")
            if value is not None:
                object.__setattr__(self, name, kv_type_name(value))
        runtime_cli_args(self.runtime_options)
        mmproj_cli_args(self.mmproj_options)
        if type(self.embedding) is not bool:
            raise ValueError("embedding must be a boolean")
        if type(self.pooling_type) is not int or self.pooling_type not in POOLING_TYPES:
            raise ValueError("pooling_type must be -1, 1, 2 or 3 for pooled embeddings")
        if not self.embedding and self.pooling_type != -1:
            raise ValueError("pooling_type requires embedding=true")
        if self.embedding:
            if self.template is not None:
                raise ValueError("Embedding workers do not accept a template override")
            if (self.runtime_options.get("spec_type", "none") != "none"
                    or "draft_n_max" in self.runtime_options or self.runtime_options.get("spm_infill")):
                raise ValueError("Embedding workers do not accept speculative decoding or FIM settings")
        if self.decision_type is not None:
            if not isinstance(self.decision_type, str) or self.decision_type not in DECISION_TYPES:
                raise ValueError("Decision worker type must be laya, nimble or clef")
            if self.embedding or self.pooling_type != -1 or self.template is not None or self.mmproj is not None:
                raise ValueError("Text Decision workers reject embedding mode, pooling, template overrides and projectors")
            if (self.runtime_options.get("spec_type", "none") != "none" or "draft_n_max" in self.runtime_options
                    or self.runtime_options.get("spm_infill") or self.runtime_options.get("kv_overrides")
                    or "lora_path" in self.runtime_options):
                raise ValueError("Decision workers reject speculative decoding, FIM, metadata overrides and LoRA")
        if self.mmproj_options and self.mmproj is None:
            raise ValueError("Projector options require a resolved mmproj path")
        if self.device == "none" and any(self.runtime_options.get(key) is True for key in ("op_offload", "offload_kqv")):
            raise ValueError("CPU placement conflicts with explicitly enabled offload")
        if self.template is not None and (not isinstance(self.template, str) or "\0" in self.template):
            raise ValueError("template must be a Jinja string without NUL or unset")
        object.__setattr__(self, "runtime_options", freeze_values(self.runtime_options))
        object.__setattr__(self, "mmproj_options", freeze_values(self.mmproj_options))
        for name in ("load_timeout", "request_timeout", "terminate_timeout", "kill_timeout"):
            value = getattr(self, name)
            if value is None and name == "request_timeout":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive duration")

        environment = dict(self.environment)
        for name, value in environment.items():
            if not isinstance(name, str) or not name or "=" in name or "\0" in name:
                raise ValueError("Invalid worker environment name")
            if not isinstance(value, str) or "\0" in value:
                raise ValueError(f"Invalid environment value for {name}")
            if _owned_environment(name):
                raise ValueError(f"{name} is controlled by the worker launch contract")
        object.__setattr__(self, "environment", MappingProxyType(environment))

    @property
    def mode(self) -> str:
        return "decision" if self.decision_type is not None else "embedding" if self.embedding else "generation"

    def child_environment(self, inherited: Mapping[str, str]) -> dict[str, str]:
        """Copy the environment, removing unrelated native server configuration."""
        env = {name: value for name, value in inherited.items() if not _owned_environment(name)}
        env.update(self.environment)
        return env

    def command(self, api_key_file: Path, template_file: Path | None = None,
                socket_file: Path | None = None) -> tuple[str, ...]:
        if self.device is None:
            raise ValueError("Automatic device selection must resolve before building a worker command")
        if not api_key_file.is_absolute():
            raise ValueError("api_key_file must be an absolute path")
        if socket_file is not None and (not socket_file.is_absolute() or "\0" in str(socket_file)
                                        or not str(socket_file).endswith('.sock')):
            raise ValueError("socket_file must be an absolute .sock path")
        host = str(socket_file or api_key_file.parent / 'worker.sock') if self.transport == 'uds' else '127.0.0.1'
        layers = 0 if self.device == "none" else self.n_gpu_layers
        layers_arg = "auto" if self.fit and layers is None else "all" if layers in (None, -1) else str(layers)
        args = [str(self.executable), "--model", str(self.model), "--alias", self.alias,
                "--host", host, "--port", str(self.port), "--parallel", "1",
                "--device", self.device, "--n-gpu-layers", layers_arg,
                "--ctx-size", str(self.n_ctx), "--fit", "on" if self.fit else "off", "--sleep-idle-seconds", "-1",
                "--api-key-file", str(api_key_file), "--offline",
                "--log-colors", "off", "--no-log-prefix", "--no-log-timestamps",
                "--log-verbosity", "5"]
        if self.device == "none" and "op_offload" not in self.runtime_options:
            args.append("--no-op-offload")
        if self.mmproj is not None:
            args += ["--mmproj", str(self.mmproj)]
            if self.device == "none":
                args.append("--no-mmproj-offload")
        if self.flash_attn is not None:
            args += ["--flash-attn", "on" if self.flash_attn else "off"]
        for name, flag in (("type_k", "--cache-type-k"), ("type_v", "--cache-type-v")):
            value = getattr(self, name)
            if value is not None:
                args += [flag, value]
        if "context_shift" not in self.runtime_options:
            args.append("--no-context-shift")
        args += runtime_cli_args(self.runtime_options)
        args += mmproj_cli_args(self.mmproj_options)
        if self.memory_tensor_diagnostics:
            args += ["--memory-tensor-diagnostics", str(self.memory_tensor_diagnostics)]
        if self.decision_type is not None and "spec_type" not in self.runtime_options:
            args += ["--spec-type", "none"]
        if self.embedding:
            args.append("--embeddings")
            if "spec_type" not in self.runtime_options:
                args += ["--spec-type", "none"]
            if POOLING_TYPES[self.pooling_type] is not None:
                args += ["--pooling", POOLING_TYPES[self.pooling_type]]
        if self.template is not None:
            if template_file is None or not template_file.is_absolute():
                raise ValueError("A private absolute template_file is required for a template override")
            args += ["--jinja", "--chat-template-file", str(template_file)]
        return tuple(args)
