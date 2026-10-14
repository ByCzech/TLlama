"""Private native worker launch parameters, independent of inference bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


RUNTIME_FINGERPRINT = "b11552-ec2797836f"
DEFAULT_EXECUTABLE = Path("/usr/lib/tllama/server/tllama-server")
_OWNED_ENV = {"GGML_BACKEND_PATH", "MTMD_BACKEND_DEVICE", "LLAMA_API_KEY"}


def _owned_environment(name: str) -> bool:
    return name.startswith(("LLAMA_ARG_", "AIP_")) or name in _OWNED_ENV


@dataclass(frozen=True)
class WorkerLaunchSpec:
    """Resolved configuration for one worker; public TOML mapping comes later."""

    model: Path
    device: str
    alias: str = "tllama-worker"
    executable: Path = DEFAULT_EXECUTABLE
    mmproj: Path | None = None
    n_ctx: int = 2048
    n_gpu_layers: int | None = None
    type_k: str | None = None
    type_v: str | None = None
    flash_attn: bool | None = None
    port: int = 0
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    runtime_fingerprint: str = RUNTIME_FINGERPRINT
    load_timeout: float = 600.0
    request_timeout: float | None = None
    terminate_timeout: float = 5.0
    kill_timeout: float = 5.0
    log_limit_bytes: int = 4 * 1024 * 1024

    def __post_init__(self):
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
            if not isinstance(value, str) or not value or "\0" in value:
                raise ValueError(f"{name} must be a nonempty string")
        devices = self.device.split(",")
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

    def child_environment(self, inherited: Mapping[str, str]) -> dict[str, str]:
        """Copy the environment, removing unrelated native server configuration."""
        env = {name: value for name, value in inherited.items() if not _owned_environment(name)}
        env.update(self.environment)
        return env

    def command(self, api_key_file: Path) -> tuple[str, ...]:
        if not api_key_file.is_absolute():
            raise ValueError("api_key_file must be an absolute path")
        layers = 0 if self.device == "none" else self.n_gpu_layers
        layers_arg = "all" if layers in (None, -1) else str(layers)
        args = [str(self.executable), "--model", str(self.model), "--alias", self.alias,
                "--host", "127.0.0.1", "--port", str(self.port), "--parallel", "1",
                "--device", self.device, "--n-gpu-layers", layers_arg,
                "--ctx-size", str(self.n_ctx), "--fit", "off", "--sleep-idle-seconds", "-1",
                "--api-key-file", str(api_key_file), "--offline", "--no-context-shift",
                "--log-colors", "off", "--no-log-prefix", "--no-log-timestamps",
                "--log-verbosity", "5"]
        if self.device == "none":
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
        return tuple(args)
