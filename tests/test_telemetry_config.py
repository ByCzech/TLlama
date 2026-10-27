"""Global engine/telemetry policy must not leak into per-model CLI options."""

from pathlib import Path

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.daemon_config import load_daemon_config
from tllama.native_config import resolve_worker_config
from tllama.worker_config import DEFAULT_EXECUTABLE, WorkerLaunchSpec


def test_environment_globals_reach_launch_without_becoming_runtime_parameters(monkeypatch):
    monkeypatch.setenv("TLLAMA_RUNTIME_EXECUTABLE", "/opt/experimental build/llama-server")
    monkeypatch.setenv("TLLAMA_RUNTIME_VERSION_POLICY", "STRICT")
    monkeypatch.setenv("TLLAMA_MEMORY_SOURCE", "log")
    monkeypatch.setenv("TLLAMA_MEMORY_COMPARE", "yes")
    monkeypatch.setenv("TLLAMA_RUNTIME_DEVICE", "Vulkan0")
    config = load_daemon_config(False, discover=False).backend
    assert config.runtime_overrides == {"device": "Vulkan0"}
    resolved = resolve_worker_config(config, Path("/models/m.gguf"))
    assert resolved.launch.executable == Path("/opt/experimental build/llama-server")
    assert resolved.launch.version_policy == "strict"
    assert resolved.launch.memory_source == "log" and resolved.launch.memory_compare
    assert resolved.launch.command(Path("/key"))[0] == str(config.runtime_executable)
    assert "memory_source" not in resolved.runtime
    assert resolve_worker_config(config, Path("/models/m.gguf"), executable=Path("/explicit/server")).launch.executable == Path("/explicit/server")


@pytest.mark.parametrize("name,value", [
    ("TLLAMA_RUNTIME_EXECUTABLE", "relative/server"),
    ("TLLAMA_RUNTIME_VERSION_POLICY", "ignore"), ("TLLAMA_MEMORY_SOURCE", "both"),
    ("TLLAMA_MEMORY_COMPARE", "maybe"),
])
def test_invalid_environment_globals_fail_with_the_variable_name(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        load_daemon_config(False, discover=False).backend


def test_empty_environment_uses_conservative_declared_defaults(monkeypatch):
    for name in ("TLLAMA_RUNTIME_EXECUTABLE", "TLLAMA_RUNTIME_VERSION_POLICY",
                 "TLLAMA_MEMORY_SOURCE", "TLLAMA_MEMORY_COMPARE"):
        monkeypatch.setenv(name, " ")
    config = load_daemon_config(False, discover=False).backend
    assert config.runtime_executable == DEFAULT_EXECUTABLE
    assert config.runtime_version_policy == "warn" and config.memory_source == "auto"
    assert config.memory_compare is False


@pytest.mark.parametrize("values", [
    {"version_policy": "ignore"}, {"memory_source": "both"}, {"memory_compare": 1},
])
def test_launch_policy_is_also_validated_for_direct_callers(values):
    with pytest.raises(ValueError):
        WorkerLaunchSpec(model=Path("/model"), device="none", **values)


@pytest.mark.parametrize("values", [
    {"runtime_executable": "relative"}, {"runtime_version_policy": "ignore"},
    {"memory_source": "both"}, {"memory_compare": "yes"},
])
def test_backend_policy_is_also_validated_for_direct_callers(values):
    with pytest.raises(ConfigError):
        BackendConfig(**values)
