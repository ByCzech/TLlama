"""Verify diagnostic profiles and saved evidence with a real loopback subprocess."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from test_native_memory_api import payload


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "memory-smoke.py"
module_spec = importlib.util.spec_from_file_location("memory_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(smoke)


@pytest.mark.parametrize("profile,layers,load_mode", [
    ("cpu", "0", "mmap"), ("partial", "20", "mmap"),
    ("vision", "all", "mmap"), ("no-mmap", "all", "none"),
])
def test_profiles_are_explicit_and_dry_run_writes_nothing(tmp_path, capsys, monkeypatch, profile, layers, load_mode):
    monkeypatch.setenv("TLLAMA_MEMORY_SOURCE", "log")
    monkeypatch.setenv("TLLAMA_MEMORY_COMPARE", "false")
    monkeypatch.setenv("TLLAMA_RUNTIME_DEVICE", "unrelated-device")
    output = tmp_path / "output"
    args = ["--profile", profile, "--model", str(tmp_path / "model.gguf"),
            "--output", str(output), "--dry-run"]
    if profile == "vision":
        args += ["--mmproj", str(tmp_path / "projector.gguf")]
    assert smoke.main(args) == 0
    command = json.loads(capsys.readouterr().out)["argv"]
    assert command[command.index("--n-gpu-layers") + 1] == layers
    assert command[command.index("--load-mode") + 1] == load_mode
    assert command[command.index("--device") + 1] == ("none" if profile == "cpu" else "Vulkan0")
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--cache-type-k") + 1] == ("q8_0" if profile == "vision" else "q4_0")
    assert ("--no-kv-offload" in command) == (profile == "cpu")
    assert ("--no-op-offload" in command) == (profile == "cpu")
    assert not output.exists()


@pytest.mark.parametrize("profile,status", [
    ("cpu", 200), ("partial", 200), ("vision", 200), ("no-mmap", 200), ("cpu", 404),
])
def test_saved_evidence_inference_and_cleanup(tmp_path, payload, profile, status):
    root = tmp_path / "stub"
    root.mkdir()
    executable = tmp_path / "native-stub"
    executable.write_text(f"#!{sys.executable}\n" + Path(__file__).with_name("native_worker_stub.py").read_text())
    executable.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.touch()
    payload["runtime_fingerprint"] = "b11552-d1570abd3d"
    for group in payload["placement"]["layer_groups"]:
        if profile == "cpu":
            group.update(location="cpu", device_ids=["cpu"])
        elif profile != "partial" and group["component"] != "input_weights":
            group.update(location="accelerator", device_ids=["gpu0"])
    if profile == "cpu":
        for buffer in payload["buffers"]:
            buffer.update(memory_kind="host", device_ids=["cpu"])
    if profile == "no-mmap":
        for buffer in payload["buffers"]:
            buffer["backing"] = "allocated"
    (root / "memory-report").write_text(json.dumps(payload))
    (root / "memory-status").write_text(str(status))
    output = tmp_path / "results"
    command = [sys.executable, str(SCRIPT), "--profile", profile, "--model", str(model),
               "--executable", str(executable), "--output", str(output)]
    if profile == "vision":
        projector = tmp_path / "projector.gguf"
        projector.touch()
        command += ["--mmproj", str(projector)]
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1] / "src"),
                   "STUB_ROOT": str(root), "STUB_BUILD": "b11552-d1570abd3d",
                   "TLLAMA_MEMORY_SOURCE": "log", "TLLAMA_MEMORY_COMPARE": "false"}
    run = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=20)
    assert run.returncode == (0 if status == 200 else 1), run.stdout + run.stderr
    raw = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
    concise = [json.loads(line) for line in (output / "summary.jsonl").read_text().splitlines()]
    assert run.stdout == (output / "summary.jsonl").read_text()
    measurements = [record for record in raw if "telemetry" in record]
    assert len(measurements) == (4 if profile == "vision" else 2)
    assert all(record["telemetry"]["mode"] == "auto" for record in measurements)
    assert all("comparison" in record["telemetry"] for record in measurements)
    assert not any("telemetry" in record for record in concise)
    assert next(record for record in raw if record["stage"] == "chat")["response"]["choices"]
    assert raw[-2]["stage"] == "closed" and raw[-2]["directory_removed"] is True
    assert raw[-2]["returncode"] in (0, -signal.SIGTERM)
    assert raw[-1]["ok"] is (status == 200)
    if status == 200:
        summary = measurements[0]["summary"]
        assert all(summary["checks"].values())
        # The shared compute leaf has two owners but contributes its capacity once.
        assert sum(summary["backing_bytes"].values()) == summary["totals"]["allocated_bytes"] == 1350
        assert summary["source"] == "native_memory_api"
    else:
        assert measurements[0]["summary"]["checks"] == {"native_inventory_known": False}
        assert measurements[0]["summary"]["fallback_reason"] == "endpoint_not_supported"
    assert (output / "worker.log").is_file()
    # A repeat must preserve the existing results without launching another child.
    before = (root / "launch").read_text()
    repeat = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=10)
    assert repeat.returncode == 1 and (root / "launch").read_text() == before
