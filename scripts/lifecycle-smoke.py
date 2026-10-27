#!/usr/bin/env python3
"""Capture an exact native build's supervisor, cancellation and sleep lifecycle.

Uses physical GGUF paths and NativeResidency, with no registered-model edits.
Native idle sleep is enabled only for the final diagnostic generation.
"""

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
import logging
import math
import runpy
import signal
from pathlib import Path

from tllama.daemon_config import load_daemon_config
from tllama.native_config import ResolvedWorkerConfig
from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_summary, public_memory
from tllama.residency import ModelDefinition, NativeResidency
from tllama.worker import NativeWorker, WorkerError, WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec


@dataclass(frozen=True)
class DiagnosticLaunchSpec(WorkerLaunchSpec):
    # Keep this outside production launch/config: ordinary workers never sleep.
    sleep_idle_seconds: int = -1

    def __post_init__(self):
        super().__post_init__()
        if type(self.sleep_idle_seconds) is not int or not (self.sleep_idle_seconds == -1 or 1 <= self.sleep_idle_seconds <= 60):
            raise ValueError("Diagnostic sleep interval must be -1 or an integer in 1..60")

    def command(self, *args, **kwargs):
        command = list(super().command(*args, **kwargs))
        command[command.index("--sleep-idle-seconds") + 1] = str(self.sleep_idle_seconds)
        return tuple(command)


def launch_spec(args):
    config = load_daemon_config(False, discover=False).backend
    if args.ctx <= 0:
        raise ValueError("--ctx must be positive")
    if not args.expected_fingerprint.strip():
        raise ValueError("--expected-fingerprint must be nonempty")
    for name in ("transition_timeout", "load_timeout", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.transition_timeout <= args.sleep_idle_seconds:
        raise ValueError("--transition-timeout must exceed --sleep-idle-seconds")
    # All CLI inputs are explicit. Do not inherit daemon runtime overrides or
    # an allowlisted identity in place of this capture's requested fingerprint.
    return DiagnosticLaunchSpec(
        model=args.model.expanduser().resolve(), device=args.device,
        executable=args.executable.expanduser().resolve() if args.executable else config.runtime_executable,
        runtime_fingerprint=args.expected_fingerprint, expected_runtime_fingerprint=args.expected_fingerprint, version_policy="strict",
        transport=args.transport, n_ctx=args.ctx, n_gpu_layers=0 if args.device == "none" else -1,
        type_k=args.type_k, type_v=args.type_v, flash_attn=args.flash_attn == "on",
        memory_source="native", memory_compare=True,
        memory_tensor_diagnostics=args.memory_tensor_diagnostics,
        runtime_options={"use_mmap": True, "use_mlock": False, "n_batch": 512, "n_ubatch": 512,
                         "op_offload": args.device != "none", "offload_kqv": args.device != "none"},
        load_timeout=args.load_timeout, request_timeout=args.request_timeout,
    )


class Capture:
    def __init__(self, output):
        self.output = output
        self.checks = []

    def write(self, filename, stage, **values):
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, **values}, ensure_ascii=False) + "\n")

    def check(self, stage, condition, **details):
        self.checks.append({"stage": stage, "ok": bool(condition)})
        self.write("summary.jsonl", stage, ok=bool(condition), **details)
        if not condition:
            raise WorkerError(f"Lifecycle check failed: {stage}")


def fresh_telemetry(telemetry):
    """A newly constructed worker has no inherited native report or totals."""
    return (telemetry.get("measurement") is None
            and (telemetry.get("comparison") or {}).get("native") is None
            and telemetry.get("native", {}).get("support") == "unknown")


def evidence_checks(report, expected, state, diagnostics, *, device=None):
    """Validate each observation separately; native IDs are snapshot-local."""
    snapshot = report.snapshot()
    weights = report.weight_totals()
    locality = snapshot.get("allocation_locality")
    storage = snapshot.get("weight_storage") or {}
    tensor = storage.get("tensor_diagnostics")
    checks = {
        "fingerprint": snapshot["runtime_fingerprint"] == expected,
        "model_state": snapshot["model_state"] == state,
        "inventory_known": snapshot["status"] == "known",
        "extensions_valid": not snapshot.get("extension_errors"),
        "weights_known": weights is not None and weights.status == "known"
            and weights.payload_bytes is not None and weights.cpu_bytes is not None
            and weights.accelerator_bytes is not None,
        "locality_present": locality is not None,
    }
    if diagnostics:
        checks["diagnostics_present"] = tensor is not None
    if state == "sleeping":
        checks["empty_after_unload"] = (
            snapshot["snapshot_stage"] == "after_unload" and not snapshot["buffers"]
            and not snapshot["owners"] and not snapshot["placement"]["layer_groups"]
            and report.totals().allocated_bytes == 0 and weights is not None and weights.payload_bytes == 0
            and not storage.get("intervals") and locality is not None
            and not locality["allocations"] and not locality["bindings"]
            and (tensor is None or not tensor["records"]))
    else:
        if device is not None:
            checks["requested_offload"] = (weights is not None and weights.accelerator_bytes is not None
                and (weights.accelerator_bytes == 0 if device == "none" else weights.accelerator_bytes > 0))
        checks["nonempty_weights"] = weights is not None and (weights.payload_bytes or 0) > 0
    return checks


async def observe(worker, capture, stage, *, state="loaded", after=None):
    # Use the existing raw collector rather than introducing another transfer
    # path. Raw and selected snapshots are distinct observations, never joined.
    raw_memory = runpy.run_path(str(Path(__file__).with_name("memory-smoke.py")))["raw_memory"]
    raw = await raw_memory(worker)
    await worker.refresh_memory()
    telemetry = worker.memory_snapshot()
    capture.write("raw.jsonl", stage, pid=worker.pid, raw_native=raw, telemetry=telemetry)
    capture.check(stage + ":http", raw.get("http_status") == 200 and "error" not in raw, raw_status=raw.get("http_status"))
    raw_report = NativeMemoryReport(raw["body"])
    native = (telemetry.get("comparison") or {}).get("native")
    capture.check(stage + ":selected", telemetry.get("source") == "native_memory_api" and native is not None
                  and not native["report"].get("extension_errors"),
                  native=telemetry.get("native"))
    selected = NativeMemoryReport(native["report"])
    for name, report in (("raw", raw_report), ("selected", selected)):
        checks = evidence_checks(report, worker.spec.runtime_fingerprint, state,
                                 worker.spec.memory_tensor_diagnostics, device=worker.spec.device)
        if after is not None:
            checks["sequence_advanced"] = report.snapshot()["snapshot_sequence"] > after
        capture.check(stage + ":" + name, all(checks.values()), checks=checks,
                      extension_errors=report.snapshot().get("extension_errors", []))
    memory = public_memory(telemetry)
    display = ollama_memory_summary(memory)
    capture.check(stage + ":display", display["size"] is not None and display["size_vram"] is not None,
                  display=display)
    return max(raw_report.snapshot()["snapshot_sequence"], selected.snapshot()["snapshot_sequence"])


async def props(worker):
    # GET /props bypasses native sleep, unlike inference/tokenizer tasks.
    async with worker._client.stream("GET", "/props") as response:
        return await worker._json_body(response)


async def wait_sleep(worker, timeout):
    async def wait():
        while True:
            status = await props(worker)
            if status.get("is_sleeping") is True:
                return status
            await asyncio.sleep(0.1)
    return await asyncio.wait_for(wait(), timeout)


def content(chunk):
    return any(choice.get("delta", {}).get(key) for choice in chunk.get("choices", [])
               for key in ("content", "reasoning_content"))


async def first_content(events):
    async for chunk in events:
        if content(chunk):
            if any(choice.get("finish_reason") is not None for choice in chunk.get("choices", [])):
                raise WorkerError("Stream already finished; cancellation was not exercised")
            return chunk
    raise WorkerError("Stream ended before content/reasoning; cancellation was not exercised")


def answer(response):
    return any(choice.get("message", {}).get("content") for choice in response.get("choices", []))


BODY = {"messages": [{"role": "user", "content": "Say hello briefly."}],
        "reasoning_effort": "none", "max_tokens": 32, "temperature": 0}
LONG_BODY = {**BODY, "max_tokens": 2048, "ignore_eos": True}


async def inference(worker, capture):
    rendered = await worker.render_chat(BODY)
    tokens = await worker.tokenize(rendered)
    capture.check("template-tokenize", bool(rendered) and bool(tokens), prompt=rendered, token_count=len(tokens))
    response = await worker.chat(BODY)
    capture.check("chat", answer(response), response=response)
    response = await worker.completion({"prompt": "Hello", "n_predict": 8, "temperature": 0})
    capture.check("completion", bool(response.get("content")) and response.get("stop") is True,
                  response=response)
    async with worker.chat_stream(BODY) as events:
        chunks = [chunk async for chunk in events]
    capture.check("chat-stream", any(content(chunk) for chunk in chunks)
                  and any("usage" in chunk for chunk in chunks), chunks=chunks)
    async with worker.completion_stream({"prompt": "Hello", "n_predict": 8, "temperature": 0}) as events:
        chunks = [chunk async for chunk in events]
    capture.check("completion-stream", bool(chunks) and chunks[-1].get("stop") is True, chunks=chunks)
    async with worker.chat_stream(LONG_BODY) as events:
        first = await asyncio.wait_for(first_content(events), worker.spec.request_timeout)
        capture.write("summary.jsonl", "cancel-after-content", first=first)
    response = await worker.chat(BODY)
    capture.check("after-stream-cancel", answer(response), response=response)


async def exercise(args, spec, capture):
    active = [spec]
    workers = []

    async def resolve(name, num_ctx):
        resolved = ResolvedWorkerConfig(active[0], {}, {}, {}, None, None, None)
        return ModelDefinition(name, resolved)

    def factory(launch):
        worker = NativeWorker(launch)
        workers.append(worker)
        capture.check(f"new-worker-{len(workers)}", fresh_telemetry(worker.memory_snapshot()),
                      telemetry=worker.memory_snapshot())
        return worker

    # A second slot is used only for the tiny invalid replacement; no second
    # valid model is loaded while the original lease is held.
    residency = NativeResidency(resolve, max_loaded_models=2, worker_factory=factory)
    name = "lifecycle-diagnostic"
    try:
        async with residency.lease(name, keep_alive_seconds=None) as initial:
            worker = initial.worker
            capture.write("summary.jsonl", "ready", generation=initial.generation, pid=worker.pid,
                          argv=worker.command, fingerprint=worker.runtime_fingerprint, props=worker.props)
            await observe(worker, capture, "initial-ready")
            await inference(worker, capture)
            await observe(worker, capture, "after-cancellation")
            async with worker.chat_stream(LONG_BODY) as events:
                await asyncio.wait_for(first_content(events), args.request_timeout)
                await residency.unload(name)
                rows = residency.snapshot()
                capture.check("unload-during-stream", len(rows) == 1 and rows[0]["state"] == "draining"
                              and rows[0]["leases"] == 1 and worker.process.returncode is None, residency=rows)
                await observe(worker, capture, "draining-stream")
            # Context exit cancels the transport first, then releases the lease.
        await asyncio.wait_for(residency.wait_closed(initial.generation), args.transition_timeout)
        capture.check("unloaded", not residency.reporting_snapshot() and worker.process.returncode in (0, -signal.SIGTERM)
                      and not worker.runtime_directory.exists(), returncode=worker.process.returncode)

        async with residency.lease(name, keep_alive_seconds=None) as reloaded:
            capture.check("new-generation", reloaded.generation > initial.generation
                          and reloaded.worker is not initial.worker, generation=reloaded.generation)
            await observe(reloaded.worker, capture, "reloaded")
            invalid = capture.output / "invalid-model.gguf"
            invalid.write_bytes(b"Intentionally invalid GGUF for lifecycle qualification.\n")
            active[0] = replace(spec, model=invalid)
            failed = None
            try:
                await residency.preload(name, keep_alive_seconds=None)
            except WorkerStartupError as exc:
                failed = exc
            finally:
                active[0] = spec
            broken = workers[-1]
            capture.check("failed-replacement", failed is not None and broken is not reloaded.worker
                          and broken.process is not None and broken.process.returncode is not None and broken.process.returncode > 0
                          and not broken.runtime_directory.exists(), error=str(failed),
                          returncode=broken.process.returncode if broken.process else None)
            rows = residency.reporting_snapshot()
            capture.check("failed-generation-absent", len(rows) == 1 and rows[0]["generation"] == reloaded.generation
                          and rows[0]["state"] == "draining", residency=rows)
            response = await reloaded.worker.chat(BODY)
            capture.check("old-lease-survives", answer(response), response=response)
            await observe(reloaded.worker, capture, "old-after-failure")
        await asyncio.wait_for(residency.wait_closed(reloaded.generation), args.transition_timeout)
        capture.check("replacement-retired", not residency.reporting_snapshot()
                      and reloaded.worker.process.returncode in (0, -signal.SIGTERM))

        active[0] = replace(spec, sleep_idle_seconds=args.sleep_idle_seconds)
        async with residency.lease(name, keep_alive_seconds=None) as recovered:
            worker = recovered.worker
            capture.check("recovered-generation", recovered.generation > reloaded.generation
                          and worker is not reloaded.worker, generation=recovered.generation,
                          argv=worker.command, pid=worker.pid)
            await worker.chat(BODY)  # Reset idle timer before capturing loaded evidence.
            before = await observe(worker, capture, "recovered")
            status = await wait_sleep(worker, args.transition_timeout)
            capture.check("native-sleep", status.get("is_sleeping") is True and worker.process.returncode is None,
                          props=status, pid=worker.pid)
            sleeping = await observe(worker, capture, "sleeping", state="sleeping", after=before)
            # Memory/props reads must not wake the model or retain old tensors.
            status = await props(worker)
            capture.check("sleep-read-is-passive", status.get("is_sleeping") is True, props=status)
            await observe(worker, capture, "sleeping-again", state="sleeping")
            pid = worker.pid
            response = await worker.chat(BODY)  # Native lazy wake; same process/generation.
            status = await props(worker)
            capture.check("native-wake", answer(response) and status.get("is_sleeping") is False
                          and worker.pid == pid, props=status, response=response)
            await observe(worker, capture, "awakened", after=sleeping)
            await residency.unload(name)
        await asyncio.wait_for(residency.wait_closed(recovered.generation), args.transition_timeout)
        capture.check("final-unload", not residency.reporting_snapshot()
                      and worker.process.returncode in (0, -signal.SIGTERM))
    finally:
        try:
            await residency.shutdown()
        finally:
            # Include every load/failure's redacted logs, even after an early error.
            for index, worker in enumerate(workers, 1):
                (capture.output / f"worker-{index}.load.log").write_text(worker.load_log, encoding="utf-8")
            capture.check("cleanup", not residency.reporting_snapshot() and all(
                (w.process is None or w.process.returncode is not None)
                and (w.runtime_directory is None or not w.runtime_directory.exists()) for w in workers),
                workers=[{"pid": w.pid, "state": w.state, "argv": w.command,
                          "fingerprint": w.runtime_fingerprint, "load_log_complete": w.load_log_complete,
                          "returncode": w.process.returncode if w.process else None} for w in workers])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--device", required=True, help="Exact native selector: CUDA0, ROCm0, Vulkan0 or none")
    result.add_argument("--expected-fingerprint", required=True)
    result.add_argument("--executable", type=Path)
    result.add_argument("--output", type=Path, required=True, help="New directory; retained on failure")
    result.add_argument("--ctx", type=int, default=4096)
    result.add_argument("--type-k", default="q4_0")
    result.add_argument("--type-v", default="q4_0")
    result.add_argument("--flash-attn", choices=("on", "off"), default="on")
    result.add_argument("--memory-tensor-diagnostics", type=int, default=4096, choices=range(0, 4097), metavar="0..4096")
    result.add_argument("--sleep-idle-seconds", type=int, choices=range(1, 61), default=5, metavar="1..60")
    result.add_argument("--transition-timeout", type=float, default=60)
    result.add_argument("--load-timeout", type=float, default=600)
    result.add_argument("--request-timeout", type=float, default=120)
    result.add_argument("--transport", choices=("uds", "tcp"), default="uds")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        spec = launch_spec(args)
        # Validate the final sleep launch even during dry-run.
        sleeping = replace(spec, sleep_idle_seconds=args.sleep_idle_seconds)
        if args.dry_run:
            key = Path("/diagnostic/api-key")
            print(json.dumps({"initial_argv": spec.command(key), "sleep_argv": sleeping.command(key)}, ensure_ascii=False))
            return 0
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    capture = Capture(output)
    logging.basicConfig(level=logging.DEBUG, filename=output / "worker.log",
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    ok, error = False, None
    try:
        asyncio.run(exercise(args, spec, capture))
        ok = all(check["ok"] for check in capture.checks)
    except KeyboardInterrupt:
        error = "Interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {"ok": ok, "error": error, "expected_fingerprint": args.expected_fingerprint,
              "checks": capture.checks, "output": str(output)}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
