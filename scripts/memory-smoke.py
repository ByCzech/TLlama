#!/usr/bin/env python3
"""Collect memory API/log evidence for one explicit 08c hardware profile.

Uses physical GGUF paths and a private worker, independently of model TOML.
See memory-smoke-08c.md for the four initial runs and result interpretation.
"""

import argparse
import asyncio
from collections import defaultdict
import json
import logging
from pathlib import Path
import runpy
import signal

from tllama.config import load_backend_config_from_env
from tllama.native_reporting import ollama_memory_sizes, public_memory
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError
from tllama.worker_config import RUNTIME_FINGERPRINT, WorkerLaunchSpec


def launch_spec(args):
    config = load_backend_config_from_env()
    cpu = args.profile == "cpu"
    vision = args.profile == "vision"
    if vision != (args.mmproj is not None):
        raise ValueError("--mmproj is required only for the vision profile")
    if args.ctx <= 0 or args.gpu_layers <= 0:
        raise ValueError("--ctx and --gpu-layers must be positive")
    if args.device == "none" and not cpu:
        raise ValueError("Accelerator profiles require --device other than none")
    if args.expected_fingerprint is not None and not args.expected_fingerprint.strip():
        raise ValueError("--expected-fingerprint must be nonempty")
    return WorkerLaunchSpec(
        model=args.model.expanduser().resolve(),
        executable=args.executable.expanduser().resolve() if args.executable else config.runtime_executable,
        version_policy=config.runtime_version_policy,
        runtime_fingerprint=args.expected_fingerprint or RUNTIME_FINGERPRINT,
        device="none" if cpu else args.device,
        n_gpu_layers=0 if cpu else args.gpu_layers if args.profile == "partial" else -1,
        n_ctx=args.ctx, mmproj=args.mmproj.expanduser().resolve() if vision else None,
        type_k=args.type_k or ("q8_0" if vision else "q4_0"), type_v=args.type_v,
        flash_attn=args.flash_attn == "on", memory_source="auto", memory_compare=True,
        runtime_options={"use_mmap": args.profile != "no-mmap", "use_mlock": False,
                         "n_batch": 512, "n_ubatch": 512,
                         "op_offload": not cpu, "offload_kqv": not cpu},
        load_timeout=args.load_timeout, request_timeout=args.request_timeout,
    )


async def raw_memory(worker):
    """Read the authenticated endpoint before the production V1 field filter.

    This is independent evidence, not a replacement for selected telemetry.
    Serialize with its HTTP poller, never with the inference slot. IDs from
    this response must not be joined with another telemetry snapshot.
    """
    result = {"http_status": None}

    async def read():
        async with worker._memory_lock:
            if worker._memory_client is None:
                raise WorkerError("Native memory client is unavailable")
            async with worker._memory_client.stream("GET", "/tllama/memory") as response:
                result["http_status"] = response.status_code
                try:
                    result["body"] = await worker._json_body(response)
                except WorkerRequestError as exc:
                    result.update(body=exc.payload, error=str(exc)[:2048])
    try:
        await asyncio.wait_for(read(), timeout=5)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:2048]
    return result


def locality_summary(raw, fingerprint, profile):
    """Inspect extension evidence without adding allocation spans to totals.

    This diagnostic checks presence and snapshot-local references, not the
    complete future production adapter's validation/public display policy.
    Unknown fields and nullable ranges remain untouched in raw_native.body.
    """
    report = raw.get("body") or {}
    extension = report.get("allocation_locality")
    checks = {"raw_api_ready": raw.get("http_status") == 200 and "error" not in raw,
              "raw_schema_v1": type(report.get("schema_version")) is int and report["schema_version"] == 1,
              "raw_fingerprint_matches": report.get("runtime_fingerprint") == fingerprint,
              "locality_extension_present": isinstance(extension, dict)}
    result = {"checks": checks, "snapshot": {key:report.get(key) for key in
              ("snapshot_sequence", "snapshot_stage", "snapshot_age_ms")}}
    if not isinstance(extension, dict):
        return result
    allocations = extension.get("allocations")
    bindings = extension.get("bindings")
    if (not isinstance(allocations, list) or not isinstance(bindings, list)
            or any(not isinstance(value, dict) for value in allocations + bindings)):
        checks["locality_relationship_arrays"] = False
        return result
    allocation_ids = [a.get("id") for a in allocations]
    buffers = {b["id"] for b in report.get("buffers", [])}
    devices = {d["id"]:d for d in report.get("devices", [])}
    checks["locality_scope"] = extension.get("scope") == "backend_buffer_allocations"
    checks["physical_aliasing_unknown"] = extension.get("physical_aliasing") == "unknown"
    # Unsupported CPU/CUDA/HIP/SYCL exporters are legitimate observations;
    # their independent coverage must not invalidate buffer accounting.
    checks["locality_coverage_reported"] = extension.get("status") in {"known", "partial", "unavailable"}
    checks["allocation_ids"] = (all(isinstance(value, str) and value for value in allocation_ids)
                                and len(set(allocation_ids)) == len(allocation_ids))
    checks["allocation_references"] = all(set(a.get("device_ids", [])) <= devices.keys() for a in allocations)
    checks["binding_references"] = all(b.get("buffer_id") in buffers
        and b.get("allocation_id") in allocation_ids for b in bindings)
    assignments = [{key:g.get(key) for key in
                   ("owner_id", "component", "layer_indices", "location", "device_ids", "assignment")}
                   for g in report.get("placement", {}).get("layer_groups", [])]
    checks["assignment_inventory_present"] = bool(assignments) and all(
        isinstance(g["assignment"], dict) for g in assignments)
    checks["assignment_references"] = all(set(g["assignment"].get("device_ids", [])) <= devices.keys()
        for g in assignments if isinstance(g["assignment"], dict))
    if profile == "no-mmap":
        inputs = [g for g in assignments if g["component"] == "input_weights"]
        checks["input_cpu_assignment"] = bool(inputs) and all(
            g["location"] == "cpu" and isinstance(g["assignment"], dict)
            and g["assignment"].get("status") == "known" and g["assignment"].get("device_ids")
            and all(devices.get(device, {}).get("type") == "cpu" for device in g["assignment"]["device_ids"])
            for g in inputs)
    result.update(status=extension.get("status"), missing_components=extension.get("missing_components"),
                  allocation_count=len(allocations), binding_count=len(bindings), assignments=assignments)
    return result


def summarize(telemetry, profile):
    memory = public_memory(telemetry)
    size, vram = ollama_memory_sizes(memory)
    native = (telemetry.get("comparison") or {}).get("native")
    log = ((telemetry.get("comparison") or {}).get("log") or {}).get("report") or {}
    result = {"source": memory["source"], "status": memory["status"],
              "native": telemetry["native"], "fallback_reason": telemetry["fallback_reason"],
              "display": {"size": size, "size_vram": vram,
                          "excluded_host_bytes": memory["allocated_bytes"] - size
                          if memory["allocated_bytes"] is not None and size is not None else None},
              "log": {key: log.get(key) for key in
                      ("status", "reason", "allocated_bytes", "host_bytes", "device_bytes")},
              "checks": {"native_inventory_known": False}}
    if native is None:
        return result
    report, totals = native["report"], native["totals"]
    groups = report["placement"]["layer_groups"]
    buffers = report["buffers"]
    weights = [b for b in buffers if any(o["component"] == "model_weights" for o in b["owners"])]
    projector_ids = {o["id"] for o in report["owners"] if o["kind"] == "projector"}
    grouped = defaultdict(int)
    for buffer in buffers:
        if buffer["size_bytes"] is not None:
            grouped[buffer["memory_kind"] + "/" + buffer["backing"]] += buffer["size_bytes"]
    checks = result["checks"]
    checks["native_inventory_known"] = report["status"] == "known"
    checks["placement_known"] = report["placement"]["status"] == "known"
    checks["totals_reconcile"] = (totals["allocated_bytes"] is not None
        and totals["host_bytes"] is not None and totals["device_bytes"] is not None
        and totals["allocated_bytes"] == totals["host_bytes"] + totals["device_bytes"])
    if profile == "cpu":
        checks["cpu_only"] = (totals["device_bytes"] == 0 and bool(groups)
                              and all(g["location"] == "cpu" for g in groups))
    if profile == "partial":
        noninput = [g for g in groups if g["component"] != "input_weights"]
        checks["cpu_layers"] = any(g["location"] in ("cpu", "mixed") for g in noninput)
        checks["accelerator_layers"] = any(g["location"] in ("accelerator", "mixed") for g in noninput)
    if profile in ("vision", "no-mmap"):
        noninput = [g for g in groups if g["component"] != "input_weights"]
        checks["accelerator_weights"] = bool(noninput) and all(g["location"] == "accelerator" for g in noninput)
    if profile == "vision":
        checks["projector_weights"] = any(o["owner_id"] in projector_ids
            and o["component"] == "projector_weights" and b["size_bytes"] is not None
            for b in buffers for o in b["owners"])
    if profile == "no-mmap":
        checks["allocated_weights"] = bool(weights) and all(b["backing"] == "allocated" for b in weights)
    else:
        checks["mapped_weights"] = any(b["backing"] == "mmap" for b in weights)
    result.update(totals=totals, backing_bytes=dict(grouped), placement=report["placement"],
                  missing_components=report["missing_components"],
                  snapshot={key: report[key] for key in
                            ("snapshot_sequence", "snapshot_stage", "snapshot_age_ms")},
                  projector_buffers=[b for b in buffers if any(o["owner_id"] in projector_ids for o in b["owners"])],
                  native_minus_log_bytes={key: totals[key] - log[key]
                      if totals[key] is not None and log.get(key) is not None else None
                      for key in ("allocated_bytes", "host_bytes", "device_bytes")})
    return result


async def exercise(args, spec, emit):
    worker = NativeWorker(spec)
    checks = []

    async def capture(stage):
        raw = await raw_memory(worker)
        await worker.refresh_memory()
        telemetry = worker.memory_snapshot()
        summary = summarize(telemetry, args.profile)
        if args.locality:
            try:
                evidence = locality_summary(raw, worker.runtime_fingerprint, args.profile)
            except Exception as exc:
                # Even malformed extension data must survive in raw evidence.
                evidence = {"checks": {"locality_shape": False},
                            "error": f"{type(exc).__name__}: {exc}"[:2048]}
            summary["locality"] = evidence
            summary["checks"].update(evidence["checks"])
            # Validate the selected observation on its own; its IDs/sequence may
            # differ from the independently read raw response above.
            selected = (telemetry.get("measurement") or {}).get("report") or {}
            summary["checks"]["locality_adapter_valid"] = (
                telemetry["source"] == "native_memory_api" and not selected.get("extension_errors")
                and "allocation_locality" in selected
                and all("assignment" in group for group in selected["placement"]["layer_groups"]))
        checks.append(all(summary["checks"].values()))
        emit(stage, raw_native=raw, telemetry=telemetry, summary=summary)

    try:
        await worker.start()
        emit("ready", pid=worker.pid, fingerprint=worker.runtime_fingerprint, argv=worker.command,
             context=worker.props["default_generation_settings"]["n_ctx"],
             modalities=worker.props.get("modalities"))
        if args.expected_fingerprint is not None and worker.runtime_fingerprint != args.expected_fingerprint:
            raise WorkerError(f"Expected runtime {args.expected_fingerprint}, got {worker.runtime_fingerprint}")
        await capture("memory-ready")
        response = await worker.chat({"messages": [{"role": "user", "content": "Say hello briefly."}],
                                     "reasoning_effort": "none", "max_tokens": 32, "temperature": 0})
        emit("chat", response=response)
        if not any(c.get("message", {}).get("content") for c in response.get("choices", [])):
            raise WorkerError("Text request produced no answer content")
        await capture("memory-after-text")
        if args.profile == "vision":
            vision = runpy.run_path(str(Path(__file__).with_name("vision-smoke.py")))
            for index in range(2):
                body = vision["payload"](spec.alias, "/v1/chat/completions", vision["color_png"](bool(index)))
                response = await worker.chat(body)
                emit(f"image-{index}", response=response)
                vision["check_colors"](response["choices"][0]["message"]["content"], bool(index))
                await capture(f"memory-after-image-{index}")
    finally:
        was_running = worker.process is not None and worker.process.returncode is None
        await worker.close()
        removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
        returncode = worker.process.returncode if worker.process else None
        emit("closed", returncode=returncode, directory_removed=removed)
    clean_exit = returncode == 0 or was_running and returncode == -signal.SIGTERM
    return all(checks) and clean_exit and removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "partial", "vision", "no-mmap"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="Physical GGUF path, not a registered model name")
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--device", default="Vulkan0")
    parser.add_argument("--gpu-layers", type=int, default=20, help="Partial profile only; choose fewer than all model layers")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--type-k")
    parser.add_argument("--type-v", default="q4_0")
    parser.add_argument("--flash-attn", choices=("on", "off"), default="on")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--expected-fingerprint", help="Pin this diagnostic worker to an exact build; no global policy change")
    parser.add_argument("--locality", action="store_true", help="Check the optional assignment/locality extension in raw endpoint replies")
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Show launch command without starting a worker or writing output")
    args = parser.parse_args(argv)
    try:
        spec = launch_spec(args)
        if args.dry_run:
            print(json.dumps({"profile": args.profile, "argv": spec.command(Path("/dry-run/api-key"))}))
            return 0
        for path in (spec.model, spec.mmproj, spec.executable):
            if path is not None and not path.is_file():
                raise ValueError(f"File does not exist: {path}")
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        logging.basicConfig(level=logging.DEBUG, filename=output / "worker.log",
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        with (output / "raw.jsonl").open("w") as raw, (output / "summary.jsonl").open("w") as concise:
            def emit(stage, **values):
                record = {"stage": stage, "profile": args.profile, **values}
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw.flush()
                record.pop("telemetry", None)
                record.pop("raw_native", None)
                line = json.dumps(record, ensure_ascii=False)
                concise.write(line + "\n")
                concise.flush()
                print(line, flush=True)
            try:
                ok = asyncio.run(exercise(args, spec, emit))
                emit("result", ok=ok)
                return 0 if ok else 1
            except (OSError, ValueError, WorkerError, asyncio.TimeoutError) as exc:
                emit("error", error=str(exc))
                return 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"memory-smoke: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
