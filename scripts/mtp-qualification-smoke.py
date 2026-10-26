#!/usr/bin/env python3
"""Qualify an exact native build's embedded MTP with independent off/on workers.

Physical GGUF, no registered-model edits. Raw evidence is retained on failure.
This is a native diagnostic, not public API or general model quality certification.
"""

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import runpy
import signal

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_summary, public_memory
from tllama.worker import NativeWorker


HERE = Path(__file__).resolve().parent
LIFECYCLE = runpy.run_path(str(HERE / "lifecycle-smoke.py"))
MTP = runpy.run_path(str(HERE / "mtp-smoke.py"))
RAW_MEMORY = runpy.run_path(str(HERE / "memory-smoke.py"))["raw_memory"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--device", required=True)
    result.add_argument("--expected-fingerprint", required=True)
    result.add_argument("--executable", type=Path)
    result.add_argument("--output", type=Path, required=True, help="New capture directory; retained on failure")
    result.add_argument("--ctx", type=int, default=4096)
    result.add_argument("--type-k", default="q4_0")
    result.add_argument("--type-v", default="q4_0")
    result.add_argument("--flash-attn", choices=("on", "off"), default="on")
    result.add_argument("--memory-tensor-diagnostics", type=int, choices=range(0, 4097), default=4096, metavar="0..4096")
    result.add_argument("--load-timeout", type=float, default=600)
    result.add_argument("--request-timeout", type=float, default=600)
    result.add_argument("--transport", choices=("uds", "tcp"), default="uds")
    result.add_argument("--dry-run", action="store_true")
    # Shared launch validation; no sleep transitions in this diagnostic.
    result.set_defaults(sleep_idle_seconds=5, transition_timeout=60)
    result.add_argument("--draft", type=int, choices=range(1, 33), default=3, metavar="1..32")
    result.add_argument("--predict", type=int, choices=range(32, 513), default=128, metavar="32..512")
    result.add_argument("--no-mmap", action="store_true", help="Explicit diagnostic workaround; recorded in both launches")
    return result


def launch_specs(args):
    base = LIFECYCLE["launch_spec"](args)
    return {profile: replace(base, runtime_options={**base.runtime_options,
                "use_mmap": not args.no_mmap, "spec_type": kind, "draft_n_max": args.draft})
            for profile, kind in (("off", "none"), ("mtp", "draft-mtp"))}


def file_identity(path):
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def weight_evidence(report, active):
    """Independent union of snapshot-local intervals, never a sum of role totals."""
    snapshot = report.snapshot()
    storage = snapshot.get("weight_storage") or {}
    intervals = storage.get("intervals", [])
    main = report.weight_totals(roles={"main"})
    mtp = report.weight_totals(roles={"mtp"})
    union = report.weight_totals(roles={"main", "mtp"})
    total = report.weight_totals()
    selected_owners = {o["id"] for o in snapshot["owners"]
                       if o["kind"] == "model" and {"main", "mtp"}.intersection(o["roles"])}
    independent = sum(item["size_bytes"] for item in intervals
                      if any(rel["owner_id"] in selected_owners for rel in item["relationships"]))
    heads = report.weight_totals(components={"mtp_layers"})
    shared = [o["id"] for o in snapshot["owners"] if {"main", "mtp"} <= set(o["roles"])]
    checks = {
        "main_payload": main is not None and main.status == "known" and (main.payload_bytes or 0) > 0,
        "union_payload": union is not None and union.status == "known" and union.payload_bytes == independent,
        "all_model_weights_covered": total is not None and union is not None
            and total.payload_bytes == union.payload_bytes,
    }
    if active:
        checks.update(mtp_payload=mtp is not None and mtp.status == "known" and (mtp.payload_bytes or 0) > 0,
                      mtp_head_payload=heads is not None and heads.status == "known" and (heads.payload_bytes or 0) > 0,
                      shared_model_owner=bool(shared))
    return {"checks": checks, "shared_main_mtp_owner_ids": shared,
            "unique_interval_bytes": independent,
            "totals": {name: asdict(value) if value is not None else None
                       for name, value in (("main", main), ("mtp", mtp), ("union", union), ("all", total), ("mtp_layers", heads))},
            "note": "Role subtotals overlap; no cross-profile or cross-snapshot ID join"}


async def observe(worker, capture, stage, active, baseline=None):
    raw = await RAW_MEMORY(worker)
    await worker.refresh_memory()
    telemetry = worker.memory_snapshot()
    capture.write("raw.jsonl", stage, pid=worker.pid, raw_native=raw, telemetry=telemetry)
    capture.check(stage + ":http", raw.get("http_status") == 200 and "error" not in raw,
                  raw_status=raw.get("http_status"))
    native = (telemetry.get("comparison") or {}).get("native")
    capture.check(stage + ":selected", telemetry.get("source") == "native_memory_api" and native is not None)
    observed = []
    for name, data in (("raw", raw["body"]), ("selected", native["report"])):
        report = NativeMemoryReport(data)
        checks = LIFECYCLE["evidence_checks"](report, worker.spec.runtime_fingerprint, "loaded",
                                               worker.spec.memory_tensor_diagnostics, device=worker.spec.device)
        evidence = weight_evidence(report, active)
        payloads = {role: totals["payload_bytes"] if totals else None for role, totals in evidence["totals"].items()}
        if baseline is not None:
            checks["payload_unchanged"] = payloads == baseline
        capture.check(stage + ":" + name, all(checks.values()) and all(evidence["checks"].values()),
                      checks=checks, weight_evidence=evidence, extension_errors=report.snapshot().get("extension_errors", []))
        observed.append(payloads)
    capture.check(stage + ":payload-parity", observed[0] == observed[1])
    display = ollama_memory_summary(public_memory(telemetry))
    capture.check(stage + ":display", display["size"] is not None and display["size_vram"] is not None,
                  display=display)
    return observed[0]


def compare(reference, actual):
    return {"text_match": reference["text_sha256"] == actual["text_sha256"],
            "finish_match": reference["finish_reason"] == actual["finish_reason"],
            "token_count_match": reference["metrics"]["predicted_n"] is not None
                and reference["metrics"]["predicted_n"] == actual["metrics"]["predicted_n"]}


def activation(results, active):
    checks = {}
    for stream in (False, True):
        values = [result["metrics"] for (probe, mode), result in results.items() if mode == stream]
        if active:
            checks[str(stream)] = bool(values) and all(
                value["draft_n"] is not None and value["draft_n_accepted"] is not None for value in values)
            checks[str(stream)] &= sum(value["draft_n"] or 0 for value in values) > 0
            checks[str(stream)] &= sum(value["draft_n_accepted"] or 0 for value in values) > 0
        else:
            checks[str(stream)] = bool(values) and all(
                value["draft_n"] in (None, 0) and value["draft_n_accepted"] in (None, 0) for value in values)
    return checks


async def exercise(args, specs, capture):
    profiles = {}
    baseline = None
    for profile, spec in specs.items():
        active = profile == "mtp"
        worker = NativeWorker(spec)
        results = {}

        def record(stage, **values):
            # Native responses and deltas remain available without bloating the summary.
            capture.write("raw.jsonl" if stage in {"native-event", "native-response"} else "summary.jsonl",
                          stage, profile=profile, **values)

        try:
            identity = (file_identity(spec.model), file_identity(spec.executable))
            if baseline is None:
                baseline = identity
            capture.check(profile + ":same-files", identity == baseline, identity=identity)
            capture.check(profile + ":fresh-worker", LIFECYCLE["fresh_telemetry"](worker.memory_snapshot()))
            await worker.start()
            record("ready", pid=worker.pid, argv=worker.command, fingerprint=worker.runtime_fingerprint, props=worker.props)
            payloads = await observe(worker, capture, profile + ":loaded", active)
            control_body = MTP["request"]("Reply exactly READY.", 16)
            control = await MTP["collect"](worker, control_body, False, record, probe="control")
            capture.check(profile + ":control", bool(control["content"].strip()) and control["finish_reason"] == "stop")
            for probe, prompt in MTP["PROMPTS"].items():
                for stream in (False, True):
                    result = await MTP["collect"](worker, MTP["request"](prompt, args.predict), stream, record, probe=probe)
                    results[probe, stream] = result
                    capture.check(f"{profile}:{probe}:{stream}:content", bool(result["content"].strip()))
                checks = compare(results[probe, False], results[probe, True])
                capture.check(profile + ":" + probe + ":stream-parity", all(checks.values()), checks=checks)
            checks = activation(results, active)
            capture.check(profile + ":draft-counters", all(checks.values()), checks=checks,
                          metrics=[{"probe": key[0], "stream": key[1], **value["metrics"]} for key, value in results.items()])
            await observe(worker, capture, profile + ":after-inference", active, baseline=payloads)
            pid = worker.pid
            async with worker.chat_stream(MTP["request"](MTP["CANCEL"], 512)) as events:
                first = await asyncio.wait_for(LIFECYCLE["first_content"](events), args.request_timeout)
                record("cancel-after-content", pid=pid, first=first,
                       native_stop_proof="Review worker.log for actual early native task stop; transport closure alone is insufficient")
            recovered = await MTP["collect"](worker, control_body, False, record, probe="after-cancel")
            checks = compare(control, recovered)
            capture.check(profile + ":cancel-reuse", worker.pid == pid and all(checks.values()), checks=checks, pid=worker.pid)
            await observe(worker, capture, profile + ":after-cancel", active, baseline=payloads)
            capture.check(profile + ":files-unchanged", identity == (file_identity(spec.model), file_identity(spec.executable)))
            profiles[profile] = results
        finally:
            try:
                await worker.close()
            finally:
                (capture.output / (profile + "-load.log")).write_text(worker.load_log, encoding="utf-8")
                capture.check(profile + ":cleanup", worker.state == "closed"
                    and (worker.process is None or worker.process.returncode in (0, -signal.SIGTERM))
                    and (worker.runtime_directory is None or not worker.runtime_directory.exists()),
                    pid=worker.pid, returncode=worker.process.returncode if worker.process else None,
                    directory_removed=worker.runtime_directory is None or not worker.runtime_directory.exists())
    differences = []
    for key, off in profiles["off"].items():
        mtp = profiles["mtp"][key]
        checks = compare(off, mtp)
        record("off-on-comparison", probe=key[0], stream=key[1], checks=checks)
        if not all(checks.values()):
            differences.append({"probe": key[0], "stream": key[1], "checks": checks})
    return differences


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        specs = launch_specs(args)
        if args.dry_run:
            print(json.dumps({name: spec.command(Path("/diagnostic/api-key")) for name, spec in specs.items()}))
            return 0
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    capture = LIFECYCLE["Capture"](output)
    logging.basicConfig(level=logging.DEBUG, filename=output / "worker.log",
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    error, differences = None, []
    try:
        differences = asyncio.run(exercise(args, specs, capture))
    except KeyboardInterrupt:
        error = "Interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    protocol_ok = error is None and bool(capture.checks) and all(item["ok"] for item in capture.checks)
    result = {"ok": protocol_ok and not differences, "protocol_ok": protocol_ok,
              "review_required": not protocol_ok or bool(differences), "off_on_differences": differences,
              "error": error, "checks": capture.checks, "expected_fingerprint": args.expected_fingerprint,
              "mmap": not args.no_mmap, "public_api": "not_exercised", "output": str(output)}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
