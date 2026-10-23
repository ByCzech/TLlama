#!/usr/bin/env python3
"""Exercise ModelManager native leases on one registered model, with one load."""

import argparse
import asyncio
from dataclasses import replace
import json
import logging
from pathlib import Path

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.native_config import build_native_sampling
from tllama.worker import WorkerError


def report(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


async def exercise(args):
    config = load_backend_config_from_env()
    if args.models_dir is not None:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    manager = ModelManager(config, native_launch_options={
        "executable": args.executable.expanduser().resolve() if args.executable else None,
        "load_timeout": args.load_timeout,
        "request_timeout": args.request_timeout,
    })
    worker = None
    try:
        await manager.start()
        generation = await manager.preload_native_model(args.model, num_ctx=args.ctx, keep_alive=-1)
        report("preloaded", generation=generation, residency=manager.native_snapshot())
        async with manager.acquire_native_model(args.model, num_ctx=args.ctx, keep_alive=-1) as lease:
            if lease.generation != generation:
                raise WorkerError("Definition changed between preload and use; repeat with a stable TOML")
            worker = lease.worker
            report("ready", pid=worker.pid, generation=lease.generation, address=worker.address,
                   fingerprint=worker.props["build_info"],
                   context=worker.props["default_generation_settings"]["n_ctx"], argv=worker.command)
            await worker.refresh_memory()
            report("memory-ready", telemetry=worker.memory_snapshot())
            body = {**build_native_sampling({"temperature": 0, "num_predict": 32}, resolved=lease.resolved),
                    "messages": [{"role": "user", "content": "Napiš krátký pozdrav česky."}],
                    "reasoning_effort": "none"}
            report("chat", response=await worker.chat(body))
            await worker.refresh_memory()
            report("memory-after-chat", telemetry=worker.memory_snapshot())
            text, reasoning, final = [], [], None
            unloaded = False
            async with worker.chat_stream(body) as events:
                async for chunk in events:
                    meaningful = False
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or ""
                        text.append(content)
                        reasoning.append(thought)
                        meaningful = meaningful or bool(content or thought)
                    if "usage" in chunk:
                        final = chunk
                    if meaningful and not unloaded:
                        await manager.unload_native_model(args.model)
                        snapshot = manager.native_snapshot()
                        if (worker.process.returncode is not None or len(snapshot) != 1
                                or snapshot[0]["state"] != "draining" or snapshot[0]["leases"] != 1):
                            raise WorkerError("Unload did not preserve the active stream's generation")
                        unloaded = True
                        report("unload-during-stream", pid=worker.pid, residency=snapshot)
                        await worker.refresh_memory()
                        report("memory-during-stream", telemetry=worker.memory_snapshot())
            if not unloaded:
                raise WorkerError("Stream produced no content to exercise unload during output")
            if final is None:
                raise WorkerError("Stream did not include requested native usage")
            report("stream-finished", content="".join(text), reasoning="".join(reasoning), final=final)
            if worker.process.returncode is not None:
                raise WorkerError("Worker exited before the lease was released")
        await asyncio.wait_for(manager.wait_native_closed(generation), timeout=30)
        if worker.process.returncode is None or worker.runtime_directory.exists():
            raise WorkerError("Generation cleanup left a process or private directory behind")
        report("worker-closed", pid=worker.pid, generation=generation,
               returncode=worker.process.returncode, directory_removed=True,
               residency=manager.native_snapshot())
    finally:
        await manager.shutdown()
        report("manager-closed", residency=manager.native_snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Registered TLlama name, not a GGUF path or category prefix")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.log else logging.WARNING, filename=args.log,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        asyncio.run(exercise(args))
    except (OSError, ValueError, WorkerError, asyncio.TimeoutError) as exc:
        report("error", error=str(exc))
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
