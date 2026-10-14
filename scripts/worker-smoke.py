#!/usr/bin/env python3
"""Exercise the standalone adapter against an installed native runtime."""

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

from tllama.worker import NativeWorker, WorkerError, WorkerExited
from tllama.worker_config import DEFAULT_EXECUTABLE, WorkerLaunchSpec


def report(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


async def exercise(args):
    environment = {}
    for entry in args.env:
        name, separator, value = entry.partition("=")
        if not separator:
            raise ValueError("--env requires NAME=VALUE")
        environment[name] = value
    spec = WorkerLaunchSpec(
        model=args.model.expanduser().resolve(), device=args.device,
        executable=args.executable.expanduser().resolve(),
        mmproj=args.mmproj.expanduser().resolve() if args.mmproj else None,
        n_ctx=args.ctx, n_gpu_layers=args.gpu_layers, type_k=args.type_k, type_v=args.type_v,
        flash_attn=None if args.flash_attn is None else args.flash_attn == "on",
        environment=environment, load_timeout=args.load_timeout, port=args.port,
        request_timeout=args.request_timeout,
    )
    worker = NativeWorker(spec)
    started = time.monotonic()
    try:
        if args.cancel_start_after is not None:
            if args.cancel_start_after <= 0:
                raise ValueError("--cancel-start-after must be positive")
            loading = asyncio.create_task(worker.start())
            try:
                await asyncio.sleep(args.cancel_start_after)
                if loading.done():
                    await loading
                    raise WorkerError("Worker was already ready; choose a shorter startup cancellation interval")
                loading.cancel()
                try:
                    await loading
                except asyncio.CancelledError:
                    report("startup-cancelled", pid=worker.pid)
            finally:
                if not loading.done():
                    loading.cancel()
                    await asyncio.gather(loading, return_exceptions=True)
            return

        await worker.start()
        report("ready", pid=worker.pid, address=worker.base_url,
               load_seconds=round(time.monotonic() - started, 3),
               device=spec.device, argv=worker.command,
               fingerprint=worker.props["build_info"],
               context=worker.props["default_generation_settings"]["n_ctx"],
               modalities=worker.props.get("modalities"))
        body = {"messages": [{"role": "user", "content": "Napiš krátký pozdrav česky."}],
                "reasoning_effort": "none", "max_tokens": 64, "temperature": 0}
        result = await worker.chat(body)
        report("chat", response=result)
        text, reasoning, final = [], [], None
        async with worker.chat_stream(body) as events:
            async for chunk in events:
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
                if "usage" in chunk:
                    final = chunk
        if final is None:
            raise WorkerError("Chat stream did not include requested usage")
        report("chat-stream", content="".join(text), reasoning="".join(reasoning), final=final)
        report("completion", response=await worker.completion({
            "prompt": "Hello", "n_predict": 8, "temperature": 0,
        }))
        async with worker.completion_stream({"prompt": "Hello", "n_predict": 8, "temperature": 0}) as events:
            chunks = [chunk async for chunk in events]
        report("completion-stream", response=chunks)

        long_body = {**body, "max_tokens": 2048, "ignore_eos": True}
        async with worker.chat_stream(long_body) as events:
            first = await anext(events)
            report("stream-cancel", first=first)
        await asyncio.wait_for(worker.chat(body), timeout=30)
        report("after-stream-cancel", ok=True)

        pending = asyncio.create_task(worker.chat(long_body))
        try:
            await asyncio.sleep(1)
            if pending.done():
                await pending
                raise WorkerError("Long request completed before cancellation; this run did not exercise non-stream cancel")
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        await asyncio.wait_for(worker.chat(body), timeout=30)
        report("after-non-stream-cancel", ok=True)
        if args.crash:
            os.kill(worker.pid, signal.SIGKILL)
            await asyncio.wait_for(worker.wait_for_exit(), timeout=spec.kill_timeout)
            try:
                await worker.chat(body)
            except WorkerExited as exc:
                report("crash-detected", returncode=exc.returncode)
            else:
                raise WorkerError("Request unexpectedly succeeded after worker crash")
    finally:
        await worker.close()
        report("closed", pid=worker.pid, returncode=worker.process.returncode if worker.process else None,
               directory_removed=worker.runtime_directory is None or not worker.runtime_directory.exists(),
               state=worker.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="none", help="Native selector, e.g. none, Vulkan0, ROCm0, CUDA0")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int)
    parser.add_argument("--type-k")
    parser.add_argument("--type-v")
    parser.add_argument("--flash-attn", choices=("on", "off"))
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--log", type=Path, help="Capture redacted native debug logs to this file")
    parser.add_argument("--crash", action="store_true", help="Kill only the owned worker after inference tests")
    parser.add_argument("--cancel-start-after", type=float, metavar="SECONDS", help="Test cancelled startup instead of inference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.log else logging.WARNING, filename=args.log,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Keep HTTP libraries from writing authentication headers to debug logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        asyncio.run(exercise(args))
    except (WorkerError, ValueError, asyncio.TimeoutError) as exc:
        report("error", error=str(exc))
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
