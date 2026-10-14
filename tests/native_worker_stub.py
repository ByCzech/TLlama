"""Small TCP subprocess implementing the pinned protocol for lifecycle tests."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODE = os.environ.get("STUB_MODE", "normal")
ROOT = Path(os.environ["STUB_ROOT"])
KEY = Path(arg("--api-key-file")).read_text().strip()
LOCK = asyncio.Lock()
HEALTH_READS = 0
TIMINGS = {"cache_n": 19, "prompt_n": 6, "prompt_ms": 2.5,
           "predicted_n": 10, "predicted_ms": 200.1}
USAGE = {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35,
         "prompt_tokens_details": {"cached_tokens": 19}}


def mark(name, value):
    with (ROOT / name).open("a") as file:
        file.write(json.dumps(value) + "\n")


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode()


async def json_response(writer, value, status=200):
    body = encode(value)
    writer.write(f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    await writer.drain()


async def stream_bytes(writer, value):
    # Split UTF-8 code points, delimiters and JSON across writes.
    for offset in range(0, len(value), 3):
        writer.write(value[offset:offset + 3])
        await writer.drain()
        await asyncio.sleep(0)


def event(value):
    data = json.dumps(value, ensure_ascii=False)
    split = data.find(",")
    if split != -1:
        return ("event: message\r\ndata: " + data[:split + 1] +
                "\r\ndata: " + data[split + 1:] + "\r\n\r\n").encode()
    return ("data: " + data + "\r\n\r\n").encode()


async def handle(reader, writer):
    global HEALTH_READS
    try:
        request = await reader.readline()
        if not request:
            return
        path = request.decode().split()[1]
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body_bytes = await reader.readexactly(int(headers.get("content-length", "0")))
        body = json.loads(body_bytes) if body_bytes else {}
        mark("requests", {"path": path, "body": body, "header_names": sorted(headers)})
        if headers.get("authorization") != "Bearer " + KEY:
            await json_response(writer, {"error": "unauthorized"}, 401)
            return
        if path == "/health":
            HEALTH_READS += 1
            if MODE == "loading" and HEALTH_READS <= 3:
                await json_response(writer, {"error": "Loading model"}, 503)
            else:
                await json_response(writer, {"status": "bad" if MODE == "bad-health" else "ok"})
            return
        if path == "/props":
            await json_response(writer, {
                "build_info": "wrong" if MODE == "wrong-build" else "b11552-ec2797836f",
                "model_alias": "wrong" if MODE == "wrong-alias" else arg("--alias"),
                "total_slots": 4 if MODE == "wrong-slots" else 1,
                "model_path": arg("--model"),
                "default_generation_settings": {"n_ctx": 0 if MODE == "bad-context" else 2048},
                "modalities": {"vision": "--mmproj" in sys.argv},
                "bos_token": "<bos>", "eos_token": "<eos>",
            })
            return
        async with LOCK:
            mode = body.get("stub", "hold" if body.get("ignore_eos") else "normal")
            if mode == "crash":
                os._exit(71)
            if mode == "error":
                await json_response(writer, {"error": {"code": 400, "message": "bad request"}}, 400)
                return
            chat = path == "/v1/chat/completions"
            if mode == "hold" and not body["stream"]:
                mark("started", "non-stream")
                await reader.read()
                mark("cancelled", "non-stream")
                return
            if not body["stream"] or mode == "json-stream":
                if chat:
                    result = {"choices": [{"message": {"content": "Ahoj světe", "reasoning_content": "myšlenka"},
                                           "finish_reason": "stop"}], "usage": USAGE, "timings": TIMINGS}
                else:
                    result = {"content": "Ahoj světe", "stop": True, "stop_type": "eos", "timings": TIMINGS,
                              "tokens_evaluated": 25, "tokens_predicted": 10, "tokens_cached": 35}
                await json_response(writer, result)
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            await stream_bytes(writer, b": ping\r\n\r\n")
            chunk = {"choices": [{"delta": {"content": "Ahoj světe"}, "finish_reason": None}]} if chat else {
                "content": "Ahoj světe", "stop": False}
            await stream_bytes(writer, event(chunk))
            if mode == "hold":
                mark("started", "stream")
                await reader.read()
                mark("cancelled", "stream")
                return
            if mode == "bad-json":
                await stream_bytes(writer, b"data: {broken}\n\n")
                return
            if mode == "error-stream":
                await stream_bytes(writer, event({"error": {"code": 500, "message": "decode failed"}}))
                return
            if mode == "truncated":
                return
            if chat:
                await stream_bytes(writer, event({"choices": [{"delta": {"reasoning_content": "myšlenka"},
                                                               "finish_reason": None}]}))
                await stream_bytes(writer, event({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
                await stream_bytes(writer, event({"choices": [], "usage": USAGE, "timings": TIMINGS}))
                if mode == "finish-without-done":
                    return
                await stream_bytes(writer, b"data: [DONE]\r\n\r\n")
            else:
                await stream_bytes(writer, event({"content": "", "stop": True, "stop_type": "eos", "timings": TIMINGS,
                                                  "tokens_evaluated": 25, "tokens_predicted": 10, "tokens_cached": 35}))
                if mode == "hold-after-stop":
                    mark("started", "raw-terminal")
                    await reader.read()
                    mark("cancelled", "raw-terminal")
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def main():
    mark("launch", {"argv": sys.argv[1:], "cwd": os.getcwd(),
                    "cwd_mode": oct(Path.cwd().stat().st_mode & 0o777),
                    "key_mode": oct(Path(arg("--api-key-file")).stat().st_mode & 0o777),
                    "env": {k: v for k, v in os.environ.items() if k.startswith(("LLAMA_", "AIP_"))
                            or k in ("CUDA_VISIBLE_DEVICES", "GGML_BACKEND_PATH", "MTMD_BACKEND_DEVICE")}})
    if MODE == "fatal":
        print("device kernel image is invalid", file=sys.stderr, flush=True)
        return 23
    print("E CUDA: no CUDA-capable device", file=sys.stderr, flush=True)
    print("CPU model buffer size = 10.00 MiB", flush=True)
    if MODE in ("flood", "long-log"):
        for channel in (sys.stdout, sys.stderr):
            if MODE == "long-log":
                print("ě" * 100000, file=channel, flush=True)
            else:
                for i in range(1500):
                    print(f"{i}: " + "buffer diagnostic " * 12, file=channel)
                channel.flush()
    if MODE == "key-log":
        print("private key " + KEY, flush=True)
    if os.environ.get("STUB_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    server = await asyncio.start_server(handle, "127.0.0.1", int(arg("--port")))
    port = server.sockets[0].getsockname()[1]
    if MODE != "hang-load":
        print(f"srv   run_server: listening on http://127.0.0.1:{port}", flush=True)
    else:
        print("Loading model", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
