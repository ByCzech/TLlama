"""Small TCP subprocess implementing the pinned protocol for lifecycle tests."""

import asyncio
import base64
import json
import os
import signal
import sys
import struct
import zlib
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


def image_reply(body):
    parts = [part for message in body.get('messages', []) if isinstance(message.get('content'), list)
             for part in message['content'] if part.get('type') == 'image_url']
    if not parts:
        return None
    # Decode the test/smoke PNG's actual RGB panels independently of its prompt.
    data = base64.b64decode(parts[-1]['image_url']['url'].split(',', 1)[1])
    width = struct.unpack('>I', data[16:20])[0]
    compressed, offset = b'', 8
    while offset < len(data):
        size = struct.unpack('>I', data[offset:offset+4])[0]
        if data[offset+4:offset+8] == b'IDAT':
            compressed += data[offset+8:offset+8+size]
        offset += size + 12
    row = zlib.decompress(compressed)
    names = {b'\xff\0\0':'red', b'\0\0\xff':'blue'}
    return json.dumps({'left':names.get(row[1:4], 'unknown'),
                       'right':names.get(row[1+width//2*3:4+width//2*3], 'unknown')})


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
    split, quoted, escaped = -1, False, False
    for index, char in enumerate(data):
        if escaped:
            escaped = False
        elif quoted and char == '\\':
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ',' and not quoted:
            split = index
            break
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
                "build_info": (None if MODE == "missing-build" else "wrong" if MODE == "wrong-build"
                               else os.environ.get("STUB_BUILD", "b11552-ec2797836f")),
                "model_alias": "wrong" if MODE == "wrong-alias" else arg("--alias"),
                "total_slots": 4 if MODE == "wrong-slots" else 1,
                "model_path": arg("--model"),
                "default_generation_settings": {"n_ctx": 0 if MODE == "bad-context" else 2048},
                "modalities": {"vision": "--mmproj" in sys.argv and MODE != 'no-vision'},
                "bos_token": "<bos>", "eos_token": "<eos>",
                "chat_template": (Path(arg("--chat-template-file")).read_text() if "--chat-template-file" in sys.argv
                                  else "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"),
            })
            return
        if path == "/tllama/memory":
            behavior = (ROOT / "memory-behavior").read_text() if (ROOT / "memory-behavior").exists() else "normal"
            if behavior == "hang":
                await reader.read()
                mark("cancelled", "memory")
                return
            if behavior == "disconnect":
                return
            status = int((ROOT / "memory-status").read_text()) if (ROOT / "memory-status").exists() else 200
            file = ROOT / "memory-report"
            if not file.exists():
                await json_response(writer, {"error": "not found"}, 404)
                return
            data = file.read_bytes()
            writer.write(f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
            await writer.drain()
            return
        if path == "/tokenize":
            # Deterministic IDs expose exact render text in the request log.
            text = body["content"]
            tokens = [ord(char) + 10 for char in text]
            if text.startswith("<bos>"):
                tokens = [1] + [ord(char) + 10 for char in text[5:]]
            if body.get("add_special"):
                tokens.insert(0, 1)
            await json_response(writer, {"tokens": [True] if MODE == 'bad-tokenize' else tokens})
            return
        async with LOCK:
            # The public smoke's long stream must still be running when its
            # client observes /api/ps and closes the connection.
            held = body.get('ignore_eos') or body.get('stream') and body.get('max_tokens', body.get('n_predict')) == 512
            mode = body.get("stub", "hold" if held else "normal")
            if mode == "crash":
                os._exit(71)
            if mode == "error":
                await json_response(writer, {"error": {"code": 400, "message": "bad request"}}, 400)
                return
            chat = path == "/v1/chat/completions"
            if MODE == 'bad-image':
                await json_response(writer, {'error':{'code':400, 'message':'Image decoder rejected the payload'}}, 400)
                return
            vision_text = image_reply(body)
            if mode == "hold" and not body["stream"]:
                mark("started", "non-stream")
                await reader.read()
                mark("cancelled", "non-stream")
                return
            if not body["stream"] or mode == "json-stream":
                constrained = 'response_format' in body or 'json_schema' in body
                text = vision_text or ('{"answer":42}' if constrained else "Ahoj světe")
                if chat:
                    result = {"choices": [{"message": {"content": text, "reasoning_content": "myšlenka"},
                                           "finish_reason": "stop"}], "usage": USAGE, "timings": TIMINGS}
                else:
                    result = {"content": text, "stop": True, "stop_type": "eos", "timings": TIMINGS,
                              "tokens_evaluated": 25, "tokens_predicted": 10, "tokens_cached": 35}
                await json_response(writer, result)
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            await stream_bytes(writer, b": ping\r\n\r\n")
            chunk = {"choices": [{"delta": {"content": vision_text or "Ahoj světe"}, "finish_reason": None}]} if chat else {
                "content": "Ahoj světe", "stop": False}
            await stream_bytes(writer, event(chunk))
            if mode == "hold":
                mark("started", "stream")
                await reader.read()
                mark("cancelled", "stream")
                return
            if mode == 'controlled':
                mark('started', 'controlled')
                disconnected = asyncio.create_task(reader.read())
                try:
                    while not (ROOT / 'release').exists():
                        if disconnected.done():
                            mark('cancelled', 'controlled')
                            return
                        await asyncio.sleep(0.01)
                finally:
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
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
    if MODE == 'memory-full-offload':
        print('load_tensors: offloaded 41/41 layers to GPU', flush=True)
        print('load_tensors: CPU_Mapped model buffer size = 2.00 MiB', flush=True)
        print('load_tensors: Vulkan0 model buffer size = 10.00 MiB', flush=True)
        print('llama_kv_cache: Vulkan0 KV buffer size = 1.00 MiB', flush=True)
        print('llama_memory_recurrent: Vulkan0 RS buffer size = 2.00 MiB', flush=True)
        print('llama_context: Vulkan_Host output buffer size = 0.50 MiB', flush=True)
        print('sched_reserve: Vulkan0 compute buffer size = 3.00 MiB', flush=True)
        print('sched_reserve: Vulkan_Host compute buffer size = 1.00 MiB', flush=True)
    else:
        print("load_tensors: CPU model buffer size = 10.00 MiB", flush=True)
    if MODE not in {'memory-missing', 'memory-full-offload'}:
        print("llama_kv_cache: CPU KV buffer size = 1.00 MiB", flush=True)
        print("llama_memory_recurrent: CPU RS buffer size = 2.00 MiB", flush=True)
        print("llama_context: CPU output buffer size = 0.50 MiB", flush=True)
        print("sched_reserve: CPU compute buffer size = 3.00 MiB", flush=True)
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
    print('srv    load_model: initializing, n_slots = 1, n_ctx_slot = 2048', flush=True)
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
