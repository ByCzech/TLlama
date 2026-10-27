# Client examples: Python, curl and JavaScript

All examples call a running TLlama daemon. Start `tllamad`, then replace `MODEL`
with the exact registered generation-model name from `tllama list` or
`GET /api/tags`. Use `EMBEDDING_MODEL` only for a registered embedding model.
These clients do not download missing models or require an Ollama daemon.

| Client/API | Base URL | Response/stream |
| --- | --- | --- |
| Ollama SDK or direct Ollama API | `http://127.0.0.1:54800` | JSON / NDJSON |
| OpenAI SDK | `http://127.0.0.1:54800/v1` | JSON / SSE |
| Direct OpenAI-compatible HTTP | `http://127.0.0.1:54800/v1/chat/completions` | JSON / SSE |

The `tllama` import alias below refers to the third-party `ollama` client
library, not a Python inference binding. The host must point to TLlama.
Use **Chat Completions** with OpenAI clients: `/v1/responses` and
`/v1/completions` are not implemented. The nonempty `api_key="tllama"` is an SDK
placeholder, not TLlama public authentication. For a proxy requiring a real
credential, configure that credential and its URL instead.

## Contents

- [Install Python clients](#install-python-clients)
- [Python: ollama](#python-ollama) — simple chat, history, streaming and async.
- [Python: openai](#python-openai) — simple chat, async streaming and embeddings.
- [Python: requests](#python-requests) — JSON and checked NDJSON streaming.
- [Python: httpx and httpx2](#python-httpx-and-httpx2) — sync JSON and async streaming.
- [CLI: curl](#cli-curl) — listing, chat, generation, NDJSON and SSE.
- [JavaScript: ollama](#javascript-ollama) — simple chat and streaming history.
- [JavaScript: openai](#javascript-openai) — simple chat and streaming.
- [JavaScript: fetch](#javascript-fetch) — JSON and incremental NDJSON without an SDK.
- [Model-specific workflows and failures](#model-specific-workflows-and-failures).
- [Client references](#client-references).

Each code block is independent unless it explicitly says it replaces an import
or continues a preceding example. Request timeouts allow for cold model loading;
adjust them for your host. Python HTTP read timeouts limit inactivity, not the
total duration of inference. JavaScript `AbortSignal.timeout` below bounds the
whole fetch, including reading its response.

## Install Python clients

Install only the clients you use, in your application's environment:

```sh
python -m pip install ollama
python -m pip install openai
python -m pip install requests
python -m pip install 'httpx>=0.28,<1'
python -m pip install httpx2
```

HTTPX examples target the stable 0.28 API; HTTPX2 is a separate package with
the corresponding client API. Neither package name means that TLlama requires
HTTP/2. These are optional client dependencies; installing a client does not
install the TLlama daemon or its native runtime.

## Python: ollama

### Simple chat

```python
import ollama as tllama

client = tllama.Client(host="http://127.0.0.1:54800", timeout=300.0)
reply = client.chat(
    model="MODEL",
    messages=[{"role": "user", "content": "Explain GGUF in one sentence."}],
    stream=False,
)
print(reply.message.content)
```

For a prompt without chat history, use `client.generate(model="MODEL",
prompt="Explain GGUF.", stream=False)` and read `reply.response`.

### Streaming and conversation history

The caller owns history and sends it with every turn. Append a generated answer
only after successful completion; output already printed before an error is
partial. `options` overrides supported sampling settings for these requests.

```python
import ollama as tllama

client = tllama.Client(host="http://127.0.0.1:54800", timeout=300.0)
history = [{"role": "system", "content": "Give concise technical explanations."}]

for question in ("What is GGUF?", "How does quantization reduce its size?"):
    history.append({"role": "user", "content": question})
    parts = []
    done = False
    stream = client.chat(
        model="MODEL", messages=history, stream=True,
        options={"temperature": 0.2, "num_predict": 256},
    )
    try:
        for part in stream:
            text = part.message.content or ""
            print(text, end="", flush=True)
            parts.append(text)
            done = done or part.done
    finally:
        stream.close()
    if not done:
        raise RuntimeError("Incomplete Ollama stream: missing done=true")
    history.append({"role": "assistant", "content": "".join(parts)})
    print()
```

`tllama.ResponseError` reports server errors, including in-stream errors;
transport failures also propagate. This example displays final-answer content.
Thinking, when present, is a separate `message.thinking` field and need not be
displayed or appended as answer text. A token-limit terminal can still mean a
truncated answer: inspect `part.done_reason` when completion length matters.

### Async chat

```python
import asyncio
import ollama as tllama

async def main():
    client = tllama.AsyncClient(host="http://127.0.0.1:54800", timeout=300.0)
    reply = await client.chat(
        model="MODEL",
        messages=[{"role": "user", "content": "Explain KV cache briefly."}],
        stream=False,
    )
    print(reply.message.content)

asyncio.run(main())
```

For async streaming, `stream = await client.chat(..., stream=True)`, then
`async for part in stream`; retain the same `done` check and close the generator
with `await stream.aclose()` in `finally`. In a notebook with an active event
loop, use `await main()` instead of `asyncio.run(main())`.

## Python: openai

### Simple Chat Completions

```python
from openai import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:54800/v1", api_key="tllama",
    timeout=300.0, max_retries=0,
) as client:
    reply = client.chat.completions.create(
        model="MODEL",
        messages=[{"role": "user", "content": "Explain GGUF in one sentence."}],
        temperature=0.2,
        max_tokens=256,
    )
    print(reply.choices[0].message.content)
```

`max_retries=0` makes failures visible without automatically submitting a
generation again. Applications can implement their own retry policy.

### Async streaming with terminal validation

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    async with AsyncOpenAI(
        base_url="http://127.0.0.1:54800/v1", api_key="tllama",
        timeout=300.0, max_retries=0,
    ) as client:
        stream = await client.chat.completions.create(
            model="MODEL",
            messages=[{"role": "user", "content": "Explain KV cache briefly."}],
            max_tokens=256,
            stream=True,
        )
        finished = False
        async with stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    print(choice.delta.content or "", end="", flush=True)
                    finished = finished or choice.finish_reason is not None
        if not finished:
            raise RuntimeError("Incomplete chat stream: missing finish_reason")
        print()

asyncio.run(main())
```

SDKs parse SSE framing; this check additionally rejects EOF before a finish
choice. A server error must still propagate even after partial output or a
finish choice. `finish_reason="length"` means the answer reached its token
limit. A synchronous stream uses the same `client.chat.completions.create`
call with `stream=True`, `with stream` and a normal `for` loop.

### Embeddings

```python
from openai import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:54800/v1", api_key="tllama",
    timeout=300.0, max_retries=0,
) as client:
    result = client.embeddings.create(
        model="EMBEDDING_MODEL", input=["First document.", "Second document."],
        encoding_format="float",
    )
    for item in result.data:
        print(item.index, len(item.embedding))
```

Add model-required query/document prefixes explicitly. See the
[embedding API](embedding-api.md) for dimensions, bounds and normalization.

## Python: requests

### Simple JSON request

```python
import requests

with requests.post(
    "http://127.0.0.1:54800/api/chat",
    json={
        "model": "MODEL", "stream": False,
        "messages": [{"role": "user", "content": "Explain GGUF briefly."}],
    },
    timeout=(5, 300),
) as response:
    response.raise_for_status()
    print(response.json()["message"]["content"])
```

### Streaming with a reusable session

The JSON `stream` field asks TLlama to stream. The separate Requests
`stream=True` prevents the HTTP client from buffering the entire response.

```python
import json
import requests

with requests.Session() as client:
    with client.post(
        "http://127.0.0.1:54800/api/chat",
        json={
            "model": "MODEL", "stream": True,
            "messages": [{"role": "user", "content": "Explain KV cache."}],
            "options": {"temperature": 0.2, "num_predict": 256},
        },
        stream=True, timeout=(5, 300),
    ) as response:
        response.raise_for_status()
        done = False
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            print(event.get("message", {}).get("content", ""), end="", flush=True)
            done = done or event.get("done") is True
        if not done:
            raise RuntimeError("Incomplete NDJSON stream: missing done=true")
        print()
```

`iter_lines` handles lines split across network reads. The small `chunk_size`
favors immediate display over throughput. HTTP status alone cannot detect a
failure sent inside an already-started stream.

## Python: httpx and httpx2

Both examples below work with either of these imports; choose one explicitly:

```python
import httpx
```

```python
import httpx2 as httpx
```

### Simple synchronous JSON request

```python
import httpx  # Or: import httpx2 as httpx

with httpx.Client(
    base_url="http://127.0.0.1:54800",
    timeout=httpx.Timeout(300.0, connect=5.0),
) as client:
    response = client.post("/api/chat", json={
        "model": "MODEL", "stream": False,
        "messages": [{"role": "user", "content": "Explain GGUF briefly."}],
    })
    response.raise_for_status()
    print(response.json()["message"]["content"])
```

### Async streaming

```python
import asyncio
import json
import httpx  # Or: import httpx2 as httpx

async def main():
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:54800",
        timeout=httpx.Timeout(300.0, connect=5.0),
    ) as client:
        async with client.stream("POST", "/api/chat", json={
            "model": "MODEL", "stream": True,
            "messages": [{"role": "user", "content": "Explain KV cache."}],
        }) as response:
            response.raise_for_status()
            done = False
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise RuntimeError(event["error"])
                print(event.get("message", {}).get("content", ""), end="", flush=True)
                done = done or event.get("done") is True
            if not done:
                raise RuntimeError("Incomplete NDJSON stream: missing done=true")
            print()

asyncio.run(main())
```

Reuse a client for connection pooling in a long-lived application. Direct HTTP
also reaches the OpenAI-compatible API: post to `/v1/chat/completions`, omit
Ollama's `options` wrapper and read `choices[0].message.content` for nonstreaming
chat. SSE streaming needs an SSE parser; it is not the NDJSON loop above.

## CLI: curl

These commands use a POSIX shell and a curl supporting `--fail-with-body`
(7.76 or later). Set the exact registered model and server URL once:

```sh
TLLAMA_URL=http://127.0.0.1:54800
MODEL=MODEL
curl --fail-with-body --silent --show-error "$TLLAMA_URL/api/tags"
```

`jq` builds JSON safely even if the name or prompt contains shell/JSON-sensitive
characters. It is needed by the following commands; curl itself sends the HTTP.

### Chat and generation

```sh
jq -n --arg model "$MODEL" --arg prompt 'Explain GGUF briefly.' \
  '{model:$model,messages:[{role:"user",content:$prompt}],stream:false}' |
  curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 300 \
    "$TLLAMA_URL/api/chat" -H 'Content-Type: application/json' --data-binary @-

jq -n --arg model "$MODEL" --arg prompt 'Explain KV cache.' \
  '{model:$model,prompt:$prompt,stream:false}' |
  curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 300 \
    "$TLLAMA_URL/api/generate" -H 'Content-Type: application/json' --data-binary @-
```

### Ollama NDJSON and OpenAI SSE streaming

```sh
jq -n --arg model "$MODEL" \
  '{model:$model,messages:[{role:"user",content:"Explain GGUF."}],stream:true}' |
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout 5 --max-time 300 "$TLLAMA_URL/api/chat" \
    -H 'Content-Type: application/json' --data-binary @-

jq -n --arg model "$MODEL" \
  '{model:$model,messages:[{role:"user",content:"Explain GGUF."}],stream:true}' |
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout 5 --max-time 300 "$TLLAMA_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary @-
```

These display the raw protocol: NDJSON ends with `done:true`; successful SSE
includes a finish choice and `data: [DONE]`. Inspect body errors and terminals.
`--fail-with-body` catches HTTP errors, but a streamed error after HTTP 200 can
still leave curl with exit code 0. Use a checked client above for automation.
`--max-time` limits total request time, including cold loading and generation.

## JavaScript: ollama

Official JavaScript clients exist for both compatible APIs. These SDK examples
use Node.js 22+ and ES modules: save each as `example.mjs` and run
`node example.mjs`. Install only the package needed by that example:

```sh
npm install ollama
npm install openai
```

### Simple chat

```javascript
import { Ollama } from 'ollama';

const tllama = new Ollama({ host: 'http://127.0.0.1:54800' });
const reply = await tllama.chat({
  model: 'MODEL',
  messages: [{ role: 'user', content: 'Explain GGUF in one sentence.' }],
  stream: false,
});
console.log(reply.message.content);
```

### Streaming and history

```javascript
import { Ollama } from 'ollama';

const tllama = new Ollama({ host: 'http://127.0.0.1:54800' });
const messages = [{ role: 'system', content: 'Give concise technical answers.' }];

for (const question of ['What is GGUF?', 'How does quantization reduce its size?']) {
  messages.push({ role: 'user', content: question });
  const stream = await tllama.chat({
    model: 'MODEL', messages, stream: true,
    options: { temperature: 0.2, num_predict: 256 },
  });
  let answer = '';
  let done = false;
  try {
    for await (const part of stream) {
      const text = part.message.content || '';
      process.stdout.write(text);
      answer += text;
      done ||= part.done === true;
    }
  } finally {
    stream.abort();
  }
  if (!done) throw new Error('Incomplete Ollama stream: missing done=true');
  messages.push({ role: 'assistant', content: answer });
  console.log();
}
```

Errors propagate instead of committing partial text to history. `stream.abort()`
also supports cancelling that request in a UI. Browser bundlers can import
`Ollama` from `ollama/browser`; replace `process.stdout.write` with UI updates.

## JavaScript: openai

### Simple Chat Completions

```javascript
import OpenAI from 'openai';

const tllama = new OpenAI({
  baseURL: 'http://127.0.0.1:54800/v1', apiKey: 'tllama',
  timeout: 300_000, maxRetries: 0,
});
const reply = await tllama.chat.completions.create({
  model: 'MODEL',
  messages: [{ role: 'user', content: 'Explain GGUF in one sentence.' }],
  temperature: 0.2, max_tokens: 256,
});
console.log(reply.choices[0].message.content);
```

### Streaming

```javascript
import OpenAI from 'openai';

const tllama = new OpenAI({
  baseURL: 'http://127.0.0.1:54800/v1', apiKey: 'tllama',
  timeout: 300_000, maxRetries: 0,
});
const stream = await tllama.chat.completions.create({
  model: 'MODEL',
  messages: [{ role: 'user', content: 'Explain KV cache briefly.' }],
  stream: true, max_tokens: 256,
});
let finished = false;
try {
  for await (const chunk of stream) {
    for (const choice of chunk.choices) {
      process.stdout.write(choice.delta.content || '');
      finished ||= choice.finish_reason != null;
    }
  }
} finally {
  stream.controller.abort();
}
if (!finished) throw new Error('Incomplete chat stream: missing finish_reason');
console.log();
```

Use the SDK's Chat Completions methods, not its default Responses tutorial.
The final-answer text and any reasoning extension are separate; see
[public API behavior](native-public-api.md). The SDK examples above run in Node;
the fetch examples below also fit a browser without an SDK-specific browser option.

## JavaScript: fetch

Node.js 22+ and modern browsers provide `fetch`; no npm library is needed.
In Node, run these as `.mjs`. In a browser, place them in a module script and
replace console output with UI updates. Serve a local page over HTTP rather
than opening it as `file://`. Built-in CORS origins cover localhost/loopback
pages; add other page origins with `server.origins`, `--origins` or
`TLLAMA_ORIGINS` as described in [daemon configuration](daemon.md).
An HTTPS page can also be blocked from calling a plain-HTTP daemon by browser
mixed-content/local-network rules; use an appropriate same-origin proxy when needed.

### Simple JSON request

```javascript
const response = await fetch('http://127.0.0.1:54800/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'MODEL', stream: false,
    messages: [{ role: 'user', content: 'Explain GGUF briefly.' }],
  }),
  signal: AbortSignal.timeout(300_000),
});
if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
const reply = await response.json();
console.log(reply.message.content);
```

### Incremental NDJSON streaming

Network chunks do not coincide with JSON lines or UTF-8 characters. This parser
retains incomplete lines, decodes UTF-8 incrementally, checks body errors and
requires a successful terminal. `console.log` is the replaceable output callback.

```javascript
const response = await fetch('http://127.0.0.1:54800/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'MODEL', stream: true,
    messages: [{ role: 'user', content: 'Explain KV cache briefly.' }],
  }),
  signal: AbortSignal.timeout(300_000),
});
if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
if (!response.body) throw new Error('Missing response body');

const reader = response.body.getReader();
const decoder = new TextDecoder('utf-8', { fatal: true });
let pending = '';
let done = false;
function consume(line) {
  if (!line.trim()) return;
  const event = JSON.parse(line);
  if ('error' in event) throw new Error(String(event.error));
  if (event.message?.content) console.log(event.message.content);
  done ||= event.done === true;
}
try {
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    pending += decoder.decode(chunk.value, { stream: true });
    let end;
    while ((end = pending.indexOf('\n')) !== -1) {
      consume(pending.slice(0, end));
      pending = pending.slice(end + 1);
    }
  }
  pending += decoder.decode();
  consume(pending);
  if (!done) throw new Error('Incomplete NDJSON stream: missing done=true');
} finally {
  await reader.cancel();
  reader.releaseLock();
}
```

For OpenAI SSE, use the OpenAI library or an SSE parser that handles complete
events, error frames and `[DONE]`. Do not apply `JSON.parse` directly to an SSE
line starting with `data:`.

## Model-specific workflows and failures

The examples show transport and client usage. Actual model capabilities,
templates and runtime settings determine whether a request is supported:

- [Tools](native-tools.md): the caller executes allowed functions and returns
  results with the required history/IDs. Never execute partial streamed calls.
- [Vision](native-vision.md): configure a matching projector; use supported inline
  image representations rather than remote image URLs.
- [Structured output and reasoning](native-public-api.md): use the documented
  fields and supported schemas; validate returned content in the application.
- [FIM](native-fim.md): use `/api/generate` with prefix and nonempty `suffix` on
  a suitable FIM model; ordinary coding chat is a different workflow.
- [Decision](decision-api.md): use direct HTTP JSON at `/v1/systemone`; neither
  standard SDK has a Decision convenience method.
- [Per-model configuration](configuration.md): device, context, K/V precision and
  MTP belong to the documented runtime/model contract. Do not assume arbitrary
  SDK fields become supported request overrides.

For an HTTP error, retain its status and response body. For SDK calls, let SDK
API/transport exceptions reach your application's error handling. A stream can
fail after printing output, so store an answer as complete only after successful
iteration and its terminal check. Explicitly cancelled requests are partial too.
Reuse clients where appropriate and close streams on cancellation/error.

## Client references

Client call shapes reviewed 27 October 2026 against their official documentation:

- [Ollama Python](https://github.com/ollama/ollama-python): custom host,
  `Client`/`AsyncClient`, chat, streaming and response errors.
- [OpenAI Python](https://github.com/openai/openai-python): custom base URL,
  Chat Completions, async streams, timeout/retries and embeddings.
- [Requests](https://requests.readthedocs.io/en/latest/user/advanced/): sessions,
  streaming, `iter_lines` and response cleanup.
- [HTTPX](https://www.python-httpx.org/async/) and
  [HTTPX2](https://github.com/pydantic/httpx2): sync/async clients and streaming.
- [curl](https://curl.se/docs/manpage.html): HTTP failure reporting, no-buffer and
  request time limits.
- [Ollama JavaScript](https://github.com/ollama/ollama-js): custom host, streaming
  async iterators, abort and browser imports.
- [OpenAI JavaScript](https://github.com/openai/openai-node): custom base URL,
  Chat Completions, streaming and abort.

TLlama's [route inventory](api-compatibility.md), [request contract](request-compatibility.md)
and [runtime audit](runtime-audit-070.md) define its supported surface and tested
hardware/model scope. General SDK documentation also describes methods that
TLlama does not implement.
