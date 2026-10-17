# Public Tools checkpoints 0045 / 0046

Working verification instructions; final user documentation is deferred.
0044 established the native reference. This checkpoint implements the public
argument representations and stream state. Complete Tools acceptance still
requires this hardware run and the remaining admission/client checks.

0045 public hardware evidence is now reviewed: 18/22 exact initial matches,
16/16 executed follow-ups, stable parallel identities, live thinking, one-time
Ollama object delivery, native cancel at n_gen=10/512 and same-worker reuse.
Only the four known escaped-key semantic mismatches remain, one per API/mode;
no public protocol failure was observed. Worker exit/cleanup succeeded.

0046 adds pre-admission validation and actual Python SDK checks. For the next
hardware check use **--clients-only** below; the accepted raw matrix need not
be repeated unless something changes.

## SDK-only check after 0046

The optional Python packages `openai` and `ollama` must be installed in the
chosen test environment. They are not TLlama runtime dependencies. Locally
verified with openai 3.33.0 (httpx2) and ollama 0.6.3 (httpx); other installed
versions are recorded by the diagnostic. Use a virtual environment if an
isolated installation is needed; do not change the system environment solely
for this test. Optional pytest cases skip when a client package is absent.

```bash
pytest -q tests/test_native_tool_input.py tests/test_tools_clients.py tests/test_tools_api_smoke.py

export PYTHONPATH=src
export TLLAMA_RUNTIME_DEVICE=Vulkan0
tools_results="/tmp/tllama-tools-clients-$(date +%Y%m%d-%H%M%S)"
python scripts/tools-api-smoke.py --clients-only \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$tools_results"
tar -cJf "$tools_results.tar.xz" -C "$tools_results" .
```

This runs four initial parallel-call requests: each SDK with stream=false/true.
Exact matches receive two fixed fixture results, followed by a non-stream SDK
answer that must include both markers. SDK objects are replayed directly for
Ollama, which may discard server IDs/type in its typed call model. This tests
the real ID-less history path; the raw 0045 matrix retained IDs. Neither SDK
executes model-generated code. No model download or native rebuild is needed.
SDK events/versions/results and native logs are saved in the same archive form.

Ollama name-only results for repeated calls to the same function must be returned
in original call order. Unnamed results follow the full call order. Explicit
IDs, when present, take precedence and permit results in any order. TLlama fills
missing result IDs only from matched assistant calls; it never invents IDs.
The OpenAI contract requires IDs and string arguments. Historical tools need
not appear in the current declarations, allowing tool sets to change per turn.

TLlama sends the native call ID by default in Ollama result and stream calls,
and accepts tool_call_id on returned tool results. This behavior already exists
in 0045; SDK compatibility does not remove IDs from the HTTP protocol. The
current upstream Ollama Go API also defines optional ToolCall.id and
Message.tool_call_id (api/types.go, checked 2026-10-17). The Python SDK's typed
objects can discard those fields, so a client using IDs must preserve them on
both deserialization and replay. A server switch cannot change SDK behavior.
Default ID emission stays enabled; there is no new disable switch without a
demonstrated incompatible client.

Invalid declarations, duplicate function names/IDs within a turn, orphan or
duplicate results, incomplete call/result histories and malformed completed
arguments now return HTTP 400 before model admission. The complete result set
must precede the next ordinary message or inference request. Tool JSON Schema
extensions remain unchanged; the native runtime evaluates schema support.

## Original raw HTTP matrix

Run from the patched checkout, in the same Python environment as pytest:

```bash
pytest -q tests/test_native_tools.py tests/test_native_public_tools.py tests/test_tools_api_smoke.py

export PYTHONPATH=src
export TLLAMA_RUNTIME_DEVICE=Vulkan0
tools_results="/tmp/tllama-tools-public-$(date +%Y%m%d-%H%M%S)"
python scripts/tools-api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$tools_results"
tar -cJf "$tools_results.tar.xz" -C "$tools_results" .
```

First release the running service's loaded model so it does not compete with
this diagnostic. No new model or native rebuild is needed. This script starts
an isolated loopback TLlama HTTP server and its private authenticated worker.
It does not connect to OLLAMA_HOST or modify registered TOML files. All inference
requests go through /api/chat or /v1/chat/completions, including result replay.

Optional --models-dir and --executable choose an alternate repository/binary.
The default runtime remains b11552-3144c167ef under the normal warn/strict policy.
Registered device/KV/flash-attention/template/mmap settings stay in force.
The diagnostic explicitly overrides context to --ctx (4096) at every admission,
including OpenAI which has no context-size request field, and batch/ubatch to
512. Actual argv/context/fingerprint are saved. Request temperature is zero;
Ollama additionally receives seed 42. OpenAI has no seed field in the current
public schema. --predict=256, --load-timeout=600 and --request-timeout=180 are
configurable. Output directories must be new; evidence is never overwritten.

## Coverage and evidence

Both result and stream modes use an independent client consumer. It appends
OpenAI name/argument string fragments, while Ollama must deliver whole argument
objects exactly once. The consumer does not import production assembly code.
Only declarations/prompts and semantic inspection are shared with the 0044
reference diagnostic.

- Ollama: automatic single call, automatic parallel calls to the same function,
  escaped Czech/quoted/backslash/newline key, and thinking with Tools.
- OpenAI: the six native reference probes (auto, required, none, parallel,
  escaped key, thinking), plus a named forced choice.
- Matching calls receive fixed in-memory fixture results carrying unique marker
  strings; the next model response must incorporate those exact markers. This
  is caller-controlled test dispatch only: no generated code, arbitrary function
  execution, network tool or automatic server agent loop.
- OpenAI cancellation closes the public connection after the first argument
  fragment of a long call, dispatches nothing, then verifies another tool request
  on the same worker. Retain worker.log: a native `cancel task` before the token
  limit is the evidence of early cancellation; reuse alone is insufficient.
- Cleanup checks worker exit, temporary directory removal and empty residency.

Ollama exposes no public tool_choice or parallel_tool_calls switch, so its
parallel-auto probe tests model choice rather than claiming forced parallelism.
The full matrix contains 22 initial requests and, if all semantics match, 20
follow-ups. Model mismatches skip follow-up dispatch and remain review evidence.
The escaped key mismatch seen in 0044 (literal backslash+n instead of newline)
may recur. Preserve it; do not change escaping or repair the model's arguments.

raw.jsonl retains public requests, result bodies and every parsed stream event
with local receipt times, independently assembled messages, semantic checks,
cancellation and cleanup. summary.jsonl/stdout omit verbose raw frames/requests.
worker.log and NN-load.log retain native task/load logs. Return the whole archive.
Times are diagnostic receipt times, not controlled latency benchmarks.

Exit 0 means transport/protocol and cleanup completed; review_required can still
be true for wrong model arguments, missing markers, length-limited reasoning,
or a cancellation probe that did not reach live arguments. HTTP errors, malformed
completed call data and broken terminal framing fail the protocol diagnostic.
Synthetic subprocess pytest coverage does not certify model quality.

## Public stream policy

Native full-name revisions become OpenAI suffix deltas, preserving append-based
clients. IDs stay stable per index; two calls to one function remain independent.
Argument text stays byte-for-byte unchanged on OpenAI output.

Ollama receives object arguments only after the native tool_calls finish AND
successful transport termination ([DONE]). The inspected native protocol has
no individual call commit marker. A parseable JSON prefix is insufficient:
the model can append more. Thus an earlier parallel call waits for the end of
the response. Ordinary content and thinking still stream immediately.

Completed Ollama calls are emitted once in a done=false message, followed by
the separate done=true statistics message. Successful Ollama tool turns end
with done_reason=stop; OpenAI retains finish_reason=tool_calls. Non-stream
Ollama returns object calls in its final message. IDs/type are retained and
function.index identifies the call within the turn.

Length, cancellation, malformed JSON, missing [DONE] and upstream failure never
produce a completed Ollama call. OpenAI can already have delivered partial
strings; callers must wait for a valid tool_calls terminal before dispatch.
There is no hidden replay or tool execution. Accumulation is request-local and
bounded to 128 calls and 16 MiB of UTF-8 ID/name/argument data. Completed argument
JSON must be an object without duplicate keys/nonfinite numbers; no repair.

Sources: native 3144c167ef common/chat.cpp/.h and server-chat.cpp; captured
0044 archive SHA-256 85f9528f24ed8c80897ccfc158598a98b81c509886e762224b5de26ac93020fd
(selected unmodified decoded frames in tests/fixtures/native_tools_3144.json).
Ollama public object contract: https://docs.ollama.com/capabilities/tool-calling;
terminal reason mapping checked against upstream llm/server.go DoneReason.String
and server/routes.go ChatHandler on 2026-10-17.
