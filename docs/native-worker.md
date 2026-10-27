# Private native worker

ModelManager and native residency manage private worker generations for public
inference. Each generation owns its launch settings, process and transports.

`tllama.worker_config.WorkerLaunchSpec` describes one resolved launch without
importing llama-cpp-python. The caller supplies absolute model/projector paths,
a native device selector, load parameters and optional child environment values.
CPU, CUDA, ROCm, Vulkan and other server device names use the same contract.
Model/default/request resolution is handled by `native_config.resolve_worker_config`;
see [native-config.md](native-config.md).

Workers use one slot and resolved context settings. The standalone launch
contract's default 2048 context does not change the public default of 0
(trained maximum). Automatic placement permits native fitting; explicit
placement and context shifting follow the model/request contract. K/V types
remain independent; public defaults are documented in [native-config.md](native-config.md).

CPU placement is `device="none"`, which disables GPU layers, operation offload
and projector offload. A GPU selector with unset layers, or legacy `-1`, maps to
native `all`; native numeric `-1` would mean auto instead. Conflicting CPU/GPU
layer settings are refused.

The supervisor owns a private `0700` cwd and a fresh
`secrets.token_urlsafe(32)` bearer token (256 bits of randomness). On Linux with
`memfd_create` and procfs, `--api-key-file /proc/self/fd/N` reads the token from
an anonymous memory file inherited through `pass_fds`. The parent closes its
descriptor after spawn; the child retains its copy until exit. Where Python
exposes file-sealing constants, the memory file is sealed against writes and
resizing. No named credential file is created in this mode. The portable
fallback is an exclusively created `0600` file, unlinked after authenticated
readiness or during failed-start cleanup. Its contents are the plaintext token
plus a newline. Token values never enter argv or the child environment; the
HTTP client sends an Authorization bearer header and diagnostic text redacts
the token. This is not public TLlama API authentication, encryption, or isolation
from another process running as the same service user or root. Anonymous memory
is not a guarantee against swap, core dumps or privileged process inspection.

The default local transport is HTTP over a private Unix domain socket:
`--host /PRIVATE/worker.sock`. Generation, tokenization, embeddings, Decision and
the independent memory-telemetry client use that same socket with bearer auth.
No TCP listener or DNS lookup is involved. `worker.address` reports its
`unix://` location; `worker.base_url` is only the HTTP origin used by the client.
The parent prefers an absolute, service-user-owned private `RUNTIME_DIRECTORY`,
then `XDG_RUNTIME_DIR`, otherwise a private temporary directory. The supplied
systemd unit creates `/run/tllama` with mode `0700`. The socket is set to `0600`
after readiness and the worker directory is removed on close. Overlong paths
fail before spawn with a request to select a shorter runtime directory/TMPDIR.
UDS failures never silently switch to TCP.

`WorkerLaunchSpec(transport="tcp")` retains the explicitly selected **local
loopback** adapter. It normally uses port 0; its announced bound address is
discovered from the pinned native log. A fixed loopback port can be supplied
for controlled tests. `scripts/worker-smoke.py --transport tcp` exercises this
alternative; `--port` requires it. This is not remote worker support: process
ownership, model paths and teardown remain local. A future remote adapter must
define authenticated/encrypted transport, remote lifecycle and model placement
separately. Parent native
server settings (`LLAMA_ARG_*`, `LLAMA_API_KEY`, `MTMD_BACKEND_DEVICE`, Vertex
`AIP_*`) and `GGML_BACKEND_PATH` are removed. Per-worker GPU visibility variables
can override inherited values without changing global `os.environ`.

This does not isolate SDK plugins. Installed plugin loading and system
alternatives retain their current packaging behavior; do not switch alternatives
per request or interpret `GGML_BACKEND_PATH` as an SDK selection override.

The default native fingerprint is `b11850-0e9433bfd3`; additional audited builds
are listed in `worker_config.SUPPORTED_RUNTIME_FINGERPRINTS`. New builds require
confirming protocol/CLI compatibility under the configured version policy.
Current feature and hardware audit boundaries are in
[runtime-audit-070.md](runtime-audit-070.md); Decision has its own audit set.
An executable alone is not proof of usable kernels on every discovered device.

## Lifecycle and transport

`tllama.worker.NativeWorker` owns the process, HTTP client and private directory.
`start()` waits for readiness within `load_timeout`; HTTP 503 while loading is
normal. Alias, build fingerprint, slot count and effective context are checked
against native `/props`. Startup failures and cancelled starts clean up the child.
`close()` closes HTTP connections, sends TERM, escalates to KILL if necessary,
reaps the process and removes its private directory. The supervisor is Linux/POSIX.

The transport serializes inference per worker. `chat()` and `completion()`
return native response objects, preserving usage/timings and reasoning. They do
not themselves translate public TLlama API shapes; public adapters own that
mapping. Use streams as context managers:

```python
async with NativeWorker(spec) as worker:
    async with worker.chat_stream({"messages": messages}) as events:
        async for event in events:
            ...
```

Leaving the stream context closes the upstream response, including after early
break or cancellation. No public headers are forwarded; `X-Conversation-Id`
would enable native reconnectable streams and change disconnect semantics.
Chat streams request usage internally and preserve its final empty `choices`.
SSE comments and fragmented UTF-8/multiline data are handled. Missing terminal
events, protocol errors, HTTP errors and worker crashes are reported without
replaying requests. Raw completion uses its `stop=true` terminal event instead
of requiring OpenAI `[DONE]`.

Both log pipes are drained continuously. Startup log capture and diagnostic tail
are bounded; `load_log_complete` indicates whether capture hit a truncation limit.
Private keys are redacted. Memory telemetry selects native inventory or the
supported log fallback; public reporting preserves provenance and unknown
values. See [native-memory.md](native-memory.md).

## Hardware smoke test

Run from the repository root with its Python dependencies installed. This entry
imports the standalone worker, not the application or inference binding. No
compiler/SDK build is needed. The default smoke run loads once, tests chat/raw
completion with and without streaming, both cancellation paths and graceful
shutdown. It is not a performance benchmark.

```bash
PYTHONPATH=src python scripts/worker-smoke.py \
  --model /var/lib/tllama/models/HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --device none --ctx 2048 --flash-attn on --type-k q4_0 --type-v q4_0 \
  --log worker-cpu.log
```

For the second run change `--device none` to `--device Vulkan0` and the log name.
CUDA/ROCm use their actual native selectors through the same interface. Optional
visibility overrides are per child, for example `--env CUDA_VISIBLE_DEVICES=0`.
The system-selected SDK plugin must already support the hardware.

`--crash` adds a forced crash of only the owned test worker before cleanup.
`--cancel-start-after 1` instead tests a cancelled load; choose a delay shorter
than that model's load time. Each option can be tested on a separate run when
needed. A successful run ends with `stage=closed`, a reaped return code and
`directory_removed=true`; send the JSON stage output and relevant native log.

`request_timeout` is an HTTP inactivity timeout, distinct from the whole startup
deadline and external cancellation. Model-specific template/vision/API parity,
multi-model management and memory accounting are not asserted by this smoke test.

See [native-residency.md](native-residency.md),
[native-public-api.md](native-public-api.md) and [native-memory.md](native-memory.md)
for managed lifecycle, public request/response mapping and allocation reporting.
