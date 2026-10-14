# Private native worker (development)

The worker adapter is being developed on `refactor/tllama-server`. It does not
replace ModelManager or route public inference requests yet.

`tllama.worker_config.WorkerLaunchSpec` describes one resolved launch without
importing llama-cpp-python. The caller supplies absolute model/projector paths,
a native device selector, load parameters and optional child environment values.
CPU, CUDA, ROCm, Vulkan and other server device names use the same contract.
TOML/environment resolution is available through `native_config.resolve_worker_config`; see [native-config.md](native-config.md). ModelManager/public-route integration is the next milestone.

The initial contract uses one slot, explicit context size, fit disabled and
context shifting disabled. Its default 2048 context is for standalone testing,
not a change to TLlama's public context default. KV K/V types remain independent;
unset flash attention/cache types use the native defaults.

CPU placement is `device="none"`, which disables GPU layers, operation offload
and projector offload. A GPU selector with unset layers, or legacy `-1`, maps to
native `all`; native numeric `-1` would mean auto instead. Conflicting CPU/GPU
layer settings are refused.

The supervisor owns a private cwd and API key file. The loopback server normally
uses port 0; its announced bound address is discovered from the pinned native
log. A fixed loopback port can be supplied for controlled tests. Parent native
server settings (`LLAMA_ARG_*`, `LLAMA_API_KEY`, `MTMD_BACKEND_DEVICE`, Vertex
`AIP_*`) and `GGML_BACKEND_PATH` are removed. Per-worker GPU visibility variables
can override inherited values without changing global `os.environ`.

This does not isolate SDK plugins. Installed plugin loading and system
alternatives retain their current packaging behavior; do not switch alternatives
per request or interpret `GGML_BACKEND_PATH` as an SDK selection override.

The supported native fingerprint is `b11552-ec2797836f`; changing it is an
explicit launch decision and requires confirming protocol/CLI compatibility.
An executable alone is not proof of usable kernels on every discovered device.

## Lifecycle and transport

`tllama.worker.NativeWorker` owns the process, HTTP client and private directory.
`start()` waits for readiness within `load_timeout`; HTTP 503 while loading is
normal. Alias, build fingerprint, slot count and effective context are checked
against native `/props`. Startup failures and cancelled starts clean up the child.
`close()` closes HTTP connections, sends TERM, escalates to KILL if necessary,
reaps the process and removes its private directory. The supervisor is Linux/POSIX.

The initial API serializes inference per worker. `chat()` and `completion()`
return native response objects, preserving usage/timings and reasoning. They do
not yet translate public TLlama API shapes. Use streams as context managers:

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
Private keys are redacted. Buffer allocation parsing and residency reporting
remain for the ModelManager/reporting milestone.

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
