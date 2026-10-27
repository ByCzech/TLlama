# Native residency

ModelManager uses native leases for public inference and delegates process
ownership to NativeResidency. [Memory reporting](native-memory.md) observes
generations without acquiring leases or extending TTL. There is one inference
engine: private native subprocesses, without a Python inference binding.

Storage references retain their existing meaning: one segment is Local, two
segments TLlama, three segments HuggingFace. `a` identifies `Local/a.toml`;
`Local/a` would instead identify `TLlama/Local/a.toml`. A registered TOML resolves
the physical GGUF/projector path, configuration and version fingerprints.
Definitions that change during reading are retried within a small bound.
Definition/path/header I/O does not hold the residency lock.

## Ownership and policy

`residency.NativeResidency` owns native load/close/exit-observer tasks. It receives
a definition resolver and worker factory, and has no Python inference binding
dependency. ModelManager owns the janitor and delegates expiry to residency. Task cleanup is awaited even if its caller is cancelled.

Use one lease context around the entire request, including transport slot wait:

```python
async with manager.acquire_native_model(name, num_ctx=2048, keep_alive="5m") as lease:
    body = build_native_sampling(options, metadata, resolved=lease.resolved)
    async with lease.worker.chat_stream({**body, "messages": messages}) as events:
        async for event in events:
            ...
```

The lease contains generation number, definition snapshot, resolved configuration
and worker. Request-local timing/cache/reasoning data remains in native results;
public result translation is handled by the native API adapters. The manager adds no second
inference queue: NativeWorker serializes its one-slot transport.

| Event | Behavior |
|---|---|
| Equal virtual name/version/load configuration | Join one manager-owned load or reuse its ready generation |
| Client cancels during shared load | Release its own reservation; other callers and the owned load continue |
| Last loading client cancels | Load may finish as idle residency, subject to its keep_alive |
| Load fails | All joined callers receive the error; overlapping slow definition readers inherit the same failure for the same version |
| Definition resolution overlaps explicit unload | Abort the pending acquisition before it can spawn a late worker |
| Explicit unload during load | Cancel the owned load, notify waiters, reap and release capacity |
| Unload of a used worker | Remove it from selection; already-admitted leases, including slot waiters, finish |
| Definition/load configuration changes | Retire the old generation and create a new one when capacity permits |
| Old lease finishes after reload | Release only its own generation; never close the replacement by model name |
| Unexpected worker exit | Record the failure, close/reap its generation, do not replay inference |
| Later independent request after failure | May start a new generation |
| Shutdown | Reject new work, cancel loads, close active/idle/draining workers and reap owned processes |
| Cleanup failure | Keep that generation in capacity accounting and report failure, not fictitious free capacity |

TOML path/size/mtime_ns changes trigger reload, including `touch` and changes only
to sampling/system. Model/projector file fingerprints also identify replaced
files at the same path. An already-admitted lease keeps its original snapshot;
a replacement during its load does not redirect that request to another worker.
No worker sharing across distinct virtual names is introduced here.

## Capacity and keep_alive

Loading, ready and retiring/closing generations occupy capacity until physical
cleanup finishes. At max_loaded_models=1, selecting another model retires the
previous one and waits for its leases and stop/reap. Larger limits retain the
existing policy: if occupied by ready models, return the limit error; if a load
or retirement can free/change capacity, wait and reevaluate. No stop/load wait
is performed with the global condition lock held.

**Native TTL now starts at completion of the last active use**, including a
cancelled use, instead of consuming keep_alive while a long request is running.
Preload starts TTL after load/use completion. Native private seconds=None pins
residency; the ModelManager façade retains public negative keep_alive semantics.
The latest admitted request sets the generation's policy, independent of which
lease completes first. Zero keep_alive during inference stops after the last
lease; a zero-keep_alive preload only unloads and does not load a new worker.

`native_snapshot()` reports lifecycle, leases, generation, PID, requested/effective
context, keep_alive/remaining TTL and retirement/error information. Busy/pinned
generations have no running idle TTL. It is an internal diagnostic shape, not
`/api/ps` and not measured GPU/RAM residency. The public `/api/ps` adapter uses a separate reporting snapshot with memory
provenance.
Generation numbers remain distinct across full shutdown/start on the same manager.

## ModelManager methods and integration boundary

- `acquire_native_model()` holds a lease for the caller's scope.
- `preload_native_model()` loads/reuses a generation without inference.
- `unload_native_model()` marks matching generations for retirement; returns
  without waiting for active leases to finish.
- `wait_native_closed(generation)` waits for that exact generation's cleanup.
- `native_snapshot()` exposes native lifecycle only.
- `shutdown()` awaits native cleanup, including cancelled loads and active streams;
  even a failed janitor cannot skip this cleanup.

Public generation, embedding and Decision routes use the same native ownership
policy with operation-specific admission. After complete shutdown, start can
begin a fresh residency run. Metadata/storage reads remain independent of
worker loading; the backend module does not import a Python inference binding.

## Hardware check: one registered text model, one native load

Use a registered **text** model name from your existing TLlama model list. The
example assumes its TOML is `Local/Qwen3.6-35B-A3B-UD-IQ3_S.toml`. Adjust the public
reference to its actual location; the physical GGUF path is read from that TOML.

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/manager-smoke.py \
  --model Qwen3.6-35B-A3B-UD-IQ3_S --ctx 2048 --log manager-vulkan.log
```

TOML device still overrides the global selector. The context override deliberately
uses 2048 for this check. The script preloads, reuses the same generation for chat
and stream, requests unload after the first actual content/reasoning chunk,
finishes the stream on its protected lease, then verifies stop/reap and directory
removal after release. It creates no new model definitions and compiles no SDK.

Successful stages are preloaded, ready, chat, unload-during-stream,
stream-finished, worker-closed, manager-closed. Send that JSON output and the
pytest summary; attach the native log if a stage fails. This validates managed
lifecycle and native transport, not public API/vision/template parity or memory
reporting. Multi-model policy is covered by controlled workers and real socket
subprocess tests without loading several large GGUFs on the test hardware.

The [05a API adapter checkpoint](native-api-adapter.md) prepared payloads/results
and serializers ahead of the [05b public switch](native-public-api.md).
