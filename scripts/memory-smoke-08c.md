# 08c: hardware memory and allocation-locality checks

Working test instructions; final user documentation remains deferred.

For candidate 0.7.0 captures, start with
[runtime-qualification-070.md](../docs/runtime-qualification-070.md). The original
0.6.0 commands below remain an explicit historical baseline. The added `full`
profile requests all GPU layers with mmap and no projector. `--weight-storage`
requires known payload/assignment evidence for a qualification run; defaults
remain compatible with older V1 captures.

Updated for native commit `3144c167ef703e93cee0b58d7236b6a28c6e50ad`
(package `0.6.0~dev+b11552.3-1`). The complete contract remains native
`tools/server/tllama-memory-v1.md`, installed under
`/usr/share/doc/tllama-server-core/` (possibly compressed).

Run from the repository root, using the same Python environment as pytest.
The script launches its own private worker from **physical GGUF paths**.
It does not use the running TLlama service, OLLAMA_HOST, or model TOML files.
Stop the service's loaded model first so two workers do not compete for memory.
Run the profiles sequentially. Each command uses one load and cleans up its worker.

```bash
export PYTHONPATH=src
memory_model=/var/lib/tllama/models/HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
memory_projector=/var/lib/tllama/models/Local/qwen3.6-mmproj-Q8_0.gguf
memory_results="/tmp/tllama-memory-08c-$(date +%Y%m%d-%H%M%S)"
memory_expected=b11552-3144c167ef

# Verify the originally ambiguous CPU input assignment first.
python scripts/memory-smoke.py --locality --expected-fingerprint "$memory_expected" \
  --profile no-mmap --device Vulkan0 \
  --model "$memory_model" --output "$memory_results/no-mmap"

python scripts/memory-smoke.py --locality --expected-fingerprint "$memory_expected" --profile cpu \
  --model "$memory_model" --output "$memory_results/cpu"

python scripts/memory-smoke.py --locality --expected-fingerprint "$memory_expected" \
  --profile partial --device Vulkan0 --gpu-layers 20 \
  --model "$memory_model" --output "$memory_results/partial"

python scripts/memory-smoke.py --locality --expected-fingerprint "$memory_expected" \
  --profile vision --device Vulkan0 \
  --model "$memory_model" --mmproj "$memory_projector" --output "$memory_results/vision"

tar -cJf "$memory_results.tar.xz" -C "$memory_results" .
```

Each output directory must be new; existing results are never overwritten.
Add `--dry-run` to inspect the launch command without starting a worker or creating files.
`TLLAMA_RUNTIME_EXECUTABLE` / `--executable` and the version policy are honored.
`--expected-fingerprint` explicitly pins only this private diagnostic worker;
it does not change TLlama's global supported-build list or warn/strict policy.
A mismatch fails with cleanup before model inference. Omit it for a deliberately
different experiment. `--locality` enables the new extension checks; omit it to
retain the older V1 checkpoint requirements.
The test fixes memory selection to `auto` with independent native/log comparison,
regardless of `TLLAMA_MEMORY_SOURCE` / `TLLAMA_MEMORY_COMPARE`.
Other model/global runtime and sampling defaults are deliberately bypassed.

Defaults: context 4096, batch/ubatch 512, flash attention on, K/V q4_0;
vision uses K q8_0 and V q4_0. All profiles disable mlock.
CPU disables layer, operation and KV offload. The partial profile uses 20 GPU
layers of this 40-layer model. Vision and no-mmap use all GPU layers, with
operation/KV offload enabled. Only no-mmap disables mmap (`--load-mode none`).
`--ctx`, `--type-k`, `--type-v`, `--flash-attn`, and both timeouts can be changed
explicitly if a backend rejects the initial profile. Keep those changes with the results.
CPU inference can be slow; each text request is limited to 32 generated tokens.

| Profile | Evidence to inspect |
| --- | --- |
| CPU + mmap | Known host capacity, zero device bytes, CPU placement, mapped weights |
| Partial Vulkan + mmap | CPU **non-input** weights/layers and accelerator layers; complete host/device totals |
| Vision Vulkan + mmap | Projector weights and compute buffers before images and after each of two different images |
| Full Vulkan without mmap | Allocated weight backing, complete totals and accelerator non-input weights |

Files in each profile directory:

- `summary.jsonl`: concise snapshots, placement, backing totals, projector buffers,
  native-minus-log differences, inference responses and diagnostic check results.
- `raw.jsonl`: all summary records, existing filtered telemetry and **raw_native**:
  the authenticated endpoint's HTTP status and complete JSON body before the
  production V1 adapter filters optional fields. This retains assignment,
  allocation_locality, unknown future fields and independently nullable ranges.
- `worker.log`: redacted native/debug logs for analysis if a check fails.

Send the complete archive for this checkpoint; raw JSON is essential. The stdout
output is the same as the summary file.

Raw reads use the owned memory HTTP client/poller lock and a bounded timeout;
they do not acquire an inference slot. They and filtered telemetry are separate
HTTP observations and may have different sequences. `summary.locality.snapshot`
belongs to raw_native; `summary.snapshot` belongs to filtered telemetry. Never
join IDs between these observations. HTTP error bodies/status are retained too;
401/503 must not be interpreted as proof the endpoint is unsupported.

Since patch 0043, production telemetry retains the understood assignment and
allocation-locality fields too. `locality_adapter_valid` checks their presence
and absence of adapter validation errors in the independently selected native
observation. Existing V1 workers without these optional fields remain usable
in production; this diagnostic flag intentionally requires the new extension.

The locality summary checks extension presence, assignment/allocation/binding
references within that response, coverage vocabulary and CPU input assignment
in no-mmap. It is a diagnostic subset, separate from the production adapter's
full validation and ps policy. Malformed extension data remains in raw evidence
and fails these diagnostic checks without invalidating otherwise valid core
buffer totals. In production, the failed optional section becomes unavailable
with `invalid_metadata` and an `extension_errors` diagnostic; other sections
remain usable. Invalid/incomplete assignment evidence prevents a ps processor
claim rather than reverting to provider-based guessing. Allocation-locality
coverage never drives the CPU/GPU percentage.

Uncovered allocation paths may legitimately have unavailable locality, and
covered backends mixed with unsupported buffers may have partial locality.
These statuses are accepted independently of known capacities/placement.
The pinned 0.7.0 native series includes Vulkan, CUDA/HIP and SYCL exporters;
actual coverage depends on backend and allocation path. CPU, registration,
alias and split paths can remain incomplete. Zero native attribute
values are valid, null ranges are unknown, and DEVICE_LOCAL + HOST_VISIBLE may
coexist. No allocation or binding span is added to buffer totals.

## Bounded tensor capture (0117)

Add `--memory-tensor-diagnostics 4096` to an existing profile invocation when
using a worker that supports the flag. The default is 0: disabled, with no
native argument emitted. Values outside 0..4096 are rejected before output
creation or worker startup. This flag does not change the global fingerprint
policy; use `--expected-fingerprint b11850-0e9433bfd3` for this pinned build.

`summary.weight_storage` gives the selected observation's unique payload totals,
independent missing components, and optional tensor diagnostic status, limits,
truncation and record/range counts. Repeated tensor-role sizes are deliberately
not summed. The global budgets are N records and 16384 ranges across all owners.
A partial range prefix retains its full known tensor size but does not describe
complete geometry. The smoke accepts valid partial/unavailable diagnostic
coverage; malformed or missing requested diagnostics fail the diagnostic check.
Underlying inventory, inference and profile checks still apply.

For P100, repeat the original settings (model/projector/draft, context, K/V,
flash attention, mmap and offload); do not substitute this tool's profile
defaults for the original configuration. Extract layers 39/40 from each raw
observation separately:

```bash
jq 'select(.raw_native.body.weight_storage.tensor_diagnostics != null)
    | .raw_native.body as $s
    | $s.weight_storage.tensor_diagnostics.records[]
    | select(.layer_index == 39 or .layer_index == 40)
    | {tensor_name, component, layer_index, buffer_id, storage_status,
       storage_reason, size_bytes, ranges_complete, ranges, assignment,
       devices: $s.devices}' output/raw.jsonl
```

Names retain `byte_escape` encoding without presumed UTF-8 decoding. Resolve
device and buffer IDs only inside that same raw snapshot. Raw and selected
telemetry may have different sequences; never join their IDs. Use deduplicated
intervals for totals, and actual tensor assignment to investigate CPU weights.
Neither diagnostic sizes nor `nvidia-smi` prove physical residency or explain
the P100 case without a capture. This step does not qualify any GPU hardware.

Exit code 0 means inference, profile checks and cleanup passed. Exit code 1 means
a diagnostic failed, including unavailable/partial native inventory; the files
retain the observations. This stricter **test** criterion does not change TLlama's
production behavior when optional telemetry is unavailable. A CPU/partial test
must actually demonstrate its requested placement, not merely accept the flags.

Compare native and log totals only where both are complete. Rounded load logs
and later exact snapshots have different precision and potentially different
lifetimes; no fixed byte-difference threshold is imposed. A partial vision log is
expected if it does not report projector allocations. Projector compute may be
preallocated, so no increase after an image is required. Buffer IDs are local to
one snapshot and must not be matched across snapshots.

Display sizes use TLlama's existing Ollama projection; this direct-worker test
does not start a public API server. Raw totals retain host helpers even when the
display projection excludes them. These are logical backend capacities/spans,
not physical RAM/VRAM residency or the GPU driver's free-memory budget.
