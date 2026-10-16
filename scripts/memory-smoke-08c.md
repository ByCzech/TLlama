# 08c: first hardware memory checks

Working test instructions; final user documentation remains deferred.

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

python scripts/memory-smoke.py --profile cpu \
  --model "$memory_model" --output "$memory_results/cpu"

python scripts/memory-smoke.py --profile partial --device Vulkan0 --gpu-layers 20 \
  --model "$memory_model" --output "$memory_results/partial"

python scripts/memory-smoke.py --profile vision --device Vulkan0 \
  --model "$memory_model" --mmproj "$memory_projector" --output "$memory_results/vision"

python scripts/memory-smoke.py --profile no-mmap --device Vulkan0 \
  --model "$memory_model" --output "$memory_results/no-mmap"
```

Each output directory must be new; existing results are never overwritten.
Add `--dry-run` to inspect the launch command without starting a worker or creating files.
`TLLAMA_RUNTIME_EXECUTABLE` / `--executable` and the version policy are honored.
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
- `raw.jsonl`: full independent inventories and all summary records.
- `worker.log`: redacted native/debug logs for analysis if a check fails.

For initial review, send the four `summary.jsonl` files. Keep the raw files/logs
for any discrepancy. The stdout output is the same as the summary file.

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
