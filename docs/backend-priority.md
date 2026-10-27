# Automatic backend priorities

For explicit per-model selectors and requested mixed-backend lists, see the
[placement guide](features.md#model-placement-across-devices-and-backends).
Automatic selection below keeps one backend family per attempt.

When neither model `[runtime].device` nor `TLLAMA_RUNTIME_DEVICE` is set,
TLlama selects native devices at model load. Daemon startup does not probe GPUs
or load models. Failed model startup cleans its child and leaves the daemon
available for other requests. Explicit placement bypasses this policy entirely,
even when the device's backend is absent from the list. An explicit device
failure does not trigger a different device or CPU retry.

```bash
export TLLAMA_BACKEND_PRIORITY="CUDA,ROCm,SYCL,Vulkan"
```

That is the default for an unset variable. An empty value is CPU-only, unlike
the empty-means-default convention of older scalar settings. `CPU` is terminal:
`CPU,Vulkan` selects CPU immediately; `Vulkan,CPU,ROCm` never tries ROCm.
If no CPU entry is present, CPU is the final implicit candidate. CPU uses native
`device=none`, zero GPU layers, disabled operation/KV/projector offload, and
retains available native CPU optimizations and helper libraries such as BLAS.
No attempt is made to select BLAS, BLIS, OpenBLAS, MKL or ZenDNN independently.

Names are case-insensitive with trimmed whitespace. HIP aliases ROCm, MTL aliases
Metal and HTP aliases Hexagon. Duplicates retain their first occurrence. Invalid
entries/CPU helpers warn and are skipped; a list with no remaining entries means
CPU, never restoration of the default GPU allowlist. Programmatic BackendConfig
also accepts an ordered sequence through its `backend_priority` field. This is
a global policy; per-model placement continues to use `[runtime].device`.

## Discovery and selector matching

An owned `--list-devices` subprocess uses the exact worker executable and cleaned
child environment, including visibility settings. No model/key/listener arguments
are supplied. The probe has a combined 4 MiB output cap and a deadline of the
smaller of 10 seconds and one quarter of the model load deadline. Cancellation,
overflow, timeout and normal completion reap the direct child, terminate its
process group and remove its private cwd. Unsupported/malformed/duplicate tables
warn and select CPU. Discovery does not require an audited commit fingerprint;
the existing standalone 0070 evidence diagnostic retains its independent guard.

Only actual selectors from a complete native stdout table are used, retaining
their spelling and order. One attempt groups the devices matching one backend;
there is no cross-backend device combination, physical deduplication, capacity
sum, architecture table or ranking by GPU model. This is not proof of working
kernels, stable physical identity or optimal placement.

| Policy name | Native selector convention in the audited base |
| --- | --- |
| CUDA / ROCm / SYCL / Vulkan / CANN / MUSA / ET | Name plus decimal index |
| Metal | MTL plus decimal index |
| Hexagon | HTP plus decimal index, optionally colon and virtual index |
| OpenCL | GPUOpenCL |
| OpenVINO | OPENVINO plus index; actual native target still depends on GGML_OPENVINO_DEVICE |
| WebGPU / zDNN | Exact name |
| RPC | RPC plus index; endpoint configuration is independently required |
| VirtGPU | Literal `[virtgpu] ` followed by the remote selector |

This table describes selector matching, not integration or successful inference
certification for every backend. A new unknown policy name can match an exact
selector or that literal name followed only by ASCII digits. No arbitrary prefix,
regex, punctuation extension or hardware-name inference is attempted. A matching
unknown family warns that registry identity is unverified; an unmatched unknown
name warns and is skipped. This allows usual new naming conventions without a
Python whitelist, but unconventional new names require a matcher update.

## Startup attempts and limits

Attempts are sequential within one admitted residency generation and one overall
model load deadline. A failed child must be cleaned before another is created.
Known selected-accelerator failures (for example ROCm missing kernel image or
Tensile library, or failure to initialize the selected Vulkan backend) advance
to the next candidate and ultimately CPU. Diagnostic warnings from unrelated
backends are not sufficient retry evidence. Generic failures, model corruption,
out of memory, protocol/version errors and unclassified startup timeouts are
reported without trying every backend. CPU can also fail: missing model files,
insufficient RAM or invalid model settings remain real errors.

Successful startup remains authoritative only for that worker. No inference
request is replayed and no backend switch occurs after readiness or during a
stream. Default native warmup is not disabled; failures that appear only in a
later real computation still fail that request. Logs record candidate, failure
and final selection. There is no global blacklist of a GPU based on one model.

For automatic GPU placement only, an unset GPU layer count uses native
`--n-gpu-layers auto --fit on`. Explicit GPU layer counts retain their existing
meaning and disable fitting; CPU fallback necessarily sets the offload count to
zero. Context, K/V types, FA, batches and registered definitions are not rewritten.
The audited parser preserves an explicitly passed positive context and treats
`--ctx-size 0` as the trained maximum with context reduction disabled. Fitting
is native estimation, not a guarantee that every model fits; no TLlama memory
scheduler or context-shrinking retry is introduced.

`scripts/worker-config.py` remains offline: automatic selection produces
`argv: null` with the priority list, rather than pretending that discovery ran.
Explicit placement still prints its exact illustrative argv.

## Validation

```bash
pytest -q tests/test_backend_priority.py tests/test_native_config.py \
  tests/test_worker_config.py tests/test_native_manager.py \
  tests/test_native_residency.py tests/test_native_public_api.py \
  tests/test_native_decision.py tests/test_config_from_env.py \
  tests/test_runtime_overrides.py tests/test_device_inventory.py
```

The subprocess tests use synthetic devices and controlled native failures.
Actual hardware acceptance must inspect automatic selection on the maintainer's
installed runtime, especially the ROCm-to-Vulkan path for their unsupported gfx
target. No native patch, GPU model download or physical-inventory helper is
required. Native discovery patch 0074 is deferred for this policy.
