# Native memory reporting

## Ollama-compatible process summary

`size` and `size_vram` are compatibility fields, separate from the detailed
allocation inventory in `memory`. Like Ollama, an observed full model-layer
offload is reported as `size = size_vram`, yielding **100% GPU**. CPU input
weights, projector compute, host state and staging still appear unchanged in
`memory.buffers`, `allocated_bytes` and `host_bytes`. This describes model-layer
placement, not a claim that every allocation or operation is on the GPU.

Native inventory requires complete known placement for every loaded model,
including draft/MTP weights, and actual device model buffers. Partial CPU/mixed
noninput placement uses allocation totals; incomplete/unknown inventory stays
unavailable. The verified log fallback requires a complete allocation report
and an exact full-offload layer count. Requested GPU settings alone are not
evidence. `memory.ollama_compatibility` explains the selected sizes, basis and
`excluded_host_bytes`; physical residency remains unmeasured.

The following historical 06a checkpoint describes the load-log source before
native inventory and the full-offload compatibility policy were added.

## Historical load-log checkpoint (06a)

`GET /api/ps` reports owned native generations. It performs no model loading,
header/TOML reread, hashing, inference request or lease acquisition. Polling does
not reset idle TTL. An empty manager returns HTTP 200 with `{"models":[]}`.

This checkpoint reports initial backend allocations from pinned runtime
`b11552-ec2797836f`. It does not measure physical RAM/VRAM residency. Thorough
comparisons with process/device telemetry remain a separate test campaign after
the refactor; no exact physical-memory accuracy is claimed here.

## Sources and values

| Public value | Source and meaning |
|---|---|
| `size` | Sum of reported initial model, KV, recurrent-state, output and compute buffers, including CPU mapped weights and host staging |
| `size_vram` | The same sum restricted to recognized device buffer types; host buffers are excluded |
| `context_length` | Effective worker `/props` context, or null while loading |
| `details` | Header snapshot captured when the generation was admitted |
| `digest` | Content digest already cached at admission, or an empty string; no new hash or path/mtime pseudo-digest |
| `expires_at` | Native idle TTL translated to a wall timestamp; pinned models use Ollama's finite “Forever” horizon; retirement uses the current time |
| `generation`, `state`, `pid`, `leases`, `retirement_reason` | TLlama ownership extensions; separate generations can have the same model name |
| `memory` | TLlama allocation provenance, categorized buffer bytes, coverage status/reason and runtime fingerprint |

Bytes are converted from the rounded MiB values printed by the runtime. The
`memory.precision` field is `rounded_mib`; it is not exact byte-level telemetry.
`memory.scope` is `initial_load_allocations`. Native allocations can change
during inference, especially with lazy projector warmup; these figures are not
a live high-water mark or an exhaustive process footprint.

Unlike the legacy processor-split accounting, CPU mapped input weights remain
in `size`. Host buffers remain host-side even when named CUDA_Host/Vulkan_Host.
These figures describe allocations, not where transformer layers do their work.
For a Vulkan iGPU, device allocations use shared system RAM; `size_vram` does
not identify an additional dedicated memory pool. No capacity clipping or
in-process binding telemetry is used. `physical_residency_bytes` remains null.

## Missing and partial information

The independent, bounded parser observes native pipe lines and freezes its
initial reading at the pinned `srv load_model: initializing, n_slots = ...`
marker. This avoids readiness racing a late pipe drain. Recognized model buffers
are summed (including multiple buffers of one type); repeated compute reserve
sizes replace the previous reservation. Projector compute uses a separate kind.
Subsequent inference/shutdown log messages do not recount initial allocations.

The parser requires a reported model buffer, compute buffer and KV or recurrent
state before publishing totals. Missing reports, an absent load marker,
truncated load diagnostics, malformed sizes or an unsupported runtime leave
totals null with an explicit reason. Unknown backend locations can retain a
reported total while making the device/host split null. A CPU-only measured
device subtotal is zero; a missing measurement is not zero. Known observed
buffers remain available even when the total is incomplete.

The pinned projector `load_hparams: model size` line describes GGUF tensor
sizes, not the backend's allocated weight buffer with its alignment/padding.
`memory.projector_weights` preserves this separately, sourced as
`native_model_size_summary`. A configured projector therefore leaves `size`
and `size_vram` null with `projector_allocation_unreported`. Multiple projector
contexts are also treated as ambiguous instead of summing the same summary.
Completing projector allocation coverage may require a later runtime reporting
change; 06a neither rebuilds nor modifies the installed runtime.

Parser failure never stops inference or pipe draining. Reporting errors do not
turn diagnostic uncertainty into model-load failure.

## Ownership

Ready, loading, draining and closing owned generations remain in the report
until their child is known to have exited. A draining generation stays listed
while existing requests hold leases. An already dead process is omitted even
before its exit observer completes cleanup. Editing/deleting definitions or
replacing GGUFs cannot rewrite the metadata/digest of a live generation.

Active requests suspend idle expiry. For a finite keep_alive, `expires_at` while
busy projects the idle deadline if the request finished now; polling does not
start that timer. Very large horizons are clipped for timestamp serialization
only; the residency policy is unchanged.

## Verification and hardware smoke

Unit tests cover captured Vulkan lines, CPU/CUDA/ROCm/Vulkan/SYCL classification,
host staging, multiple buffers/reservations, hybrid state, malformed/missing
reports, bounds and projector-summary uncertainty. Real subprocess/socket tests
cover lifecycle reporting, TTL, snapshot edits, overlapping generations, dead
children, parser failure and the public smoke entry point.

Run the complete pytest suite, then the existing text API smoke on each device:

```bash
TLLAMA_RUNTIME_DEVICE=none PYTHONPATH=src python scripts/api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --log api-memory-cpu.log

TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --log api-memory-vulkan.log
```

Context and K/V types come from the registered TOML (4096 and q4_0/q4_0 for
the supplied text profile). Use the text-only definition without mmproj.
Inspect `ps-ready`, `ps-during-stream`, `ps-after-stream-cancel`, `ps-draining`
and `ps-empty`. Retirement is observed under an explicit admission lease so
even a fast device cannot finish before this state check. The script checks
one PID/generation, effective context, allocation split consistency, lease
counts, post-cancel reuse and removal after unload. The CPU device subtotal
must be zero; Vulkan must show actual Vulkan buffers. These are allocation and
lifecycle checks, not a validation of physical residency or dedicated VRAM.
