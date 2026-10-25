# Native memory reporting

## Ollama-compatible process summary

`size` and `size_vram` are compatibility fields, separate from the detailed
buffer inventory in `memory`. With known `weight_storage`, SIZE uses unique
occupied weight bytes plus nonweight buffer capacities (KV, state, compute,
output, projector, adapters and other runtime components). Weight buffers are
replaced, never added to their payload. This removes sparse mmap holes and
padding without dropping runtime costs. Main/MTP shared bytes count once;
independent draft storage is included. CPU/GPU weight bytes follow interval
assignment, while nonweight buffers retain their memory-kind accounting.

Like Ollama, proved full noninput offload is reported as `size = size_vram`,
yielding **100% GPU**. Every model owner must have proved noninput accelerator
weights, with no CPU noninput interval. In that case CPU input-only payload
and host nonweight buffers are excluded from this display. Tied input/output
bytes stay because they are also output weights. CPU projector compute, input
weights and host staging still appear unchanged in `memory.buffers`,
`allocated_bytes` and `host_bytes`. Real CPU noninput weights, including a CPU
draft, keep the mixed display and include all input payload and runtime costs.
This is a storage/assignment summary, not a claim about GPU execution or
physical residency. GPU-assigned weights can use host buffers, so the derived
`size_vram` need not equal the inventory's `device_bytes`.

The new metric requires complete buffer inventory and known weight coverage,
classifiable interval assignments and nonweight memory kinds. It uses the
weight section's own proved relationships/assignments independently of the
older placement and allocation-locality collectors. A mixed CPU+GPU interval
cannot be split proportionally. A shared weight/nonweight buffer cannot be
decomposed by subtracting payload from capacity: the remainder can include
holes and padding. Such a display stays unknown while detail remains visible.
Future weight components retain their geometry but do not acquire inferred
display semantics. Present partial, unavailable or invalid weight evidence
never falls back to buffer spans or log totals.

Older V1 snapshots without `weight_storage` retain the previous buffer-capacity
metric: complete placement for every model, including draft/MTP, with actual
device model buffers required for full GPU display. Mixed placement uses
capacity totals. The verified log fallback uses its own complete allocation
report and exact full-offload layer count. Requested GPU settings alone are
not evidence. Native and log measurements are never joined.

`memory.ollama_compatibility` declares the display metric and its derivation:

| Field | Meaning |
|---|---|
| `basis` | `native_weight_storage`, legacy `native_weight_placement`, or the selected log basis |
| `metric` | `weight_payload_and_nonweight_buffers`, legacy `backend_buffers`, or log `initial_load_allocations` |
| `precision` | `exact_bytes` for native data, `rounded_mib` for load-log capacities |
| `weight_capacity_reduction_bytes` | Declared pure weight-buffer capacity minus occupied payload; 0 for the legacy capacity metric |
| `policy_excluded_bytes` | Bytes omitted by the full-offload display policy, independently of the capacity reduction |
| `excluded_input_weight_bytes`, `excluded_nonweight_host_bytes` | The two policy terms for the new metric; 0 in a mixed display |
| `reason` | Full offload, mixed totals, known empty inventory or the reason no complete display can be proved |

For a complete display, `allocated_bytes - size` equals
`weight_capacity_reduction_bytes + policy_excluded_bytes`. Unknown derivations
remain null. The misleading `excluded_host_bytes` field is replaced by these
separate terms; the difference also includes mmap holes and padding.
Detailed inventory `scope`, `precision`, sizes, buffers, assignments and
allocation locality are unchanged. Both memory and API smoke scripts consume
the same display derivation and no longer require weight-assignment GPU bytes
to equal device-buffer capacity. The client only formats server-provided
sizes and their ratio; it does not infer placement from a GGUF size or layers.

## Optional tensor diagnostics (0117)

`WorkerLaunchSpec.memory_tensor_diagnostics` is an internal diagnostic setting,
an integer from 0 to 4096. Its default 0 emits no native flag and preserves
ordinary startup with older workers. A positive value passes the explicit
`--memory-tensor-diagnostics N` argument; inherited `LLAMA_ARG_*` settings are
still removed. No daemon setting, model TOML field or general native-args
passthrough is introduced. Enabling it requires a native build supporting the
flag; this does not change any supported/default fingerprint.

When present, `memory.weight_storage.tensor_diagnostics` follows the pinned
[native weight contract](https://github.com/ByCzech/llama.cpp/blob/0e9433bfd33d8cdb194285b70c0d421d39135df2/docs/development/tllama-weight-storage.md#optional-tensor-diagnostics).
It preserves snapshot-local tensor-role visits, nullable names/components and
buffer references, native layer indices, checked assignment and exact or
explicitly incomplete geometry. Limits apply across all owners: at most 4096
records and 16384 copied ranges. Escaped names remain byte-preserving strings
in `byte_escape` form; they are not decoded as presumed UTF-8.

`size_bytes` can remain known when the range budget is exhausted;
`ranges_complete=false` means those ranges are only a prefix. `truncated=false`
does not prove complete coverage. Inspect status, missing components, storage
status, assignment and per-record range completeness. Empty proved geometry
is distinct from unavailable storage. Names and repeated visits are not storage
identities; record sizes are never summed into weight, buffer or allocation
totals, nor used for SIZE/PROCESSOR. Only the main deduplicated intervals drive
weight accounting. Invalid diagnostics withdraw only their own evidence with
an `extension_errors` path; buffers, locality and main weights remain valid.
Replacement snapshots can remove or restore diagnostics without stale records.

`scripts/memory-smoke.py --memory-tensor-diagnostics 4096` enables capture using
the existing profiles and inference/cleanup checks. Its weight summary reports
payload totals separately from diagnostic record/range counts and coverage.
Raw replies retain all native fields, including malformed diagnostic evidence;
selected telemetry is validated independently. Partial bounded diagnostics are
useful captures, not a claim that all tensors were enumerated. See
[capture instructions](../scripts/memory-smoke-08c.md) for the P100 layer check.

## Allocation locality: CUDA, HIP and SYCL (0116)

The optional `memory.allocation_locality` section describes native allocation
objects and their buffer bindings, independently of buffer capacities and
weight assignment. The reader follows the pinned
[native allocation contract](https://github.com/ByCzech/llama.cpp/blob/0e9433bfd33d8cdb194285b70c0d421d39135df2/docs/development/tllama-memory-allocations.md).
Alongside the four Vulkan properties and `level_zero.memory_ordinal`, it retains
`cuda.device_integrated`, `hip.device_integrated` and
`level_zero.device_integrated`. These require unsigned integer **0 or 1**;
JSON booleans are invalid. Absent attributes stay absent, not zero. The Level
Zero ordinal belongs to the successful direct allocation, not a Vulkan heap;
ordinary SYCL USM does not imply an ordinal. Unknown native properties remain
filtered so private handles are not exposed.

Preserve the successful API and its conservative kind/locality. HIP managed
allocation fallback can report `hipMalloc` with `kind=device`, rather than the
failed managed attempt. CUDA `cudaMallocHost` can be `host` when page locking is
unproved; SYCL host USM is `host`, and aligned allocation is `system`. A failed
host attempt returning a CPU buffer carries no failed GPU allocation record.
Managed/shared/system kinds and integrated devices do not measure page
residency. Integrated device buffers do not imply CPU weight offload. No
allocation span or binding is added to buffer capacity or weight payload totals.

Coverage describes relationship enumeration, so `known` can still have unknown
attributes. Missing CPU, registration/alias or split exporters leave their own
coverage incomplete without erasing valid buffers or weights. Invalid recognized
properties withdraw only locality with `extension_errors`; SIZE/PROCESSOR
continue to use independent weight evidence. These parser tests do not qualify
native allocation/free paths or GPU inference; hardware checks remain separate.

## Occupied weight storage (0114)

The optional V1 `weight_storage` section describes occupied model-weight bytes
inside backend buffers. It excludes padding and holes, including sparse mmap
spans. Quantized sizes include their encoded block and scale bytes. The parser
follows native contract
[`0e9433bfd3`](https://github.com/ByCzech/llama.cpp/blob/0e9433bfd33d8cdb194285b70c0d421d39135df2/docs/development/tllama-weight-storage.md).
These are storage bytes, not resident RAM/VRAM or another allocation to add to
the buffer inventory. Projector, adapters and runtime state are outside this
weight scope and remain in their existing inventory.

`memory.weight_storage` retains the validated native evidence.
`memory.weight_storage_totals` is a separate derived summary of the entire
native model-weight scope, including independent drafts:

| Field | Meaning |
|---|---|
| `status` | Native weight coverage, independently of buffer/placement/locality coverage |
| `payload_bytes` | Unique occupied bytes, null unless weight coverage is known |
| `cpu_bytes`, `accelerator_bytes` | Complete assignment split, null if coverage or any selected interval's assignment classification is incomplete |
| `known_bytes` | Sum of the proved intervals, including unclassified bytes; a partial subtotal is not a complete model size |
| `known_cpu_bytes`, `known_accelerator_bytes` | Subtotals with proved homogeneous CPU/accelerator assignment |
| `unclassified_bytes` | Proved storage with partial, unavailable, CPU+accelerator, meta or unknown assignment; never distributed proportionally |
| `interval_count` | Number of unique selected intervals |

Assignments resolve devices in the same snapshot. Neither provider device IDs,
buffer memory kind nor allocation locality overrides them. Multiple accelerator
devices do not multiply one interval's bytes or distribute them among GPUs.

The internal `NativeMemoryReport.weight_totals()` can filter exact owner IDs,
owner roles and relationship components. It matches owner and component on the
same relationship, then counts each selected interval once. A shared main/MTP
owner has both roles; selecting either or both does not duplicate its storage.
Separate buffer objects remain separate even with identical names or contents.
For tied input [0,8) and output [4,12), the native partition is [0,4), [4,8),
[8,12): the union is 12 bytes and excluding input-only bytes leaves 8. Role and
component subtotals can overlap and must not be added. Filtering never upgrades
incomplete global coverage to a complete filtered total. Future role/component
names are retained and matched literally, without inferred display semantics.

The reader validates uint64 offsets, sizes and their sum, nonempty ranges,
known buffer bounds, per-buffer ordering/disjointness, snapshot-local IDs,
model ownership, native layer indices and the shared assignment contract.
An invalid optional section is replaced by unavailable evidence with
`invalid_metadata` plus `extension_errors`; valid buffer, placement and
allocation evidence survives. Absence remains unsupported (`weight_totals()`
returns `None`), never a known zero. A proved known empty section after unload
reports zero. Cached reports, comparisons and summaries are copied together;
failed reads and replacement do not carry old weight evidence forward.

Parser/detail support was added in 0114. The payload-based display accounting
above follows in 0115.
Tensor diagnostics are still ignored by the filtered reader until their own
parser is added; the existing raw smoke capture retains them independently.
Neither step changes runtime fingerprints or qualifies native GPU backends.

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
