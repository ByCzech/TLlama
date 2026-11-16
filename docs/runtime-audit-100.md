# Native 1.0.0 integration — TLlama 1.5.1

Reviewed 16 November 2026 against native commit
`c0819bbbaeafa81e506eb83cf1b43f9f6106d307`, tree
`4e951fcb49275d0c70b539d7d5d01525d663c746`, fingerprint
**b12359-c0819bbbae**, Debian runtime `1.0.0~dev+b12359.1-1`.
The upstream base is llama.cpp v1.0.0 / b12359,
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, with the project's native changes.
The maintainer has built the complete core/CPU/Vulkan/ROCm/CUDA/SYCL set.

This is an application compatibility update, not a new feature release or a
claim of complete GPU/model qualification. Native build evidence, inspected
source, controlled subprocess tests and actual model captures have different
scopes. The historical [0.7.0 audit](runtime-audit-070.md) remains unchanged
as evidence for its original builds and hardware.

## Admission and retained contracts

| Contract | Current build | Retained older builds |
| --- | --- | --- |
| Generation and embeddings | b12359-c0819bbbae | b11850-0e9433bfd3; b11552-ec2797836f; b11552-d1570abd3d; b11552-3144c167ef |
| Text Decision: Laya, Nimble, Clef | b12359-c0819bbbae | b11850-0e9433bfd3; b11552-3144c167ef |
| Discovery version/table grammar | b12359-c0819bbbae | b11850-0e9433bfd3; b11552-3144c167ef |
| Initial load-log coverage | Not promoted by this update | Existing independent log audit set remains unchanged |

`warn` and `strict` keep their existing meanings. An explicit diagnostic pin
still selects exactly one build, including when a different build is otherwise
supported. Changing the ordinary worker default does not add older ordinary
builds to the narrower Decision set. All modes still validate actual health,
alias, slot count, effective context and, for Decision, output modalities.
An accepted identity never bypasses typed output or Memory API validation.

Source review of the exact fork pin covered:

- Existing CLI construction in `common/arg.cpp` and `common/common.h`:
  local/offline authenticated workers, device/layer/fit/context, one slot,
  sleep policy, Jinja/templates, KV/load flags and the explicit speculative
  options already used by TLlama. Native draft maximum remains 3.
- `tools/server/server-context.cpp`: `/props` build_info, model_alias,
  total_slots and default_generation_settings.n_ctx; model modalities and
  the existing text/tool/embedding routes. The new revision retains the
  private worker contract; no public TLlama routes or fields are added.
- `server-task.cpp`, `server-chat.cpp` and `common/chat.cpp`: completion/chat
  message and SSE envelopes, finish reasons and timings used by the adapter.
- `server-decision.cpp`: the existing typed text Decision operation. The
  Laya/Nimble/Clef metadata and live-mode checks remain in force; this update
  does not admit additional heads or image/audio Decision inputs.
- The project's `server-memory.cpp` and snapshot path: Memory API V1 remains
  exact-byte buffer inventory with owners, devices, placement, optional weight
  storage and locality. No V2 endpoint or schema migration is introduced.

These observations are source compatibility evidence. They do not establish
the numerical quality of a model or successful kernels on every backend.

## Memory coverage: do not certify logs by changing a build number

The native Memory API remains the normal authoritative source. Valid partial
or unavailable native coverage is retained; it is not replaced by apparently
complete legacy log totals. Unknown schema, identity or ownership references
still follow the existing failure and fallback rules.

The new source contains additional `layer_inp_dev_alloc`, `h_nextn_dev_alloc`
and KV rotation allocations. They are not all part of the old rounded-MiB
load grammar. The parser now marks an observed unhandled layer-input/NextN
allocation as incomplete, as it already does for unknown KV buffer lines.
It must not quietly omit such a buffer and publish a complete total.

The new runtime is deliberately not added to `_LOG_RUNTIME_FINGERPRINTS` by
this patch. `runtime_verified=false` on the log observation therefore does
not mean the ordinary worker or its native V1 report failed admission.
It means the independent log-only coverage has not been qualified. Historical
0.7.0 replay tests explicitly use their original identity; they are not
relabeled as new-build captures. Fully accounting for new dynamic buffers
through log parsing would need its own evidence; it is not required to use
the exact native inventory.

## Automated checks and actual model smoke

Regression tests exercise the current and older identities through real
authenticated TCP stub processes, strict rejection, unchanged live-mode
checks, discovery drift and new-build Memory API V1 selection. Those processes
test the Python transport and parser; they are not the native binary and do
not prove GPU inference or speed. Source-derived additional allocation lines
are also synthetic test inputs, not measured capacities of a real model.

Representative maintainer checks with the actual installed binary should
cover an existing text model, tools/schema, vision/projector, embeddings,
text Decision and already integrated MTP/DFlash(2), as available. Reuse the
existing qualification tools and supply this exact candidate fingerprint:

```sh
/usr/lib/tllama/server/tllama-server --version
PYTHONPATH=src python3 scripts/device-inventory.py \
  --executable /usr/lib/tllama/server/tllama-server \
  --expected-fingerprint b12359-c0819bbbae --output /tmp/tllama-151-devices
```

Every capture output directory must be new. For registered text/tools and
embedding models, `functional-qualification-smoke.py` owns its private daemon
and workers, reads definitions and writes evidence/caches into its capture
area. It does not edit the definitions or require stopping the normal daemon;
ensure enough memory or stop idle production models to avoid competing loads.
See the [development guide](development.md) for the existing smoke tools.
MI50/HIP tests may retain the existing no-mmap recipe; P100 still needs CUDA
12.x. Do not change FA, KV, mmap, placement or speculation defaults merely
because the new binary offers additional settings.

## Release boundary

TLlama 1.5.1 adopts this available runtime and keeps current application
behavior. External MTP management, A5/A6, startup/cache optimization,
per-device quotas, MoE cache, FP8 and new HTTP APIs are later work. Existing
DFlash/DFlash2 and embedded MTP remain the supported speculative modes.
Repository format, content hashes, .inf companions and cache versions are
unchanged. No conversion, reimport or deliberate rehash is required.
