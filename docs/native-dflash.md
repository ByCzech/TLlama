# External DFlash draft models

A main definition can attach one imported draft without activating it:

```toml
[llm]
model = "data/TARGET_HASH/weights.gguf"

[draft]
model = "data/DRAFT_HASH/weights.gguf"

[runtime]
spec_type = "draft-dflash"
draft_n_max = 3
```

Replace the illustrative hash paths with the actual repository paths. A
standalone draft can be attached with `tllama cp SOURCE DEST --draft HELPER`;
then edit the destination's runtime table to select the strategy. The copy
operation does not silently enable speculation. Local composite imports use
`[draft].from` through `create`, exactly like projector imports.

Only one strategy is selected: `none`, embedded `draft-mtp`, or external
`draft-dflash`. DFlash and DFlash2 exports both report architecture `dflash`;
their tensor formats are handled by the native loader, not an invented
`draft-dflash2` setting. Use separate virtual main definitions to choose between
two draft artifacts while sharing the same target weights.

| Effective setting | Native launch |
| --- | --- |
| `draft-dflash` with an imported draft | `--spec-type draft-dflash --spec-draft-model PATH --spec-draft-n-max N` |
| `none` or an unset strategy | No external draft path; no draft weights/context requested |
| `draft-mtp` with a declared external draft | Embedded MTP only; no external draft path |
| Request `draft_num_predict=0` | Selects `none`; omits external draft path and native draft limit |
| Positive request `draft_num_predict=N` | Adjusts the configured active strategy's length; with no active strategy retains the existing MTP fallback |

An omitted request inherits the model configuration. The request does not
rewrite TOML or stored model identity. `draft_n_max=0` is just a native limit;
use `spec_type="none"` or request `draft_num_predict=0` to disable speculation.
Missing bindings/files/shards and a known non-`dflash` architecture fail before
worker startup. Unknown metadata is not proof of pair compatibility: the native
loader validates target/draft tensors and tokenizer compatibility. Startup
failure is reported without retrying under a different strategy.

## Captured native contract and initial defaults

The maintainer supplied `--version` and speculative `--help` output on
12 November 2026 for `0.7.0-dev`, build 11850, commit `0e9433bfd3`, GNU 14.2.0,
Linux x86_64. It confirms the above long flags and native draft-length default
**3**. TLlama emits that length explicitly when no limit was configured.
It uses neither removed `--draft`/`--draft-max` flags nor native HF download
presets; workers remain offline and inherited `LLAMA_ARG_*` settings are removed.

Omitted tuning keys preserve those captured native defaults: draft devices
follow `--device`, draft GPU layers are `auto`, draft K/V cache types are `f16`,
and draft physical batch size follows the main `--n-ubatch`. Main KV type
overrides do not override the draft KV types. Native fitting receives the draft
path and configured draft tuning in the same command as the
target, including automatic backend attempts. Runtime fingerprint warning/strict
policy remains unchanged; ordinary inference gains no mandatory new CLI flags.

## External draft tuning

These optional keys belong to the main definition's existing `[runtime]` table:

```toml
[runtime]
spec_type = "draft-dflash"
draft_n_max = 3
draft_device = "none"
draft_n_gpu_layers = 0
draft_type_k = "q8_0"
draft_type_v = "q8_0"
draft_n_ubatch = 128
```

This is an illustrative CPU draft configuration, not a performance recommendation
or a certification that a specific DFlash model supports quantized KV. The main
device, main KV cache and main ubatch remain independently configured.

| Key | Accepted value | Native flag / default |
| --- | --- | --- |
| `draft_device` | Native device selector, CSV selectors, or `none` | `--spec-draft-device`; inherits main device when unset |
| `draft_n_gpu_layers` | `"auto"`, `"all"`, -1 (=all), or nonnegative int32 | `--spec-draft-ngl`; native `auto` when unset |
| `draft_type_k` / `draft_type_v` | Same KV names/enums as main cache types | `--spec-draft-type-k` / `--spec-draft-type-v`; f16 when unset |
| `draft_n_ubatch` | Nonnegative int32; 0 follows main physical batch size | `--spec-draft-n-ubatch`; native 0 when unset |

The controls use the normal server TOML/environment/model TOML precedence;
environment names are `TLLAMA_RUNTIME_DRAFT_DEVICE`,
`TLLAMA_RUNTIME_DRAFT_N_GPU_LAYERS`, `TLLAMA_RUNTIME_DRAFT_TYPE_K`,
`TLLAMA_RUNTIME_DRAFT_TYPE_V` and `TLLAMA_RUNTIME_DRAFT_N_UBATCH`.
Model `draft_n_gpu_layers="auto"` and `draft_n_ubatch=0` can reset a global
numeric override. Invalid types, unsupported KV names and out-of-range integers
are rejected even when dormant. Active explicit `draft_device="none"` accepts
only layer `auto` or 0; positive/all GPU layers conflict with that explicit choice.

These external controls are omitted from native argv for `none`, an unset
strategy and embedded MTP. They do not activate speculation or change MTP's
settings. HTTP inference `options` rejects these load keys, as it already rejects
direct `spec_type`/`draft_n_max`; use configuration rather than new API options.

Automatic main-backend attempts retain explicit draft selectors and layer
settings. Omitted draft selectors continue to follow each main attempt, including
CPU fallback. An explicit draft selector is never rewritten to another GPU/CPU
to make an attempt work: native startup must report unavailable devices or
unsupported settings. Real memory fit/performance still needs hardware testing.

## Identity and lifetime

All declared artifacts contribute to stored model SIZE/ID and repository
references even when speculation is off. Names in `.inf` and filesystem paths
do not change content identity. The implicit draft limit and explicit 3 have
the same identity; a different active limit contributes to it. External draft
tuning contributes only when DFlash is selected. KV names/enums, all/-1 and
explicit native defaults are normalized; dormant tuning does not change stored ID.

The active worker snapshot includes all draft shard fingerprints. Rebinding a
draft or replacing a shard selects a new generation; existing requests keep the
old generation. Only active draft groups are pinned for inference, together with
the main/projector/adapter groups. Whole groups remain protected through worker
cleanup, including a child retaining inherited reservation descriptors after
loss of the supervisor's descriptor. Inactive draft data is retained by TOML
references, not by a worker that never opens it. Identity/validation may still
read inactive artifacts; this is not loading them into native inference memory.

Automated tests cover argv, configuration, public request selection, generation
ownership, group cleanup, shards and failures using GGUF headers and synthetic
native processes. They do not qualify real DFlash/DFlash2 generation or GPU
memory accounting. Native memory-report/log audit and hardware qualification
with compatible target/draft pairs remain follow-up work.
