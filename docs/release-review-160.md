# TLlama 1.6.0 — scope and integration review

This release follows the native compatibility update in 1.5.1. The application
remains independent of native version numbering. It keeps the existing
`b12359-c0819bbbae` runtime and Debian minimum `1.0.0~dev+b12359.1-1`.
No native rebuild or additional backend selection is implied by this release.

## Selected scope

- Configured repository-root unification and repository-relative internal
  references, measured startup/listing cleanup, request-local TOML reuse and
  owned metadata I/O. Existing cached digests remain valid. The maintainer's
  actual repository captures establish operation reductions and limited timing
  evidence; they are not a universal latency guarantee.
- [External MTP](native-mtp.md) import/recognition/activation/tuning through the
  existing `[draft]` slot. Active external MTP precedes the integrated head;
  explicit off stays off and failures do not silently change heads. Recognition
  uses existing GGUF headers/tensor indices, not another full weight hash.
- [Per-device residency](native-residency.md) through TOML tables or the JSON
  environment list. Reservations cover loading/draining/cleanup, LRU replaces
  eligible generations, active leases drain and cleanup completes before reuse.
  Multi-device accounting is conservative; physical identity across backend
  names is not guessed. A positive global cap and device mode cannot coexist.
- Canonical assistant `reasoning` replay/output and an input-only
  `reasoning_content` alias; meaningful nested/flat [thinking controls](thinking-controls.md)
  with warnings and narrowly qualified template defaults. Unknown templates are
  not rejected from incomplete metadata. The exact profiles do not qualify an
  entire model family or prove an arbitrary model honors a requested level.
- [Offline show diagnostics](show-diagnostics.md) and honest
  [create/pull source contracts](model-management-compatibility.md), retaining
  status codes, successful request formats and existing import semantics.

The [changelog](../CHANGELOG.md#160--2026-11-17) carries the user-facing summary.

## Upgrade decisions

OpenAI-compatible chat consumers must read `message.reasoning` and streamed
`delta.reasoning` instead of the old `reasoning_content` output. Input accepts
both names only when non-null values agree. Null is absence; empty text remains
a value. Ollama chat/generate retain `thinking`. These text fields are not
official OpenAI Chat Completions fields.

Existing global-only lifecycle configuration keeps its behavior. Per-device
mode requires removing an explicit positive global cap or setting it to zero.
`CPU` is a device name, not `none`. Every device entry requires
`max_loaded_models`; a missing named match uses the unnamed entry or limit one.
Zero is unlimited. The [server example](../examples/server.toml) and
[environment example](configuration.md#tllama_device_limits) show both formats.

Upgrading a format-2 repository from 1.4.x/1.5.x requires no conversion, reimport,
cache reset or full weight rehash. Repository format 2, metadata 9, digest 2 and
cache envelope 2 are unchanged. Older unconverted repositories still require
the explicit offline converter with all writers stopped.

Sampling, KV precision, mmap, worker concurrency and default speculative length
are unchanged. Attaching a draft does not activate it. The same private UDS
transport and systemd drain/stop-before-upgrade behavior remain production
defaults; a verification-only TCP copy is not a product default change.

## Evidence and final checkpoints

The maintainer confirmed full pytest after A5/A6 and focused/integration
selections after external MTP, per-device policy and the compatibility changes.
The final full application test result, built-wheel smoke and Debian build
belong to the 1.6.0 integration checkpoint, recorded separately with their
environment and any exclusions. A TCP fallback cannot certify UDS; an absent
Debian toolchain cannot certify a `.deb` build. The package version comes from
distribution metadata, with `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TLLAMA=1.6.0`
for untagged/Git-less candidate builds. No hard-coded runtime version is added.
An explicit source manifest retains the guides, Debian files and test fixtures
even without Git file discovery. The built-wheel smoke additionally checks
per-device configuration, reasoning replay and offline management refusals.

Source/protocol tests and owned synthetic native subprocesses establish
application behavior, admission, transport and cleanup. They do not establish
real model quality, speculative acceptance/performance or all GPU combinations.
The maintainer owns operational MTP and per-device configuration checks on the
production machines and will return discrepancies for investigation. Those
checks are not replaced by another mandatory diagnostic capture here.

The existing native build and accepted earlier hardware captures remain useful
evidence with their original identities and limits. Do not repeat every backend
build or model download solely for this application integration. Before
publishing, record the final actual pytest and Debian build/package smoke
results. A prepared changelog does not assert that a tag, upload, APT publication
or deployment occurred.

## Outside this release scope

Responses/Anthropic APIs, embedding dimension reduction, logprobs/context policy,
MoE cache/FP8/tensor split, more inference slots, other operating systems and
additional model/backend qualification remain separate decisions. They are not
implicitly activated by the native update or this integration patch.
