# External DFlash / DFlash2 evidence capture — 0210

Run from the patched checkout with a **known compatible target and draft pair**.
The script takes physical GGUF paths, not registered model names. It does not
create/edit model definitions, import weights, download a model or contact the
running TLlama daemon. It starts two owned, offline native processes sequentially
through the production `NativeWorker`: explicit `none`, then `draft-dflash` with
the external draft path. The first child is reaped before the second starts.

Run under an account that can read the checkout and both complete GGUF groups,
write the capture directory and use the repository's runtime pins. For a
daemon-owned model store this is normally the `tllama` service account, as with
the repository converter. Running as root is not a substitute for that account.
Managed groups receive normal temporary worker reservations; inactive draft
weights are not loaded by the off worker. Keep capture output outside `models/`.

Replace both placeholder paths and choose the actual native device selector:

```bash
PYTHONPATH=src python3 scripts/dflash-qualification-smoke.py \
  --model /var/lib/tllama/models/data/TARGET_HASH/weights.gguf \
  --draft-model /var/lib/tllama/models/data/DRAFT_HASH/weights.gguf \
  --device Vulkan0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --output /tmp/tllama-dflash-capture-01
```

Use `--dry-run` first to inspect the argv. It reads no GGUF, starts no worker
and creates no output. The normal run validates complete shard groups using
the existing importer header checks before starting a child. Supply the first
shard for each split artifact. A new output directory is required; existing
evidence is never overwritten. An optional `--executable` selects another native
binary; its **exact** expected fingerprint is mandatory and enforced strictly.
No daemon runtime/sampling defaults or inherited `LLAMA_ARG_*` switches select
another method. `CUDA_VISIBLE_DEVICES` and equivalent permitted backend visibility
settings still apply to both children through the normal worker contract.

Defaults: context 4096, 128 output tokens, draft limit 3, main K/V f16, flash
attention on, mmap on, main batch/ubatch 512, all main GPU layers (CPU uses 0).
`--device none` can exercise a CPU pair. Draft placement/KV use native defaults
unless explicitly set with `--draft-device`, `--draft-gpu-layers auto|all|N`,
`--draft-type-k`, `--draft-type-v`, `--draft-ubatch` and `--draft-n-max`.
These use the production typed CLI mapper and appear only in the active profile.
The script requests a fixed device, not automatic backend fallback. `--no-mmap`
is an explicit diagnostic workaround applied and recorded in both profiles.
Timeouts and optional bounded `--memory-tensor-diagnostics 0..4096` are explicit.
Do not assume every draft supports every KV type or draft length; the native
loader and the particular trained draft constrain those choices.

## Automated checks

- Exact native build; one short control and number-copy/code probes in both
  streaming modes, temperature 0, seed 42. Each completed request must report
  measured `cache_n=0` for its requested `cache_prompt=false`. Missing cache
  evidence or actual reuse cannot pass as a comparable uncached reference.
- The off profile has no draft model/context owner or draft weight payload and
  no positive draft counters. The active profile has an independent draft model
  owner and measured drafted **and accepted** tokens in each streaming mode.
  A filename, launch flag or allocated draft alone cannot prove activity.
- Raw authenticated native memory and independently selected telemetry after
  load, inference and cancellation: 12 snapshots in a successful full capture.
  Inventory and weight coverage must be known and classifiable. Role subtotals
  are not added: shared intervals count once, separate draft storage counts
  independently. Requested main/draft placement checks use each model's exclusive
  intervals; borrowed shared target weights need not move to a CPU draft's device.
  These are storage assignments, not a percentage of GPU execution. Global
  interval coverage must include all model weights. Missing/invalid inventory
  fails; the partial external-draft log fallback cannot pass in its place.
- Closure of a live long stream, followed by a valid control request using the
  same worker PID and stable weight payload. Native early task stop is a manual
  log check, not inferred from transport closure or PID reuse.
- Cleanup on success and failure, preserving the original startup/protocol error
  if cleanup also reports a problem. Stat identity of the binary and **all**
  input shards is checked before/between/after profiles. This is a stability
  check, not a fresh full-content hash or proof against malicious file changes.

## Evidence and interpretation

The capture retains `result.json`, `summary.jsonl`, `raw.jsonl`, `worker.log` and
`off-load.log` / `dflash-load.log`. Startup failures still retain available logs
and cleanup evidence. Invalid CLI/header inputs fail before output creation;
their error is printed as JSON. Raw requests, responses, assembled text/reasoning,
native timings, usage and local elapsed/first-text times remain available.
Native IDs are snapshot-local and are never joined between observations.

Exit 0 and `ok=true` mean the **automated protocol/telemetry checks** passed.
They do not certify model quality, native early cancellation or the provenance
of an export. `review_required` therefore remains true, with `manual_review`
listing those tasks. Exact text/reasoning, finish and token-count differences
within streaming pairs, off/on pairs and cancellation controls are retained in
`observations`, rather than automatically branded a model defect. Inspect the
actual answers, timing distributions and native logs; deterministic inputs do
not guarantee bit-identical quantized-backend output or speedup.

## Prefix-cache state capture

Add `--state-only` to the same invocation, with a new `--output` directory, to
replace the copy/code reference probes with a state sequence in each fresh off/on
worker. Launch flags and all memory, identity, counter and cleanup checks remain
the same. No daemon definitions or public API options are changed.

The sequence compares a cached repeat, an extended conversation with the actual
first answer, a changed branch, and a return to the shorter base against each
path's own cache-disabled control. It records native `cache_n`; cold controls
must report zero, and every completed state request must provide cache telemetry.
After interrupting a live cached stream, it compares both a cached recovery and
another cold request against the original base. Memory/weight coverage is still
checked after those operations. A positive `cache_n` proves reported reuse, not
checkpoint restoration or correctness of internal target/draft rollback.

Text, finish or output-count differences are observations requiring review;
they are not automatically a model defect. No positive reuse is also explicitly
reported for review: a run with all cache counts zero has not qualified reuse,
even when protocol checks pass. Missing telemetry or nonzero cache in a cold
control fails the automated checks. `manual_review` additionally lists
`native_cache_rollback`; inspect raw answers and the native log. This mode does
not enable context shifting or certify speculative context-shift support.

This diagnostic exercises the private native protocol, **not public HTTP API**.
Public DFlash adapter/client transport is separately covered by regression tests.
It does not qualify tools, vision, FIM or arbitrary pairs. Reference mode does
not exercise prefix-cache reuse; state mode supplies evidence for manual review.
DFlash and DFlash2 use the same native method; run separate captures with the
actual compatible exports. The script does not infer the variant from its name
or certify both formats after one successful run. Keep the full capture for
review, for example:

```bash
tar -cJf /tmp/tllama-dflash-capture-01.tar.xz \
  -C /tmp/tllama-dflash-capture-01 .
```

If the capture belongs to the service account, archive it under that account
as well. Test this harness without real inference hardware using:

```bash
pytest -q tests/test_dflash_qualification_smoke.py tests/test_dflash_cache_context.py tests/test_native_dflash.py
```
