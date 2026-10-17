# Embedded MTP native reference and state audit — 0050

This reference measures the native engine independently of production MTP TOML
settings or public statistics. Its reference and basic public hardware results
are accepted; for the next state/Tools check use
[native-mtp.md](../docs/native-mtp.md). It uses the registered model containing
embedded MTP tensors. No sidecar draft model, download or native rebuild is
requested. The regular non-MTP GGUF is not an equivalent off baseline.

Run from the patched checkout:

```bash
pytest -q tests/test_mtp_smoke.py

export PYTHONPATH=src
export TLLAMA_RUNTIME_DEVICE=Vulkan0
mtp_results="/tmp/tllama-mtp-reference-$(date +%Y%m%d-%H%M%S)"
python scripts/mtp-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$mtp_results"
tar -cJf "$mtp_results.tar.xz" -C "$mtp_results" .
```

Return the complete archive. Choose a new output directory for another run;
existing evidence is never overwritten. Do not edit model definitions for this
test. `--models-dir` and `--executable` can select another store/binary. The
normal runtime version policy remains in force; an unknown fingerprint alone
is not refused under the default warning policy.

## What runs

- Sequential fresh private workers over **the same GGUF**: `--spec-type none`,
  then `--spec-type draft-mtp --spec-draft-n-max 3`. The latter flags are added
  by a diagnostic-only launch subclass. Existing production TOML speculation
  fields are excluded from this launch before adding the selected profile, so
  flags are emitted once even when the registered model now enables MTP.
  Native `--mtp` is a download convenience, not used for this offline model.
- Registered device, weight/KV types, flash attention and template stay in force.
  Context is explicitly 4096, batch/ubatch 512; one native slot. TOML and native
  build identity must match between profiles. Embedded MTP creates its context
  against the target model instead of loading a second copy of the GGUF.
- One short warm-up, then number copying and code generation, each result/stream.
  Same inputs, temperature 0, seed 42, max 128 generated tokens, thinking disabled,
  `cache_prompt=false`. Native measured cache counts are still recorded.
- Native timings/counters, wall elapsed and first nonempty stream delta timing.
  TTFT includes text or reasoning, excludes empty role headers; non-stream TTFT
  is unknown. Native usage and raw text remain available for review.
- Explicit native memory reads after load/requests and cancellation, preserving
  status, owners and deduplicated totals. A shared model/compute buffer is counted
  once by the existing memory adapter; owner counts are not byte multipliers.
- With MTP on, close a long stream after its first live text/reasoning delta,
  then request READY on the same PID/generation. Shutdown each worker, remove its
  private directory and verify empty residency before the next profile.

`--predict`, `--draft` and timeouts permit a controlled later sweep. Start with
the defaults; this is a functional reference, not a statistically significant
performance benchmark. Avoid concurrent inference when comparing timing.

## Reading the evidence

`raw.jsonl` retains requests, all decoded native frames, props, argv, full text,
usage, timings and memory telemetry. `summary.jsonl`/stdout omit bulky raw data.
`worker.log` captures runtime diagnostics; separate `off-load.log` and
`mtp-load.log` preserve load output, including failures.

`draft_n > 0` is evidence that native drafting occurred. The acceptance ratio is
`draft_n_accepted / draft_n` only when both measured counters exist. Missing is
unknown, never a fabricated zero. Zero acceptance can be a valid measurement.
The MTP memory owner is independent corroboration of context/tensor allocation;
successful startup or the requested CLI flag alone proves neither drafting nor
a speedup. No draft counters/MTP owners are reported as review observations.

Compare content, token counts, generation rate and stream/full consistency.
Greedy output differences are preserved for review rather than repaired or
treated as automatic proof of a public adapter bug. Short-run TPS ratios have
no acceptance threshold; a working MTP implementation need not be faster for
every backend, model or draft length.

`protocol_ok` controls exit status; `review_required` must also be inspected.
Text differences or missing activation evidence require review but retain all
results. Native errors, broken stream termination, invalid counters or cleanup
failures produce nonzero exit status. A missing head or unsupported runtime must
remain an explicit error; the diagnostic does not silently disable MTP.

Review actual early native cancellation in `worker.log`. Connection closure and
a successful follow-up alone do not prove the long task stopped before its
512-token budget. The local synthetic subprocess test proves harness ownership,
transport, counter validation and cleanup, not real model quality or GPU speed.

0049 corrects the control check to compare the same request's actual reply
before/after cancellation, including reasoning and normal completion. A stable
`READY.` response is valid evidence; punctuation is not stripped or repaired.
The accepted 0048 archive already contains identical control replies, so that
original review flag needs no repeated hardware run.

## Source baseline and next gate

Inspected native revision `3144c167ef703e93cee0b58d7236b6a28c6e50ad`:
`common/arg.cpp` (`--spec-type`, `--spec-draft-n-max`), `common/common.cpp`
(automatic MTP tensor loading), `common/speculative.cpp` (embedded target/MTP
context reuse), `tools/server/server-context.cpp` (activation/counters) and
`tools/server/server-common.cpp` (`draft_n`, `draft_n_accepted` in timings).

Typed per-model MTP settings and basic public API checks were added in 0049.
0050 adds `--state-only`, replacing the original prompt pair with repeated,
extended, changed-branch and shortened inputs, each compared with its own
cache-disabled control. MTP additionally checks the cached/fresh control after
live cancellation. `cache-comparison` retains hash/finish/token agreement and
`state-evidence` records measured reuse. Inspect worker.log for the native
checkpoint path; positive cache_n alone is not proof of a checkpoint restore.
All original transport, counter, memory and cleanup checks remain in force.

Public optional statistics and a negative real-model path remain later work.
Shared-memory locality exporters for
CUDA/HIP/SYCL/CPU remain a separate native follow-up; existing buffer sizes and
placement work without them and do not block this MTP reference.
