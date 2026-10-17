# Embedded MTP configuration, state and statistics — 0051

Enable the embedded multi-token-prediction head only for a compatible model:

```toml
[runtime]
spec_type = "draft-mtp"
draft_n_max = 3
```

Add these keys to the existing runtime table, keeping its device, context, KV
and other settings. The corresponding global environment variables are
`TLLAMA_RUNTIME_SPEC_TYPE=draft-mtp` and `TLLAMA_RUNTIME_DRAFT_N_MAX=3`.
TOML takes precedence over environment. Prefer per-model configuration for
normal service use, since ordinary models may not contain an MTP head.

| Setting | Behavior |
| --- | --- |
| `spec_type` unset | No speculative CLI flags added; normal native default remains in force |
| `spec_type="none"` | Explicit `--spec-type none`; an inherited draft limit stays dormant |
| `spec_type="draft-mtp"` | `--spec-type draft-mtp`, default draft limit 3 |
| `draft_n_max=N` | Nonnegative native int32; serialized as `--spec-draft-n-max` only with MTP selected |
| `draft_n_max=0` | MTP context remains allocated, drafting disabled; not an activation proof |

The supported strategy list deliberately contains only `none` and `draft-mtp`.
No arbitrary CLI passthrough, sidecar downloads, external draft model or
automatic model-name selection is introduced. Settings are worker-load scope;
Ollama request options cannot change them and return HTTP 400 if attempted.
Changes participate in normal definition/config identity: active leases retain
their old worker, which drains, while new requests use the new configuration.

An incompatible model/backend or unknown CLI option remains a native startup
error with cleanup. TLlama never retries that load with MTP silently disabled.
Default permissive runtime fingerprint policy still permits experimental builds;
no fingerprint whitelist was added for MTP. Unconfigured ordinary inference does
not gain a new requirement for the engine to understand speculative flags.

## Verified native reference

The 0048 hardware check used the same registered Qwen3.6 embedded-MTP GGUF,
Vulkan0, context 4096 and draft limit 3, runtime `b11552-3144c167ef`.
All four off/on output comparisons matched, as did stream/full in both profiles.

| Probe | Off, tokens/s | MTP, tokens/s | Accepted/drafted |
| --- | --- | --- | --- |
| Number copy | 13.81–13.83 | 21.12–21.23 | 95/95 |
| Code | 13.77–13.81 | 18.88–18.92 | 91/107 |

These short measurements do not promise universal gains. First-token latency
was slightly higher; a three-token READY response was slower with MTP and
accepted no drafts. Cold versus warm filesystem-cache load times are not
comparable MTP performance evidence. MTP remains opt-in.

After requests the reported additional backend capacity was 778,506,324 bytes:
748,455,952 device and 30,050,372 host. It includes MTP weights/context and larger
main recurrent/output allocations, not just the head. One model owner has both
main/mtp roles; it is not counted twice. These are backend buffers, not physical
VRAM/RSS or peak memory. Existing `/api/ps` memory accounting preserves them.

Actual early native cancellation and same-PID reuse were observed. The sole
0048 review flag was a diagnostic mistake: the model consistently answered
`READY.` rather than `READY`. The corrected reference compares identical control
requests before/after cancellation, including content/reasoning and normal
finish. It still flags changed or empty control responses.

## Accepted public API check — 0049

The maintainer enabled MTP in the registered model's TOML and confirmed 46
focused tests. The public hardware archive used Vulkan0, context 4096, f16 K/V,
flash attention off and draft limit 3, runtime b11552-3144c167ef. A separately
labeled native probe measured 23 drafted / 23 accepted tokens. Public Ollama
and OpenAI chat/generate result/stream, JSON Schema, cancellation/reuse and
unload/cleanup completed. Thinking exhausted its explicit 64-token budget with
`length`; that bounded probe is not an inference failure.

The log confirms actual cancellation of task 82, release at n_tokens=25 and a
successful next request on the same PID 2337782. Seven prompt checkpoint
restores were logged. This does not cover every branch or prove that a
speculative checkpoint restore happened; native recurrent rollback can also
use its available sequence states. All sampled memory reports were known:
16,120,734,364 total bytes, including 15,645,217,304 device and 475,517,060 host.
One shared main/mtp model owner and two contexts remain correctly accounted.

### Repeating the public check if needed

Run the focused tests, then the existing public API matrix at an explicit 4096
context. The inline environment enables MTP for this test process only; model
definitions stay unchanged. An explicit TOML `spec_type="none"` still wins and
the diagnostic reports that mismatch instead of claiming MTP was tested.

```bash
pytest -q tests/test_native_mtp.py tests/test_mtp_smoke.py

mtp_results="/tmp/tllama-mtp-public-$(date +%Y%m%d-%H%M%S)"
mkdir "$mtp_results"
TLLAMA_RUNTIME_DEVICE=Vulkan0 \
TLLAMA_RUNTIME_SPEC_TYPE=draft-mtp TLLAMA_RUNTIME_DRAFT_N_MAX=3 \
PYTHONPATH=src python scripts/api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --ctx 4096 --require-mtp --log "$mtp_results/worker.log" \
  > "$mtp_results/summary.jsonl" 2> "$mtp_results/stderr.log"
tar -cJf "$mtp_results.tar.xz" -C "$mtp_results" .
```

Return the complete archive, including logs on failure. `--require-mtp` checks
effective configuration and sends a separately labeled native probe requiring
measured drafting; startup or memory ownership alone is insufficient proof.
The matrix then exercises public Ollama/OpenAI chat and generate, result/stream,
structured output, thinking, cancellation/reuse, memory, unload and cleanup.
It does not certify Tools interaction or every prefix-cache branch; those remain
the subsequent MTP checks. There is no need to repeat the accepted off/on native
reference solely for its corrected READY comparison.

Existing public token counts retain their meanings. The optional counter
extension added in 0051 is documented below. Local subprocess tests use
synthetic counters and do not replace hardware checks.

## Next hardware check: cached state and Tools — 0050

Keep the registered MTP TOML settings. First release any model loaded by the
running service so the isolated diagnostic can use the device. No model download
or native rebuild is needed. Run from the patched checkout:

```bash
pytest -q tests/test_native_mtp.py tests/test_mtp_smoke.py tests/test_tools_api_smoke.py

export PYTHONPATH=src
export TLLAMA_RUNTIME_DEVICE=Vulkan0
mtp_results="/tmp/tllama-mtp-state-tools-$(date +%Y%m%d-%H%M%S)"
mkdir "$mtp_results"
python scripts/mtp-smoke.py --state-only \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$mtp_results/state"
python scripts/tools-api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$mtp_results/tools"
python scripts/tools-api-smoke.py --clients-only \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$mtp_results/clients"
tar -cJf "$mtp_results.tar.xz" -C "$mtp_results" .
```

The last command needs the already-used optional openai/ollama SDK packages;
they remain diagnostic dependencies only. Return the complete archive even if
a command fails. The state check deliberately selects off/on over the same MTP
GGUF, overriding TOML speculation **only inside that diagnostic**. It now emits
each speculative flag once, avoiding duplicate flags after enabling MTP in TOML.
Other registered runtime settings remain inherited; context is 4096, batches
512, draft limit 3 and generation limit 128.

The state sequence checks repeated input, an extended conversation, a changed
branch and return to a shorter history. Each is compared with its own
cache-disabled control in the same worker, including exact content/reasoning,
finish reason and generated-token count. After live cancellation the MTP worker
must also reproduce both cached and cache-disabled controls. Cache reuse needs
measured cache_n > 0; the native log is required to identify checkpoint/rollback
paths. Missing reuse or changed output is retained as review evidence, not
repaired. This is a state audit, not a new speed benchmark or coverage claim for
all native branches. cache_prompt=false clears request reuse; it does not start
a new OS process. Off and MTP profiles do use separate sequential workers.

The existing Tools matrix exercises auto/required/none/named choices, parallel
calls, arguments, result replay and cancellation through public endpoints. SDK
cycles additionally check real client deserialization/replay. Both scripts
inherit production MTP settings: verify draft-mtp and a positive draft limit in
their saved argv, and actual native drafting in worker.log. Configuration alone
is not activation evidence. Known escaped-newline model mismatches may recur;
preserve their exact arguments. See [tools-api-smoke.md](../scripts/tools-api-smoke.md).

### Accepted 0050 hardware evidence

The state/Tools/SDK archive from 2026-10-17 passed all 10 cache/control
comparisons and all nine off/on text comparisons. Repeated input reused 324
tokens, conversation extension 455, changed/shorter history 283, and the
post-cancellation request 321. Content/reasoning, finish reason and output count
matched the corresponding cache-disabled controls. The MTP process logged seven
context checkpoint restores (off logged five); no speculative-checkpoint restore
branch was observed. Actual native task 348 was cancelled after two generated
tokens of a 512-token limit, then the same worker reproduced both cached and
uncached controls.

Tools passed 18/22 semantic checks and all 16 follow-ups. The four differences
are the already-recorded escaped-newline model behavior, present in stream and
non-stream on both APIs. Actual native cancellation of task 521 and same-worker
reuse passed. Both SDKs passed all four call and four follow-up steps, including
Ollama history without IDs. Saved commands and measured native draft acceptance
confirm MTP in both Tools runs. All four processes exited zero, removed private
directories and left empty residency.

## Measured public speculative statistics — 0051

When native request timings contain speculative counters, both API responses
include this optional top-level TLlama extension:

```json
{"tllama":{"speculative":{"draft_tokens":12,"accepted_tokens":6,"acceptance":0.5}}}
```

`draft_tokens` comes from native `timings.draft_n`; `accepted_tokens` comes from
`timings.draft_n_accepted`. They are cumulative counts for this request, not
worker-lifetime totals or increments. `acceptance` is accepted/drafted only when
both counts are known and drafted is positive. Each unknown field is omitted;
when both counters are missing the entire extension is absent. Measured zeros
are preserved, with no ratio for a zero denominator. Configuration, model name,
MTP owner evidence and `completion_tokens` do not substitute for measurements.

For Ollama chat/generate NDJSON, the extension occurs only on the `done=true`
record. For OpenAI chat SSE, it occurs only on the chunk carrying
`finish_reason`, independently of `stream_options.include_usage`. It is absent
from text/tool deltas, the separate usage chunk, failed or unfinished streams,
and load/unload acknowledgements. Client SDKs may ignore this custom field;
clients needing it must preserve extras or use raw JSON.

Standard Ollama `eval_count` and OpenAI usage retain the actual generated-output
count. Accepted draft tokens are already part of that output: do not add them
again or treat proposed/rejected drafts as output tokens. The extension does
not use OpenAI `accepted_prediction_tokens` / `rejected_prediction_tokens`,
which describe a different feature. Invalid native counter types, negatives or
accepted > drafted are protocol errors, consistent with existing usage checks.

After applying 0051, run:

```bash
pytest -q tests/test_speculative_stats.py tests/test_native_api.py

mtp_results="/tmp/tllama-mtp-stats-$(date +%Y%m%d-%H%M%S)"
mkdir "$mtp_results"
PYTHONPATH=src TLLAMA_RUNTIME_DEVICE=Vulkan0 python scripts/api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --ctx 4096 --require-mtp --log "$mtp_results/worker.log" \
  > "$mtp_results/summary.jsonl" 2> "$mtp_results/stderr.log"
tar -cJf "$mtp_results.tar.xz" -C "$mtp_results" .
```

Keep the model's existing MTP TOML settings. The extended `--require-mtp` now
requires measured public counters for eight longer result/stream requests:
Ollama chat, rendered/raw generate and OpenAI chat (including a usage-enabled
stream). It also retains the native probe and public cancellation/cleanup
checks. Return the complete archive on failure. Do not rerun state or Tools.
The remaining separate MTP checkpoint is explicit real-model behavior when
MTP is requested for a model without a head; no native rebuild is needed here.
