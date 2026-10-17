# Embedded MTP configuration — 0049

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

## Next hardware check: public API with production settings

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

Optional draft counters currently remain in native diagnostic evidence. Existing
public token counts retain their meanings; a public optional-counter extension
is a separate integration step. Local subprocess tests use synthetic counters
and do not replace this hardware check.
