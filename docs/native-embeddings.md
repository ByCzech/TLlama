# Native embedding workers — 0063

0063 permits an explicitly configured embedding worker to retain its registered
projector and projector options. The accepted Qwen3-VL native reference0062
proved literal text/image inputs with LAST pooling. A projector alone does not
select embedding mode or a pooling type. For that exact registered model use:

```toml
[runtime]
embedding = true
pooling_type = 3
```

Keep the existing `[llm]` and `[mmproj]` references. Normal configuration
precedence still controls context, batches, placement, K/V, FA and projector
options; production does not inherit the diagnostic's capacity settings.
The model's pooled semantic contract must still be known before public text
admission. Decision heads, per-token/rank pooling and generation settings are
not enabled by attaching a projector.

Existing public text routes also work on this worker, with the same exact-token,
limit, normalization and usage checks. `/api/show` advertises `embedding`, without
claiming public image input or generation/vision routes. Public embedding
schemas still refuse images/typed media. The internal embedding transport
retains its authenticated bounded slot and raw protocol; its caller owns media
validation and must check the admitted generation's live capabilities.

Projector path, file fingerprint and options already participate in the frozen
definition identity. Changes replace the generation through ordinary draining;
admitted requests retain their old worker. Memory telemetry retains projector
ownership and unknown/partial accounting as reported; it does not reuse the
text-only memory estimate or invent physical residency. Generation requests
continue to fail operation admission before replacing an embedding worker.

Use the [managed reference recipe](../scripts/embeddings-multimodal-smoke.md)
to exercise production resolution, `WorkerLaunchSpec` and `NativeWorker` without
editing the registered TOML. Public media contracts are the next checkpoint.

## Original worker foundation — 0057

Public adapters were added in 0058; see [embedding-api.md](embedding-api.md)
for the current public contract and remaining hardware gate. The sections
below describe the 0057 worker checkpoint.

This checkpoint integrates explicit text embedding workers with ModelManager.
The accepted independent reference used MiniLM Q8_0 and Qwen3-Embedding 0.6B
Q8_0 on native b11552-3144c167ef. Public `/api/embed`, legacy `/api/embeddings`
and `/v1/embeddings` adapters are the following checkpoint. No native rebuild
or additional model download is required for this change.

## Model configuration

The existing `[runtime]` table now accepts a strict boolean `embedding` and
an integer `pooling_type`. Defaults are false and -1 respectively. These are
load settings: they cannot be selected by a public inference request.

For example, an explicitly configured text embedding definition can use:

```toml
[llm]
model = "HuggingFace/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf"

[runtime]
embedding = true
pooling_type = -1
n_ctx = 256
n_batch = 512
n_ubatch = 512
type_kv = "f16"
flash_attn = false
```

Device selection retains its normal precedence. No TOML is rewritten and no
mode is inferred from a model name, architecture or generic native embedding
flag. Existing definitions stay in generation mode until explicitly configured.
Preparing the definitions is optional at this stage; public embedding calls
are not implemented yet.

| `pooling_type` | Launch behavior |
| --- | --- |
| -1 | Omit `--pooling`; use the native model default |
| 1 | `--pooling mean` |
| 2 | `--pooling cls` |
| 3 | `--pooling last` |

An explicit pooling override must match the exact model's documented contract.
None/0 (per-token output) and rank/4 (reranking) are separate operations and
are refused by this pooled-text configuration. An automatic model default
is still resolved by native; -1 is not evidence that the loaded output is a
semantic retrieval vector or even a supported pooled output. In particular,
Laya's decision head uses an embedding tensor for scores. Its decision output
must be verified separately before exposing semantic embeddings.

Attention remains the model default. The pinned `common/arg.cpp` exposes
`--attention` only to the standalone embedding example, not the server.
`attention_type` therefore remains rejected rather than emitting an unsupported
server flag. No generation parameter is used to guess attention or pooling.

The usual global environment coercion also accepts `TLLAMA_RUNTIME_EMBEDDING`
and `TLLAMA_RUNTIME_POOLING_TYPE`; per-model TOML wins. For a mixed generation
and embedding inventory, prefer per-model settings. A non-default pooling
requires embedding=true. Template overrides, model system/sampling/
FIM settings and speculative decoding/draft settings are rejected in embedding
mode. Configure MTP per generation model when sharing a server with embeddings.
The launch explicitly disables speculation. Batch, ubatch, context, K/V and
placement settings otherwise retain normal configuration precedence; no silent
capacity tuning is performed. Native input limits still apply, including whole
encoder sequences fitting the physical ubatch. Public input validation and
truncation contracts will be implemented with the adapters.

## Ownership and operation admission

Mode and pooling are frozen fields of WorkerLaunchSpec. They participate in
resolved definition identity, so editing a model's mode/pooling replaces its
worker generation using the existing draining/capacity rules. Already admitted
leases retain the complete old configuration and process.

`acquire_native_model(..., operation="embedding")` requires an explicitly
configured embedding worker. `operation="generation"` requires generation mode;
all public generation routes now supply that requirement. The check occurs
after definition resolution but before residency admission or eviction. A
wrong operation neither starts a process nor retires an existing generation.
An unset operation is generic load control, not a declaration of capability.
Existing empty Ollama generate/chat preload and unload commands work for either
mode. Direct worker chat/completion methods, including streams, independently
reject embedding mode, and its embedding method rejects generation mode.

`NativeWorker.embeddings(payload)` owns the authenticated POST to private
`/v1/embeddings`, pins the private model alias, and reuses the single transport
slot, JSON size bound, errors and response cleanup. Tokenization uses the same
loaded tokenizer in either mode. The caller owns its lease. This is a raw
internal protocol result; it does not normalize/repair vectors, insert prefixes,
choose dimensions, count characters as tokens or claim public API validation.

Cancellation closes the transport and releases the Python slot. The accepted
hardware logs show that native may finish the cancelled batch before processing
the next call. No early compute-cancellation guarantee is introduced here.
Memory telemetry, residency accounting, TTL and unload cleanup use the existing
owned lifecycle; there is no second embedding manager or process queue.

`/api/show` suppresses generation capabilities for configured embedding workers.
It does not advertise public embedding capability until those adapters exist.

## Verification

```bash
pytest -q tests/test_native_embeddings.py tests/test_native_config.py \
  tests/test_native_residency.py tests/test_native_manager.py \
  tests/test_native_public_api.py tests/test_embeddings_smoke.py
```

Owned TCP subprocess tests cover mode/pooling validation, precedence, immutable
replacement, rejected operation admission, direct transport guards, input/alias
preservation, HTTP error recovery, cancellation/slot reuse, capability reporting
and preload/unload cleanup. Synthetic vectors do not certify model quality or
early native cancellation. Keep the independent 0054 hardware reference intact;
the next hardware checkpoint will exercise the public adapters and SDKs.
