# Managed text Decision workers — 0067

The independent native references for the exact registered Laya Q8_0,
Bespoke-Nimble-9B-v3 Q4_K_M and Clef-Flash Q4_K_M passed0065/0066. This
checkpoint integrates their text Decision contracts with ModelManager and
NativeWorker. Public Decision endpoints and the managed/public hardware
comparison follow separately. Existing generation routes reject Decision mode
before load/eviction; `/api/show` suppresses generation capabilities for a
declared Decision head and does not yet advertise a public Decision capability.

## Mode and configuration

The GGUF's `<arch>.decision.type` selects the frozen worker `decision_type`.
There is no new TOML mode flag or file/model-name heuristic. Current integrated
pairs are modern-bert/laya, qwen35/nimble and clef/clef. Native must also report
the private alias and `architecture.output_modalities: ["decisions"]` at startup.
Decision requires the audited runtime **b11552-3144c167ef**, including when the
general runtime version policy is `warn`. Other native Decision families are
not integrated by this checkpoint; that is not a claim about upstream support.
The exact PPLX v1.2 conversion retains its runner/readout/prompt follow-up; Tev's
reviewed file lacks Decision type/template. Neither is guessed into this path.

An embedded `tokenizer.chat_template.systemone` is required. Calibration keys
must be finite positive numbers. Laya also needs valid head-block/token-budget
metadata and MASK/SEP vocabulary IDs. Native validates the actual complete
head tensors, tokenizer and prompt layout. Metadata alone is not inference
proof. Cache schema7 adds these fields; digest caches remain valid.

Normal precedence still selects device, context, batch/ubatch, K/V and FA.
There is no silent capacity tuning or registered TOML edit. Useful bounded
reference settings are1024 for Laya and2048 for Nimble/Clef, setting all three
of `n_ctx`, `n_batch` and `n_ubatch` to the chosen value, with F16 K/V and FAoff.
Laya needs the whole prompt in a physical ubatch; other native prompt/context
limits remain authoritative. This transport neither estimates the rendered
prompt's token count nor silently truncates state/questions. Laya's own trained
head truncation remains native behavior.

Decision workers reject explicit semantic embedding mode/pooling, generation
template overrides, model sampling/stop/system/FIM settings, speculative/draft
settings, GGUF metadata overrides and LoRA. This initial text contract also
rejects projectors. Configure generation-specific runtime settings per model
when sharing a server with Decision models. Global sampling defaults are not
forwarded to the Decision endpoint. Native chooses its own head/mode from GGUF;
TLlama does not emit `--embeddings`, pooling or attention overrides for it and
explicitly disables speculation. An internal per-token score channel does not
advertise semantic retrieval vectors.

## Ownership and private protocol

```python
async with manager.acquire_native_model(name, operation="decision", keep_alive="5m") as lease:
    result = await lease.worker.decision({
        "state": {"ticket": "I was charged twice."},
        "questions": {
            "team": {"type": "choice", "instructions": "Which department?",
                     "criteria": {"billing": "payments", "shipping": "parcels"}},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        },
    })
```

Operation admission checks the resolved mode before residency/eviction. Generic
load control remains operation-unset. Mode/head changes participate in immutable
definition identity and normal draining; admitted leases keep their original
worker. Direct generation/embedding transports independently reject Decision
mode, and Decision transport rejects those modes. Tokenization shares the same
single slot and loaded tokenizer. TTL, memory telemetry, capacity, protected
unload, shutdown and cleanup retain the existing owned lifecycle.

`decision(payload)` accepts only model, non-null state and1..32 questions.
Each question accepts type, non-null instructions and criteria: choice has
1..255 keyed options, score has2..10 ordered levels, noul has an object or null.
State, instructions and descriptions may be arbitrary finite JSON, including
false,0,empty strings and nested values. All object keys must be strings. JSON
is bounded to2MiB UTF-8,depth32 and100000 value nodes. These resource bounds
are transport policy, not the model's trained context. Image/media keys, stream,
sampling and other options are rejected before POST. A caller-supplied model
value is replaced by the owned private alias.

The request is copied before awaiting the slot. Literal Unicode, newlines,
backslash-n and nested JSON retain their meaning; no de-escaping, prompt render,
marker replacement or choice sorting is done in Python. Clef's native sorted
choice probability order is validated as such; other types retain request order.
Native question IDs retain request order for all three types.

Authentication, timeouts, bounded16MiB JSON responses and HTTP errors use the
existing worker transport. Successful responses must have the owned alias,
requested typed IDs,finite/ranged probabilities,sum1,argmax choice,exact legend,
expected score and published confidence formulas. Usage must be strict positive
integer input_tokens and integer output_tokens0. Invalid request JSON raises
ValueError before POST; native HTTP errors retain WorkerRequestError/status;
malformed native results raise WorkerProtocolError. No result is repaired or
normalized. Slot release and same-process recovery work after these errors.

Nimble renders the whole schema per field and Clef uses one joint task. Their
single/combined answers and input counts need not match or add. Identical
requests retain the native authoritative accounting; this adapter invents no
Python usage estimate. The independent reference's composition observations
are not an application calibration/accuracy benchmark.

Cancellation closes the owned HTTP response and releases the transport slot;
an admitted lease still protects its worker during unload. Native early compute
stop and a cancellation latency bound are not established. Memory remains
native backend-buffer capacity, with unknown/partial fields preserved.

## Verification

```bash
pytest -q tests/test_native_decision.py tests/test_decision_smoke.py \
  tests/test_native_embeddings.py tests/test_native_config.py \
  tests/test_worker_config.py tests/test_native_residency.py \
  tests/test_native_metadata_cache.py tests/test_gguf_metadata_flags.py \
  tests/test_embedding_api.py
pytest -q
```

Tests use actual small GGUF headers and owned authenticated TCP stubs. They
verify metadata/config/mode admission, live runtime/output checks, literal JSON,
typed responses and corruption recovery, errors, cancellation, draining and
cleanup. They do not replace the accepted real native reference evidence.
No new native build/model download or repeated reference matrix is needed here.
