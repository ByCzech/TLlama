# Decision inventory and first native Laya reference — 0065

This checkpoint reads the five exact prepared local GGUF references, then runs
only `ggml-org/Laya-GGUF/Laya-Q8_0` through private native `/v1/systemone`.
It opens no public listener and adds no production Decision adapter, SDK method,
model download, external scorer or native change/build. Registered TOMLs and
GGUF files are not edited. Keep the accepted text/image embedding gates.

From the applied TLlama checkout and the same existing model-store environment:

```bash
result_dir=/tmp/tllama-decision-laya-$(date +%Y%m%d-%H%M%S)
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/decision-smoke.py --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

The five references are PPLX v1.2 Q4_K_M, Bespoke-Nimble9B v3 Q4_K_M,
Clef-Flash Q4_K_M, Tev1-4B experimental Q4_K_M and Laya Q8_0, exactly as
supplied by the maintainer. No replacement quantization or model-name capability
heuristic is used. The default first inference target is Laya; other types are
inventoried, never automatically launched. `--inventory-only` reads headers
without starting a worker, even if the native executable is unavailable.
`--inventory-model` can override the inventory list for isolated tests/local
diagnostics; repeat it for each desired registered reference.

Inventory records the resolved path, size/mtime, registered definition/hash,
architecture/context/attention/pooling-related output metadata, all native
`<arch>.decision.*` keys, embedded named templates and their hash, compact vocab
size, mask/separator pieces and tensor descriptors. It does not read tensor
values or populate the production metadata cache with a new schema. A Laya
header candidate requires ModernBERT, decision block/head budget metadata,
mask/SEP IDs and scorer tensor names. Native remains authoritative for shapes,
complete head blocks, tokenizer layout and actual inference. `header_candidate`
is a metadata prerequisite, never successful inference evidence.

All inventory follow-ups are retained in `finished.inventory_followup`; absent
Decision metadata or a template can be a valid finding for an inventoried
artifact. Inventory read failures separately require review. A successful Laya
reference does not accept the other four model conversions.

## Pinned native contract

Audited source is3144c167ef703e93cee0b58d7236b6a28c6e50ad,
build b11552-3144c167ef. `common/common.cpp::common_get_decision_type` selects
`<arch>.decision.type`; `tools/server/server-decision.cpp` requires the embedded
`tokenizer.chat_template.systemone`. Types are openjev,lev,kev,nimble,laya,clef,
pplx-decider,lfm2-d1,lfm2-d1-omni. Names/architecture alone do not select a head.
Unknown type/template problems can prevent native startup.

The API accepts `state` and an ordered map of `questions`, each with `type`,
`instructions` and type-specific `criteria`. Non-string state/instructions go
to the embedded template as JSON values. `choice` returns keyed probabilities,
argmax and confidence; `score` returns probabilities, legend, expected level
index and confidence; `noul` is the probability of true. Usage counts actual
prompt tokens and output_tokens is0. No answer tokens are generated. The
diagnostic independently checks finite/ranged values, sums, keys/order, score
expectation and native confidence formulas. Metadata temperatures are recorded;
probabilities are not asserted calibrated for an application's data.

Laya uses ModernBERT's trained head, mask/SEP markers and separate choice/score/
noul columns. Native automatically enables its per-token embedding output and
NONE pooling. This is the internal score channel, not the public pooled semantic
embedding contract. Laya is noncausal, requires its full prompt in one ubatch,
and native truncates question/options to its training budget (option text cap48
tokens plus its marker; total head budget comes from GGUF). The state still has
to fit the native context; Python does not silently truncate it. The deliberate
long-question probe records this model-specific native truncation.

By comparison Clef evaluates questions jointly, Nimble includes the whole
question list in its template, and LEV reverses choice labels in two variants.
Do not reuse Laya single/combined equivalence or additive usage assumptions
for all types. Media is a separate native extension with per-type requirements;
this first reference tests text and verifies Laya's image refusal (501).

## Exact model-source findings

- [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF) explicitly
  targets `/v1/systemone`; it is the first reference candidate. Actual local
  header/head and hardware evidence are still required.
- [Bespoke-Nimble9B v3](https://huggingface.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF)
  and [Clef-Flash](https://huggingface.co/ggml-org/Clef-Flash-GGUF) also explicitly
  target that endpoint, using different native paths.
- The exact [Jakevin PPLX v1.2 conversion](https://huggingface.co/Jakevin/pplx-decider-flash-v1.2-GGUF)
  documents a noncausal custom runner/external BF16 readout and says the built-in
  native Decision template targets an older prompt. A valid Decision metadata
  key would not resolve this semantic/runtime mismatch. The inventory reports
  a source caution and does not launch it or silently apply attention overrides.
  This finding applies to that version/conversion, not every PPLX model.
- [Tev1 experimental](https://huggingface.co/gguf/Tev1-4B-experimental-GGUF)
  has no explanatory model card; obtain exact local metadata/template evidence
  before assigning a native Decision type or claiming support.

Sources checked2026-10-19. Generic automatically generated HF app instructions
are not evidence that a conversion supports ordinary chat or this native head.
Removed Strands and mys-Laya artifacts remain out of scope.

## Launch, probes and evidence

The owned diagnostic worker retains resolved model/device/placement, overrides
ctx/batch/ubatch to1024, F16 K/V, FAoff, speculationnone, removes a projector and
generation template override for this text reference, and lets native select
the embedded Decision template/head. `ProbeWorker` adds only diagnostic private
requests using the existing authentication, transport slot, timeout, JSON bound
and cleanup. Production admission/mode methods stay unchanged; this is an
independent reference, not production Decision integration.

`raw.jsonl` retains all inventory details/templates/tensor descriptors, original
TOMLs/hashes, effective argv/props and native model modalities, requests,
responses and memory. `summary.jsonl`/stdout are concise; load/full worker logs
are retained too. Use a new output directory. `--ctx` accepts512..4096 and is
an explicit diagnostic choice, not a model's trained limit.

The first native gate checks15 successful request controls: three single
question types, combined/repeat/reversed questions, structured JSON/repeat,
literal Unicode/newline/escaped keys, one-option choice, ten-level score,
changed state, native Laya head truncation and recovery after errors/cancel.
Eight equality comparisons check answers/accounting across the independent
question path and repeated requests. Ten error probes expect nine400s and one
501. State sensitivity records whether a simple changed ticket changes the
probabilities; it is not a retrieval/accuracy benchmark or Transformers parity.

Cancellation reports whether transport was pending; native inference start and
early compute stop remain unknown. Same-PID recovery, known backend-memory
telemetry, explicit unload, process exit and owned-directory removal complete
the lifecycle gate. Buffer capacities are not physical RSS/VRAM peaks. Exit0
means protocol completion; review also requires review_required=false, the
expected comparisons/error observations and inspection of inventory follow-ups.
Return complete pytest results and the whole hardware tarball.

```bash
pytest -q tests/test_decision_smoke.py tests/test_native_worker.py \
  tests/test_native_embeddings.py tests/test_embedding_media_api.py
pytest -q
```

The tests use small synthetic GGUF headers and independently generated typed
TCP responses. They catch malformed probabilities/confidence/score/legend,
usage/order, protocol drift, missing live Decision output, cancellation and
cleanup. They cannot certify a real model's trained head or semantic accuracy.
