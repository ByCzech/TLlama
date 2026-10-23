# Decision inventory and native references — 0066

This checkpoint extends the accepted0065 Laya reference with the already
inventoried Nimble and Clef, selected explicitly through `--model` and verified
using GGUF Decision metadata. It runs private native `/v1/systemone`.
It opens no public listener and adds no production Decision adapter, SDK method,
model download, external scorer or native change/build. Registered TOMLs and
GGUF files are not edited. Keep the accepted text/image embedding gates.

From the applied TLlama checkout and the same existing model-store environment,
run these two new references. Each gets its own process and output tarball;
there is no reason to repeat the accepted Laya or embedding hardware matrices.
The explicit2048 context/batch/ubatch accommodates the whole-schema controls.
Inventory only the selected artifact; all five inventories were reviewed0065.

```bash
model=ggml-org/Bespoke-Nimble-9B-v3-GGUF/Bespoke-Nimble-9B-v3-Q4_K_M
result_dir=/tmp/tllama-decision-nimble-$(date +%Y%m%d-%H%M%S)
if TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/decision-smoke.py --model "$model" --inventory-model "$model" \
    --ctx 2048 --output "$result_dir"; then
  echo 'Nimble protocol completed; review summary and inventory follow-ups.'
else
  echo 'Nimble did not complete; return the retained evidence.'
fi
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"

model=ggml-org/Clef-Flash-GGUF/Clef-Flash-Q4_K_M
result_dir=/tmp/tllama-decision-clef-$(date +%Y%m%d-%H%M%S)
if TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/decision-smoke.py --model "$model" --inventory-model "$model" \
    --ctx 2048 --output "$result_dir"; then
  echo 'Clef protocol completed; review summary and inventory follow-ups.'
else
  echo 'Clef did not complete; return the retained evidence.'
fi
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

The five references are PPLX v1.2 Q4_K_M, Bespoke-Nimble9B v3 Q4_K_M,
Clef-Flash Q4_K_M, Tev1-4B experimental Q4_K_M and Laya Q8_0, exactly as
supplied by the maintainer. No replacement quantization or model-name capability
heuristic is used. The default inference target remains Laya for compatibility;
without `--inventory-model`, all five references are inventoried. Only the
selected model is launched. Laya, Nimble and Clef are the audited reference
types; other types or incomplete selected headers fail before launch.
`--inventory-only` reads headers
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
reference does not accept the other model conversions.

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

Nimble renders the whole question list for each requested field, selects
single-token label codes (A..Z, then eligible AA..ZZ; at most255), and reads
their logits. Its template switches label instructions above26 options. Usage
sums the actual full-schema field tasks, not the standalone requests.

Clef sorts choice keys, serializes nested object keys canonically and renders
one joint prompt. It tokenizes separated prompt pieces, assigns question-type
and option order to their spans and reads the joint head's option scores. Usage
counts that single joint task. Response question IDs retain request order;
choice probability keys follow native's sorted order, including literal keys.
Noul's true/false option order differs internally but the reported probability
still means true. The diagnostic validates the reported contract, not a guessed
head-column position.

Do not require single/combined equality, additive usage or question-order
invariance from Nimble/Clef. Those changes can alter the trained model's input.
The diagnostic emits `composition-observation` and `usage-composition` records
without review failures for these differences; identical-request repeats remain
required comparisons. Clef's sorted-choice permutation is required to match.
LEV's two reversed variants are outside the selected references. Media remains
a separate extension: these text-only launches remove the projector and expect
inline image refusal501, even if a model could use images with a projector.

## Exact model-source findings

- [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF) explicitly
  targets `/v1/systemone`; its exact Q8_0 native reference is accepted0065.
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
  has no explanatory model card. The reviewed exact local0065 GGUF lacks both
  Decision type and the systemone template; do not assign a guessed type.

Sources checked2026-10-19. Generic automatically generated HF app instructions
are not evidence that a conversion supports ordinary chat or this native head.
Removed Strands and mys-Laya artifacts remain out of scope.

## Launch, probes and evidence

The owned diagnostic worker retains resolved model/device/placement, overrides
ctx/batch/ubatch to `--ctx` (default1024), F16 K/V, FAoff, speculationnone,
removes a projector and
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

The retained Laya gate checks15 successful request controls: three single
question types, combined/repeat/reversed questions, structured JSON/repeat,
literal Unicode/newline/escaped keys, one-option choice, ten-level score,
changed state, native Laya head truncation and recovery after errors/cancel.
Eight equality comparisons check answers/accounting across the independent
question path and repeated requests.

Nimble/Clef each check19 successful controls: retain the common Laya controls,
omit its head-truncation probe, add a repeat of reversed questions, reversed
choice options and their repeat,27 choices in descending key order, and a noul
question with structured instructions/criteria. Six required comparisons cover
repeat,reversed-repeat,JSON-repeat,option-permutation-repeat,error recovery and
cancel recovery. Clef adds a seventh for sorted-option permutation equivalence.
Nimble records five composition observations; Clef records four. Their native
budget record explicitly marks the Laya truncation control not applicable.

Ten error probes expect nine400s and one501. State sensitivity records whether
a simple changed ticket changes the
probabilities; it is not a retrieval/accuracy benchmark or Transformers parity.

Cancellation uses32 independent questions for Laya but only two for Nimble/Clef,
to bound the whole-schema/joint prompt instead of overflowing the context.
It can complete before cancellation; that response is validated and retained.
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
cleanup. Whole-schema stub answers deliberately vary with composition and
use non-additive counts; Clef sorts keys, and repeated-response/usage mutations
must still request review. They cannot certify a real model's trained head or
semantic accuracy.
