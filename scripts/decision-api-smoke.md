# Decision native / managed / public comparison — 0069

Apply 0069 after the accepted 0068. Run from the checkout with the same installed
native binary, model store and device as the accepted 0065/0066 references.
The default selection is the three existing registered models, in order:

| Model | Native type | Context / batch / ubatch |
| --- | --- | --- |
| `ggml-org/Laya-GGUF/Laya-Q8_0` | Laya | 1024 |
| `ggml-org/Bespoke-Nimble-9B-v3-GGUF/Bespoke-Nimble-9B-v3-Q4_K_M` | Nimble | 2048 |
| `ggml-org/Clef-Flash-GGUF/Clef-Flash-Q4_K_M` | Clef | 2048 |

```bash
result_dir=/tmp/tllama-decision-api-$(date +%Y%m%d-%H%M%S)
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/decision-api-smoke.py --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

The listener binds a kernel-selected loopback port. It does not contact or modify
a running TLlama service, install packages, download models or rebuild native
code. Its workers share the hardware with other workloads; use the same available
memory budget as the accepted references. No Ollama/OpenAI SDK is required:
`/v1/systemone` is an extension with no claimed standard high-level SDK method.

`--model` can be repeated to select a subset; `--models-dir` and `--executable`
override the usual environment settings. `--ctx 512..4096` overrides the default
for a separately reviewed run. The three-model default needs no additional
downloads. PPLX v1.2 and Tev follow-ups remain separate; removed Strands and
mys-Laya are out of scope. Accepted embedding hardware need not be repeated.

## Two sequential workers per model

The tool reads exact GGUF metadata, tokenizer markers and tensor descriptors
using the independent 0065/0066 inventory, without reading tensor data. It saves
the original registered TOML, hashes and GGUF size/mtime. Each model gets its own
`model-N/managed-model.toml` under the new output directory. That temporary
definition retains registered weight location and runtime placement, explicitly
sets the table's capacities, F16 K/V, FA off, speculation none and no semantic
embedding/pooling output. It omits projector, template, sampling, system and FIM
sections; draft/FIM companions are removed from model/global overrides. The
native head and embedded systemone template still own tokenization and scoring.
Registered definitions are checked byte-for-byte unchanged; caches stay private.

First, the independent `ProbeWorker` uses the accepted native reference recipe
and raw authenticated/bounded `/v1/systemone` transport. Its independent formula,
key/order and usage checks do not import production Decision validators. It
unloads and exits before the second worker starts.

Second, the first valid public HTTP request performs a cold load through ordinary
production configuration, metadata admission and `NativeWorker`, without
post-resolution mutation. Its answer and exact usage must match the independent
combined control. All later private/public calls reuse that worker. The effective
native command must equal the independent recipe after excluding only dynamic
port and owned API-key-file location. Both require audited native fingerprint
`b11552-3144c167ef` and live `output_modalities=["decisions"]`. Ready rows retain
argv, props, metadata, runtime, PID and generation for independent review against
the accepted native archives as well as the fresh reference.

Every successful control then runs through both managed `NativeWorker.decision`
and the actual public TCP `POST /v1/systemone`. The only permitted model-envelope
difference is native private alias versus admitted public reference. Answers and
native usage are compared in all three directions. Default probability thresholds
are `--atol 1e-4 --rtol 1e-3`; usage must match exactly regardless of tolerance.
No normalization, escape repair, prompt prefix, score recalibration or synthetic
usage is introduced.

## Controls and error scopes

There are 15 successful controls for Laya and 19 for each Nimble/Clef, retaining
the accepted independent reference bodies: single choice/score/noul, combined,
repeat, reversed questions, structured state/repeat, literal Unicode/newline/
backslash keys, one choice, ten score levels, changed state and recovery after
errors/cancellation. Laya retains its trained-head truncation control; Nimble/
Clef add reversed-question repeat, reversed-option/repeat, 27 choices and typed
instructions/criteria. No claim is made that real models fit 255 choices or 32
arbitrary questions in these contexts merely because the transport permits them.

The fresh independent phase retains its 8/6/7 required reference comparisons.
Laya's independent questions have additive usage. Nimble whole-schema field tasks
and Clef joint tasks need not match standalone answers/counts; their composition
differences remain observations. The cross-path gate compares the same schema,
never an independently changed schema. It performs 45 cross-path comparisons for
Laya and 57 each for Nimble/Clef: 159 total in the default selection, plus three
separately recorded cold-start comparisons.

Before any managed worker loads, the public phase checks capability `decision`
and 16 refusal probes: unknown/extra settings, streaming, media including the
native-recognized chat-state layouts, invalid JSON/duplicate keys/nonfinite
values, null state, empty questions and missing model. All must leave residency
empty. Two wrong-operation probes after load must preserve the Decision worker.
A raw private native 400 exercises recovery separately. The independent phase
also retains a native unsupported-image 501. Public text validation refuses
that image with 400 before admission: these intentionally different error scopes
are recorded separately, not asserted identical. URLs in normal state remain
literal; remote media controls use `example.invalid` and are not fetched.

Each phase exercises bounded cancellation input (32 Laya questions, two Nimble/
Clef questions). The managed phase cancels both a leased private call and an
actual public HTTP request. A completed request is validated and retained;
pending transport is closed, leases must drain, and recovery must retain one PID
and generation within the phase. There is no reliable inference-start event;
`inference_started_observed` and `native_early_stop_observed` remain unknown.
Transport cleanup is not proof of early native compute stop or a latency bound.

The public phase unloads via ordinary `/api/generate` with `keep_alive:0`.
Both workers must exit cleanly with empty residency and owned directories
removed. Native memory snapshots retain owners, status and deduplicated totals
after load/requests. Retained buffer capacities are not physical RAM/VRAM peaks.

## Results and pytest

`raw.jsonl` retains complete request/response bodies, inventory, original and
temporary definitions, launch details and native memory. `summary.jsonl` equals
stdout and omits bulky bodies; full worker/load logs remain in the result.
Credential values and authentication headers are not logged. The output path
must be new. Exit 0 means hard protocol/lifecycle completion, not automatic
acceptance: review `review_required`, observations and complete evidence. Exit 1
is a hard failure; exit 2 is invalid CLI/output. Return the whole tarball and
pytest results. This is adapter equivalence, not an accuracy/quality benchmark.

```bash
pytest -q tests/test_decision_api_smoke.py tests/test_decision_smoke.py \
  tests/test_native_decision.py tests/test_decision_api.py
pytest -q
```

Synthetic tests exercise all three head contracts, genuinely separate owned
processes, actual public TCP, unchanged files, pre-admission refusal, corruption
and count differences, both fast-completion/pending-cancel paths and cleanup.
They validate the harness; maintainer hardware results are still required.
