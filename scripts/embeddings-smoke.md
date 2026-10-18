# Native text embedding reference

Run from the TLlama checkout after applying 0054. This is an independent
diagnostic over managed, private native workers. It does not expose a public
embedding API, change a registered TOML, download a model or require a native
rebuild. Synthetic tests verify the harness, not GPU/model behavior.

The default references are the already prepared MiniLM Q8_0 and Qwen3-Embedding
0.6B Q8_0. They run sequentially, each in a fresh process. Existing model-store,
executable and device configuration are used. The public TLlama server's
residency is separate: account for other workloads before running this tool.

```bash
PYTHONPATH=src python3 scripts/embeddings-smoke.py \
  --output /tmp/tllama-embeddings-reference-$(date +%Y%m%d-%H%M%S)
```

Optional `--models-dir` and `--executable` override the normal environment.
`--model` selects a registered model; repeat it for several sequential runs:

```bash
PYTHONPATH=src python3 scripts/embeddings-smoke.py \
  --model second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0 \
  --model Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0 \
  --output /tmp/tllama-embeddings-reference-custom
```

Use a new output directory every time; the tool refuses to overwrite results.
No projector is loaded in this text-only checkpoint, even if one is registered.
It does not certify Qwen3-VL or EmbeddingGemma 2 image/audio/video embeddings.
Record those separately after reviewing the text reference and native input
contracts. A projector's presence alone does not establish embedding support.

Diagnostic overrides: embeddings enabled; MTP explicitly disabled; context 256
by default; batch/ubatch at least 512 and larger than the context; F16 KV types;
flash attention off; no chat-template override. Device, weights and other load
settings are inherited. `--ctx` can select 64..16384 tokens. `--pooling auto`
omits the pooling flag and respects the native/model default. Explicit `mean`,
`cls` or `last` is available for a separately documented model-contract probe;
do not choose it solely from a model's name or silently repair a failed run.

The diagnostic sends the displayed strings exactly, including Unicode and
newlines. It inserts no task prefixes. Model-specific retrieval instructions
remain caller/model semantics; these probes check the protocol, not retrieval
quality or a model ranking.

## Evidence and checks

- Single texts, a batch, a repeated batch, reordered inputs and duplicate texts.
- Nonempty, finite, consistently sized vectors and ordered response indices.
- Native unit normalization is measured, without modifying the vectors.
- Exact measured `/tokenize` counts versus reported embedding usage. Missing
  usage stays unknown; tokenizer counts are retained as a separate measurement.
- Near-context and over-context token-ID inputs, invalid encoding, empty array
  and model-dependent empty-string behavior. Errors are followed by a control
  request on the same PID.
- A non-streaming transport cancellation followed by the same control batch.
  The request can complete before cancellation, especially on a fast GPU.
  `pending_at_cancel` records that distinction. Neither socket closure nor
  same-worker recovery proves an early native stop: inspect `worker.log` for
  actual native cancellation and task release.
- Native memory snapshots after load and requests, owners and deduplicated
  totals; final worker exit, private-directory cleanup and empty residency.

Comparison defaults: absolute tolerance `1e-4`, relative tolerance `1e-3`;
unit-norm tolerance `1e-3`. These are review thresholds, not promises of bitwise
identity between batching or hardware backends. Raw vectors and maximum
absolute differences are retained. Override with `--atol`, `--rtol` and
`--norm-tolerance` only for an explicitly reviewed comparison.

`raw.jsonl` includes full requests, responses/vectors, props and memory reports.
`summary.jsonl` is also printed to stdout. `worker.log` retains native runtime
logs; per-worker `*-load.log` files retain load evidence. Native API keys are
never included. Timings include transport and token-count verification and are
not inference-only benchmarks. Memory capacities are not OS RSS/VRAM peaks.

Exit 0 means transport/schema/lifecycle checks completed. Inspect the final
`review_required` and `observations` even on exit 0: numerical differences,
normalization, usage, error-status or memory-evidence deviations require review.
Exit 1 means a hard error; exit 2 means invalid CLI arguments/output directory.
No run certifies the future Ollama/OpenAI public adapters or SDK compatibility.

Return the entire result directory for reference review before public API work.
