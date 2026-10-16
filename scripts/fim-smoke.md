# Native FIM reference smoke

This is a measurement checkpoint, before implementing Ollama `suffix` in TLlama.
It resolves registered models through `ModelManager`, holds a lease for all
requests, and uses only its own authenticated loopback worker. It does not call
the public `/api/generate` route, change model TOML, or contact a running service.

## Run

From the TLlama checkout, with the normal model directory/runtime environment:

```bash
PYTHONPATH=src python scripts/fim-smoke.py --output /tmp/tllama-fim-reference
```

The output directory must be new. Defaults run these registered models, one at a
time, with independent PSM and SPM workers (no concurrent model loads):

| Registered model | Role |
| --- | --- |
| `codegemma-2b-Q8_0` | FIM candidate |
| `starcoder2-3b-Q8_0` | FIM candidate |
| `qwen2.5-coder-7b.Q4_K_M` | jsdevtom TypeScript FIM autocomplete fine-tune |
| `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L` | Instruct candidate; measure FIM rather than assume support |
| `Codestral-22B-v0.1-Q4_K_M` | Instruct/FIM candidate, including SPM comparison |
| `unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0` | Negative candidate, PSM worker only |

Use the exact registered name if an alias differs. An explicit selection replaces
the entire default set; repeat either model option to select multiple models:

```bash
PYTHONPATH=src python scripts/fim-smoke.py \
  --model codegemma-2b-Q8_0 --model starcoder2-3b-Q8_0 \
  --output /tmp/tllama-fim-small

PYTHONPATH=src python scripts/fim-smoke.py \
  --negative-model unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0 \
  --output /tmp/tllama-fim-negative
```

`--models-dir` and `--executable` override their environment defaults. Device,
offload, cache types, adapters and other load settings follow the normal TOML/env
resolution. Set `TLLAMA_RUNTIME_DEVICE=Vulkan0` if no model/global device is set.
No download or new engine build is needed.

The diagnostic deliberately overrides context (`--ctx`, default 4096), logical
batch (`--batch`, default 512), microbatch (at most the selected batch), and
`spm_infill` (`--order both|psm|spm`, default both). Actual launch arguments are
saved in `raw.jsonl`. Generation uses greedy sampling, fixed seed, no prompt
cache, no user stop strings/grammar/chat template, and 64 output tokens by
default (`--predict`). These controlled probes do not measure TOML sampling or
template precedence. The model's native EOG handling remains active.

## What is measured

1. Record the loaded vocabulary's FIM IDs/pieces from native load diagnostics,
   then verify them against the live tokenizer. This diagnostic parser is not a
   proposed production capability mechanism. `/infill` independently confirms
   missing-token refusal for the negative candidate. Missing diagnostics on a
   positive candidate fail the probe rather than imply lack of FIM support.
2. Two Python prefix/suffix probes require addition versus multiplication. Each
   runs `/infill` and explicit full FIM `/completion`, streamed and nonstreamed.
   Python AST inspection reports whether the generated expression fits the
   suffix. Generated code is **never executed**. This narrow check is not a model
   quality benchmark; equivalent but more elaborate code can need manual review,
   and the TypeScript fine-tune is intentionally tested on the same Python probes.
3. A third prompt reproduces the inspected b11552 native formatter's synthetic
   `myproject` / `filename` header and batch-derived clipping. This is a comparison
   hypothesis, not a chosen TLlama implementation. It helps separate formatting
   differences from model behavior. PSM and SPM are explicit token sequences;
   prefix/suffix text itself is tokenized without parsing special-token strings.
4. A longer prefix and suffix exceed both batch limits but fit in the real
   context. One-token generation records actual `tokens_evaluated` versus both
   full and clipped prompt counts. For batch 512 the inspected formatter retains
   at most 384 prefix tokens and 126 suffix tokens with empty extra `prompt`.
5. Cancel a live `/infill` stream, then run a normal FIM request on the same
   worker. Negative tests require model-specific missing-token errors in stream
   and nonstream and a successful ordinary completion afterward. A 404 or other
   arbitrary failure does not count as a correct negative result.
6. Release leases, shut down every owned worker, and record process exit,
   temporary directory removal and empty residency. Errors are recorded per
   profile and remaining profiles continue.

## Results

- Stdout and `summary.jsonl`: compact results, outputs, semantic flags, reference
  comparisons, token counts, cleanup and final review summary.
- `raw.jsonl`: full launches, explicit input token sequences, long input text,
  native replies and errors. Native replies also retain stopping information.
- `worker.log`: redacted native/supervisor debug log for all profiles.
- `NN-psm-load.log` / `NN-spm-load.log`: per-worker captured load diagnostics.

Exit 0 means the **harness, protocol and cleanup** checks passed. It does not mean
every model/order produced correct FIM. Inspect `semantic_matches`,
`semantic_total`, `review_required`, per-profile observations and actual outputs.
Reference/count differences are reported for analysis, not silently repaired or
ignored. Native version changes can invalidate the formatter hypothesis.

Send the summary and, preferably, archive the whole evidence directory. These
measurements will determine public suffix mapping, unsupported-model errors,
ordering, stopping and long-context policy in the next checkpoint.
