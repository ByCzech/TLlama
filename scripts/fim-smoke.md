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
time, with independent PSM and SPM workers (Codestral and the negative use
one worker; no concurrent model loads):

| Registered model | Role |
| --- | --- |
| `codegemma-2b-Q8_0` | FIM candidate |
| `starcoder2-7b_Q4_K_M` | FIM candidate |
| `qwen2.5-coder-7b.Q4_K_M` | jsdevtom TypeScript FIM autocomplete fine-tune |
| `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L` | Instruct candidate; measure FIM rather than assume support |
| `Codestral-22B-v0.1-Q4_K_M` | Separate official Mistral FIM reference |
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
cache, no user grammar/chat template, diagnostic delimiter stops, and 64 output tokens by
default (`--predict`). These controlled probes do not measure TOML sampling or
template precedence. The second checkpoint below describes the forced-stop and cancellation exceptions.

## What is measured

1. Record the loaded vocabulary's FIM IDs/pieces from native load diagnostics,
   then verify them against the live tokenizer. This diagnostic parser is not a
   proposed production capability mechanism. `/infill` independently confirms
   missing-token refusal for the negative candidate. Missing diagnostics on a
   positive candidate are recorded and investigated; the second checkpoint below
   adds a separate Codestral recipe instead of inferring lack of model FIM ability.
2. Six prefix/suffix probes cover arithmetic, Python keys and TypeScript
   properties. Each generic probe runs `/infill` and explicit full FIM `/completion`, streamed and nonstreamed.
   Python AST inspection reports whether the generated expression fits the
   suffix. Generated code is **never executed**. This narrow check is not a model
   quality benchmark; equivalent but more elaborate code can need manual review.
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

## Second reference checkpoint (0037)

The default command is unchanged; use a **new** output directory, for example:

```bash
PYTHONPATH=src python scripts/fim-smoke.py --output /tmp/tllama-fim-reference-v2
```

Keep the registered StarCoder2 name when replacing its GGUF, or select the new
name with repeated `--model` options. As before, any explicit selection replaces
the entire default matrix. A missing pre-tokenizer warning is now recorded as a
quality limitation; it does not stop transport measurements.

Changes from the first checkpoint:

- Six probes: the original arithmetic pair, a Python dictionary-key pair, and a
  TypeScript property-name pair. In each pair the prefix is identical and only
  the suffix changes. The new pairs request `violet` versus `amber` directly,
  avoiding arithmetic as the only measure of suffix influence. Exact identifier
  checks and Python AST checks execute no generated code. These are diagnostic
  tasks, not a general model benchmark; a mismatch still needs manual review.
- Main comparisons use the loaded vocabulary's FIM delimiters as native `stop`
  strings **and** `preserved_tokens`. CodeGemma additionally uses its documented
  `<|file_separator|>`, StarCoder2 `<file_sep>`. Each additional delimiter is
  verified as one token by the live tokenizer. These are explicit diagnostic
  recipes, not production capability discovery or user TOML stop precedence.
- Before measuring quality, `stop-check` forces a prefix delimiter and each extra
  file delimiter using logit bias, in stream/nonstream. The response must be
  empty with native `stop_type=word` and the exact `stopping_word`. This detects
  engines which ignore the preservation/stop combination. No output is stripped
  after generation to manufacture a successful semantic result.
- `unstopped-baseline` retains one ordinary completion per generic profile,
  without these stops, for comparison with the first run. Cancellation also
  omits delimiter stops, so a quickly generated terminator cannot defeat the
  cancellation test. Native EOG processing remains active except in the forced
  stop and cancellation probes. Token arrays and terminal data remain in raw
  evidence, including control tokens hidden in normal text.
- `--order spm` still measures the **native** layout
  `SUF suffix PRE prefix MID`. It now also measures an independent, unclipped
  `training-spm` reference `PRE SUF suffix MID prefix`, with no synthetic header,
  streamed/nonstreamed. This is a common training reference, not a claim that
  every model supports it. PSM remains `PRE prefix SUF suffix MID`; optional BOS
  follows the observed tokenizer behavior in all generic references.
- Registered names containing `codestral` use the separate Mistral V3 reference
  once, irrespective of `--order`: `BOS SUFFIX suffix PREFIX prefix`, without a
  MIDDLE marker. The suffix follows `mistral-common`'s smiley-sentinel encoding to
  remove SentencePiece's dummy leading space. Live BOS, delimiter tokens and the
  two-token sentinel are checked before use. `/infill` is actually called in both
  modes: a model-specific missing-token refusal is recorded as
  `native-fim-unassigned`, with successful explicit `/completion` probes still
  allowed. Accepted `/infill` replies are retained separately. Arbitrary errors
  (including 404) still fail. This recipe measures the model independently of
  engine FIM assignments; it does not patch the engine. No generic batch-clipping
  comparison is claimed for this distinct format. Cancel/reuse and cleanup run.

`diagnostic-version: 2` identifies the revised evidence. Codestral native token
assignment failures and tokenizer warnings require review, but are distinct from
harness/protocol failures. `semantic_matches` counts stopped main references;
forced-stop checks and unstopped baselines are excluded. Inspect per-case and
per-reference records instead of treating the aggregate as a support rate.

Sources for the diagnostic recipes (checked 2026-10-16):

- [CodeGemma official formatting](https://ai.google.dev/gemma/docs/codegemma/prompt-structure)
  and [the user's GGUF conversion](https://huggingface.co/lmstudio-community/codegemma-2b-GGUF).
- [StarCoder2's official tokenizer configuration](https://huggingface.co/bigcode/starcoder2-3b/blob/main/tokenizer_config.json)
  lists its FIM and file tokens. This checkpoint tests the proposed delimiter
  policy explicitly; it does not infer all stopping behavior from token names.
- [Hugging Face FIM training example](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu)
  supplies the independent PSM/SPM permutation reference.
- [Mistral's tokenizer source](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/instruct.py):
  `InstructTokenizerV2.encode_fim()` and `_encode_infilling()`, inherited by V3;
  [official generated API reference](https://mistralai.github.io/mistral-common/code_reference/mistral_common/tokens/tokenizers/instruct/).
- Native `d1570abd3d`: `server-schema.cpp` accepts token strings in
  `preserved_tokens`; `server-context.cpp` renders those tokens before matching
  native text stop conditions. The forced-stop probes verify this live.


## Delimiter resolution correction (0038)

The v2b hardware run exposed two diagnostic bugs, not model transport failures:
CodeGemma and Qwen use the same PRE/SUF/MID spellings, and a delimiter's input
encoding need not be exactly one token. The revised policy prefers a declared
SEP and does not add CodeGemma's `<|file_separator|>` to Qwen.

For a multi-token input encoding, the diagnostic requests native `/tokenize`
`with_pieces`. It accepts one exact delimiter piece with only whitespace framing
around it, verifies every returned ID against the original encoding, and records
the full encoding/pieces. It never guesses that the final ID is the delimiter.
Anything else fails with the actual token/piece evidence. Native forced-output
stop checks remain mandatory: an accepted encoding is not proof that the engine
renders or stops on that delimiter correctly. No content is stripped afterward.

The evidence now starts with `diagnostic-version: 3`. The default matrix selects
`starcoder2-7b_Q4_K_M`, the user's replacement; its tested conversion still warns
about a missing pre-tokenizer. Further model replacement is not required for
these two corrected profiles. Repeat only their measurements in a new directory:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/fim-smoke.py \
  --model codegemma-2b-Q8_0 \
  --model qwen2.5-coder-7b.Q4_K_M \
  --output /tmp/tllama-fim-reference-v3
```

Keep the v2b evidence for StarCoder2, DeepSeek, Codestral and the negative model;
those completed measurements do not need repeating for this diagnostic fix.
