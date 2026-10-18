# Public FIM suffix smoke (0040)

From the checkout, with the configured model store and native runtime:

```bash
pytest -q tests/test_native_fim.py
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/fim-api-smoke.py \
  --output /tmp/tllama-fim-public-v1
```

The output directory must be new. The script starts a private loopback TLlama
server and its own workers; it does not use or alter a running TLlama instance.
It does not edit model TOML or native sources. Device/backend, caches, adapters,
batch and other load settings follow TOML/environment; `--ctx` overrides context
for the smoke (default 4096). `--models-dir`, `--executable`, `--load-timeout` and
`--request-timeout` are optional. `--predict` defaults to 64.

By default it exercises CodeGemma 2B, StarCoder2 7B, the specialized Qwen2.5 Coder
7B fine-tune, and DeepSeek Coder V2 Lite Instruct. Codestral and Llama 3.2 Instruct
are expected to return unsupported FIM HTTP 501. Codestral is deliberately not
mapped to a generic PSM sequence: its official recipe and output boundaries need
further investigation. Its ordinary completion must still work afterward.

Any explicit selection replaces the whole default matrix; repeat either option:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/fim-api-smoke.py \
  --model codegemma-2b-Q8_0 --model starcoder2-7b_Q4_K_M \
  --output /tmp/tllama-fim-public-small
```

Keep StarCoder2's already verified tokenizer configuration in its model TOML:

```toml
[runtime.kv_overrides]
"tokenizer.ggml.pre" = "starcoder"
```

The evidence directory contains `results.jsonl` (also stdout), `worker.log`, and
per-model `NN-load.log`. Return the complete archive:

```bash
tar -C /tmp -cJf /tmp/tllama-fim-public-v1.tar.xz tllama-fim-public-v1
```

## Acceptance scope

- The same six arithmetic/Python/TypeScript probes from the reference checkpoint
  go through public `/api/generate` with `suffix`, in both stream modes. Fixed
  greedy sampling is used; the chosen model TOML's stop defaults still apply.
- Public output must equal an independently composed full token `/completion`
  reference on the same admitted worker. Input counts include evaluated and
  cached tokens; a longer probe must retain the entire prefix and suffix.
- Semantic AST/exact-identifier matches are recorded separately. Generated code
  is never executed. A transport pass does not imply a model-quality pass.
- The script closes a public stream after its first content and checks worker
  reuse. `terminal_seen` records whether that public frame was already terminal;
  the script does not claim the GPU was necessarily still generating when the
  connection closed. Local subprocess tests separately hold the native stream
  open and verify upstream cancellation/lease release.
- Unsupported FIM must be HTTP 501 before streaming headers/inference, followed
  by successful ordinary completion. Workers must exit cleanly and remove their
  temporary directories; the final manager residency must be empty.

## Public suffix policy

A nonempty `suffix` selects FIM. An omitted, null or empty-string suffix keeps
ordinary generate semantics, matching Ollama's Go client, which serializes an
empty suffix even for preload/unload requests. With no prompt or images, that
request is load/unload control rather than FIM inference. An empty prefix with
a nonempty suffix remains valid FIM, including with `keep_alive=0` (infer, then
retire). Since 0055, an empty suffix no longer selects prefix-only FIM.

Capability comes from a compact snapshot of exact, unique GGUF vocabulary
entries, checked against the live worker tokenizer. It does not depend on model
filenames, chat-template guessing, or load logs. CodeGemma, Qwen, StarCoder and
DeepSeek marker sets select the generic recipe. Missing/ambiguous recipes and
`disabled` return HTTP 501. A live vocabulary mismatch is a protocol error.
Metadata cache version 3 refreshes old inventories; digest caches remain valid.

The default is full PSM `BOS? PRE prefix SUF suffix MID`. Literal text is tokenized
with special-token parsing disabled. BOS follows the loaded tokenizer's actual
policy; an automatically appended EOS is omitted. No synthetic repository/file
headers, batch-window clipping, or semantic output trimming are applied.

Optional per-model configuration:

```toml
[fim]
order = "auto" # auto | psm | training_spm | native_spm | disabled
```

| Order | Tokens after optional BOS |
| --- | --- |
| `auto`, `psm` | `PRE prefix SUF suffix MID` |
| `training_spm` | `PRE SUF suffix MID prefix` |
| `native_spm` | `SUF suffix PRE prefix MID` |

The SPM variants are explicit experimental choices, not default recommendations.
`runtime.spm_infill=true` alone is refused for public suffix with a configuration
hint: it controls native `/infill`, whose layout must not silently select the
public recipe. An explicit `[fim].order` takes precedence for the public path.

Input length, including framing, must be **less than the actual native slot
context**. Overflow returns HTTP 400 before completion; nothing is clipped.
`num_predict` retains ordinary generate semantics and sampling precedence.
Generation can end with `done_reason="length"` at its output/context limit; the
output budget is not reserved by truncating the input.

Request `raw=true`, `template`, `system`, images, nonempty legacy `context`,
thinking, and chat template variables are refused with HTTP 400. Model chat
system/template defaults are bypassed for FIM. `think=false` or absence is valid;
completion output is literal code, including text resembling reasoning markers.
`format`/JSON Schema, sampling defaults/request overrides, statistics,
streaming, `keep_alive`, cancellation and error handling use the normal native
completion path. Mandatory model FIM/file delimiters supplement the normal stop
precedence and cannot be removed by a request stop override. Stops happen in the
native engine via `stop` plus `preserved_tokens`, including CodeGemma's NORMAL
file token; TLlama does not strip the output afterward.
