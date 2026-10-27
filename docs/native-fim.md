# Fill-in-the-middle

A nonempty `suffix` on `/api/generate` selects code insertion between literal
`prompt` and `suffix`. The output is the insertion alone. Omitted/null/empty
suffix keeps ordinary generate semantics, including empty-prompt load control.
An empty prefix with a nonempty suffix is a valid FIM request.

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"REGISTERED_FIM_MODEL","prompt":"def add(a, b):\n    return ","suffix":"\n","stream":false,"options":{"temperature":0,"num_predict":32}}'
```

Choose a model with a verified FIM recipe; this example does not certify the
resulting code. There is no public `/v1/completions` or native `/infill` proxy.

## Recipes and boundaries

Capability comes from exact, unique GGUF vocabulary entries checked against the
live tokenizer. Supported marker inventories cover CodeGemma, Qwen, StarCoder
and DeepSeek recipes. Names or chat templates do not select capability.
Missing/ambiguous inventories and explicitly disabled recipes return HTTP 501;
a vocabulary mismatch is a protocol error. Codestral has no generic mapping.

Optional model configuration:

```toml
[fim]
order = "auto"
```

| Order | Tokens after the tokenizer's optional BOS |
| --- | --- |
| `auto`, `psm` | `PRE prefix SUF suffix MID` |
| `training_spm` | `PRE SUF suffix MID prefix` |
| `native_spm` | `SUF suffix PRE prefix MID` |
| `disabled` | Refuse FIM |

`auto` selects PSM. SPM variants are explicit experimental choices.
`runtime.spm_infill=true` without an explicit public recipe is refused, since
native `/infill` has a distinct layout. Literal code is tokenized without special
parsing; the tokenizer's automatic terminal EOS is omitted. No synthetic
repository headers, batch-window clipping or semantic output trimming is applied.

Framed input must be strictly shorter than the actual native slot context.
Overflow is HTTP 400 before completion; input is not truncated to reserve an
output budget. Output/context exhaustion can end with `done_reason="length"`.
Native stop strings and preserved FIM/file delimiters supplement ordinary stop
precedence and cannot be removed by a request override. TLlama does not repair
or strip generated code afterward.

`raw=true`, template/system overrides, images, nonempty legacy context, enabled
thinking and chat-template variables are refused. Model chat template/system
defaults are bypassed; absent/false thinking is valid. Sampling, format/schema,
streaming, statistics, keep_alive and cancellation use the completion contract.
Code resembling a thinking marker remains literal code.

## Protocol evidence and quality

[The public diagnostic](../scripts/fim-api-smoke.md) compares six tasks in both
stream modes with an independently composed full-token native reference. It
records content checks separately, without executing generated code. Prompt
reuse must be aligned for meaningful deterministic path comparisons; disabling
prefix reuse does not disable the K/V cache used during inference.

Native 0.7.0 MI50 Qwen results matched publicly/natively in all 12 responses after
aligning prompt reuse. Content checks passed 0/12 with K/V q4_0 and 2/12 with
K/V f16, keeping Q4_K_M weights unchanged. K/V precision affected results; parity
alone does not establish correct code or a model-only root cause. CodeGemma
protocol and an unsupported Llama control have separate reviewed evidence.
See [runtime-audit-070.md](runtime-audit-070.md).

Selection of reliable reference models continues separately from TLlama runtime
qualification. Supported framing is not a general FIM quality guarantee, and
this documentation does not decide that work's release disposition.
