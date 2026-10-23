# Ollama client parameters

The 20 fields accepted by the audited Ollama client's `/set parameter` command
are applied by `/api/chat` and `/api/generate`, for full and streamed replies.
The client stores `/set` values locally and sends them with its next request;
TLlama does not receive the command itself. This contract covers request options,
not a new Modelfile import or arbitrary CLI passthrough.

Request values override model TOML, then global settings, without writing either.
Load controls use the same immutable generation ownership as TOML changes:
a changed effective configuration retires the old worker, existing leases drain,
and new requests acquire the replacement. The same effective settings reuse it.
Omitting a previously supplied override returns to TOML/environment defaults.

| Ollama option | Native setting / behavior | Reload |
| --- | --- | --- |
| `num_ctx` | `n_ctx`, `--ctx-size`; 0 uses the trained context | On change |
| `num_batch` | `n_batch` and `n_ubatch`, matching the Ollama runner | On change |
| `num_gpu` | -1 selects native auto/fit, 0 forces CPU with offload disabled, positive values set layer count | On change |
| `main_gpu` | `--main-gpu`; native split-mode semantics apply, not a backend selector | On change |
| `use_mmap` | Mmap part of `--load-mode`; configured mlock remains independent | On change |
| `num_thread` | Positive: `--threads` and `--threads-batch`; nonpositive: native automatic defaults, overriding configured thread counts | On change |
| `draft_num_predict` | Positive N selects embedded `draft-mtp` with `draft_n_max=N`; 0 selects `none` | On change |
| `num_keep` | `n_keep`; -1 preserves the initial prompt, 0 keeps no ordinary prefix tokens | Enables context shifting when first requested; value changes do not reload |
| `repeat_last_n` | Native repeat window; -1 = context, 0 = disabled; overrides runtime `last_n_tokens_size` | No |
| `num_predict` | Positive token budget; 0 passes native zero unchanged; -1 is unbounded; -2 fills remaining context with shifting disabled | -2 can change the worker's context-shift mode |
| `seed` | Native request seed, including explicit 0 and random -1 | No |
| `temperature` | Native temperature | No |
| `top_k` | Native top-k; nonpositive disables it | No |
| `top_p` | Native top-p | No |
| `min_p` | Native min-p | No |
| `typical_p` | Native typical sampling, accepted even though the audited Ollama marks it deprecated | No |
| `repeat_penalty` | Native repeat penalty | No |
| `presence_penalty` | Native presence penalty | No |
| `frequency_penalty` | Native frequency penalty | No |
| `stop` | Stop strings; explicit empty list clears configured stop defaults | No |

`num_gpu=0` is an explicit CPU request even with a GPU in model/global settings.
Other layer counts retain the configured device or automatic backend priorities.
A positive GPU count conflicting with explicit `device=none` returns an error.
An invalid request override does not retire an existing working generation.

Integer and boolean options are validated without coercing strings, fractions or
booleans into integers. Native int32 limits apply. Batch sizes must be positive;
context/main-GPU indices nonnegative; GPU count at least -1; draft count
nonnegative; keep/repeat window at least -1. Sampling uses TLlama's existing
native validation. Errors are HTTP 400 before preload/unload or inference.

## Context handling

Workers keep shifting disabled unless configured with runtime `context_shift=true`
or an explicit request `num_keep`. The matching environment setting is
`TLLAMA_RUNTIME_CONTEXT_SHIFT=true`. Explicit `num_predict=-2` disables shifting
for that worker, so native -1 stops at context capacity instead of rejecting -2.

When an explicit `num_keep` is supplied, oversized text generate prompts preserve
the requested initial prefix (including an automatically added BOS) and recent
tail. Chat uses the loaded native template and tokenizer, removes complete oldest
user turns with their assistant/tool exchanges, and retains system messages and
the latest turn. An oversized latest turn/system returns 400. FIM prompts retain
their existing nontruncating delimiter/recipe protection. Native shifting during
generation uses the request's `n_keep` count.

The pinned native runtime does not support shifting with a projector; explicit
`num_keep` is rejected for those workers instead of silently ignored. Other
native context limitations still apply (for example, recurrent contexts without
sequence removal). Backend/model startup failure remains an error; the server
never retries with a requested MTP strategy silently disabled.

## Zero limits and upstream behavior

Ollama `draft_num_predict=0` means no speculative strategy. It is translated to
`spec_type=none`, not a zero native MTP draft limit: the audited MTP loop allocates
the MTP context and can produce its first candidate before checking that limit.
Direct TOML `draft_n_max=0` remains a native setting, not a guaranteed off switch.

Public `num_predict=0` is passed unchanged as native zero, including its output
and measured statistics. In the audited native pin `3144c167ef`, zero can return
one sampled token despite the documented prefill-only contract. The maintainer
chose to leave this behavior to upstream: TLlama neither hides the token nor
changes the counters, and has no prefill emulation or native correction requirement.
Proposed native patch 0078 is not adopted. A separate upstream bug report is
planned; see the project TODO. When upstream fixes it, native zero behavior will
flow through this same adapter.

Runner options also reach Ollama embedding workers. Generation-only modes such
as MTP remain incompatible with embeddings and are rejected by the worker's
existing mode contract; no sampling effect is claimed for embeddings.
