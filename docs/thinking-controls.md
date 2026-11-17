# Thinking controls

Ollama chat/generate `think` and OpenAI chat `reasoning_effort` /
`reasoning.effort` use a shared resolver. These request-level controls are
separate from assistant `message.reasoning` history and the output parser.

## Selection and warnings

Nonempty `reasoning.effort` takes precedence over `reasoning_effort`. Missing,
null, empty and whitespace-only values fall through to the other field. If
both root keys were explicitly supplied and their normalized values differ,
one warning records the values and selected source. Equal qualified aliases
do not warn. `reasoning: {}` does not hide a meaningful flat value. If neither
has a meaningful value, the existing template default remains untouched.

Strings are trimmed and lowercased as in the existing template controls.
`none`/`false` explicitly disable; `true` enables. Wrong types remain request
errors even on the losing side, before admission, load or keep_alive=0 retirement.
No conflict error is introduced. The selected field's later model fallback
never causes selection to switch back to the other request field.

## Qualified versus unknown contracts

Only exact SHA-256 identities of the following templates qualify a closed
contract. Model display names and reasoning markers are not used. GPT-OSS also
requires the GGUF architecture `gpt-oss`; the named levels are a model contract,
not an inference from the template accepting arbitrary strings.

| Template from the pinned native source | Contract | Default | Aliases |
| --- | --- | --- | --- |
| `Qwen-Qwen3-0.6B.jinja` | boolean on/off | on | recognized positive efforts select on |
| `Qwen3.5-4B.jinja` | boolean on/off | on | recognized positive efforts select on |
| `openai-gpt-oss-120b.jinja` with `arch=gpt-oss` | low / medium / high, cannot disable | medium | minimal → low; xhigh / ultra → high |

Supported exact names take precedence over aliases. A value outside a qualified
contract falls back to its actual default with a warning. Attempting to disable
a qualified always-thinking model uses its lowest level with a warning. For the
qualified GPT-OSS contract, `max` therefore falls back to medium; it is not an
alias for high. Reasoning output is never suppressed to pretend thinking is off.

Unknown or edited templates retain the existing backend passthrough, including
custom levels. Absence of a value in incomplete metadata is not evidence of
non-support. No request is rejected merely because our model knowledge lacks
the value. New templates must be qualified before adding another closed profile.
These three template matches do not claim complete Qwen/GPT-OSS family coverage
or inference-quality/hardware qualification.

The pinned native `/props` supplies template sources, including a tool-use
variant. Its `reasoning_effort_levels` is a render probe over a finite candidate
set; it is not an authoritative exhaustive list of model-trained levels or
evidence of a model's default/ability to disable thinking. It is not used as
a restrictive allowlist.

## Effective templates and unchanged defaults

Chat uses the actual worker template; a named tool template applies only when
tools are used and no explicit TOML template overrides it. Generate uses the
actual model template or the request override. An override or changed template
must qualify independently. Both native chat arguments and the existing local
generate renderer use the same resolver; optional prefill comparison does not
duplicate warnings. Raw generate continues to own its literal prompt and does
not render thinking controls. No second renderer, HTTP field, TOML field or
startup hardware/model probe is added.

Missing/null/empty controls do not inject flags or effort or change model,
sampling, worker, KV, MTP or cache defaults. The parser still separates actual
thinking output even if the model ignores a disable request.

## Sources and reproducible qualification

Fixture templates and their MIT license are copied unchanged from native
commit `c0819bbbaeafa81e506eb83cf1b43f9f6106d307`, tree
`4e951fcb49275d0c70b539d7d5d01525d663c746`, `models/templates/`. Their exact
fingerprints are the private registry in `helpers/thinking.py`; tests exercise
the complete real templates and prove changed bytes invalidate qualification.

GPT-OSS's three levels are documented by the
[OpenAI model card](https://huggingface.co/openai/gpt-oss-120b/blob/88fa8c29c62204ea1fc3cda5718328290d148fbf/README.md).
The pinned template establishes medium as its rendering default. The
[Ollama compatibility documentation](https://docs.ollama.com/api/openai-compatibility)
documents GPT-OSS aliases and lowest-level substitution for none, and the
[thinking documentation](https://docs.ollama.com/capabilities/thinking) documents
default substitution. These online documents were consulted 2026-11-17;
the source/fixture pin, not moving documentation, identifies the qualified bytes.
