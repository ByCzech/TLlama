# Offline show diagnostics

`POST /api/show` describes the model definition's default chat configuration.
It does not describe the last request, a resident worker's live parser, or a
generate request's template override. It reads definition/header metadata only;
it does not load weights, start/probe workers, acquire leases or evict models.

For main generation definitions these fields are returned:

| Field | Value and meaning |
| --- | --- |
| `renderer` | `"llama.cpp"`: the configured default chat rendering engine |
| `parser` | `"native"`: the native output-processing path, not a particular live internal parser |
| `template_source` | `"gguf_chat_template"`, `"tllama_toml"` or `"unknown"` |
| `thinking` | Optional qualified `values` / `default` from C2; omitted for unknown or incompatible variants |

Field names match Ollama while values identify TLlama's actual configuration.
The default chat path uses native rendering; generate uses TLlama's existing
local renderer and is not described by the `renderer` field here.

Only nonempty TOML templates replace the GGUF template. Provenance follows
the definition even if the override text equals the GGUF text. Empty TOML
overrides retain the model fallback and its source. Without a known template,
`template` is empty and `template_source` is unknown; no synthetic Go template
is substituted and no unobserved native builtin fallback is claimed.

`thinking` is present only when both default and tool variants have the same
qualified C2 profile, including values, default and aliases. Unknown/different
variants omit it. Older cached metadata without tool-variant evidence omits
it as well. Absence means unknown, not unsupported. Boolean values remain JSON
booleans; named levels remain strings. No new capability profile is inferred.

Embedding, Decision (including unintegrated heads) and separately registered
projector/draft helpers omit all four chat diagnostic fields. Existing
capabilities and other show metadata keep their established contract.

Provenance is attached to a fresh per-definition result, never persisted into
the shared GGUF cache. TOML edits and aliases take effect without changing
weights, cache schemas or requiring cache removal. `modelfile` remains original
TOML, including an empty override. Requesting show does not change active
worker state or send native requests.

New parser/template previews, verbose tensor output and show request overrides
remain separate contracts. This is offline configuration information, not a
runtime/hardware qualification or full Ollama show parity.
