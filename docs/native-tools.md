# Native Tools

TLlama forwards modern function tools through `/api/chat` and
`/v1/chat/completions`. The caller executes tools and supplies their results;
TLlama does not execute model-selected functions. Model/template quality and
native parser support still determine the calls produced.

OpenAI streams expose appendable name/argument-string deltas keyed by call
index, with stable IDs. Native cumulative name revisions become suffix deltas.
Ollama needs argument objects, so complete validated calls are emitted together
after native completion and transport validation, before the final stats frame.
Parseable partial JSON is insufficient: native provides no per-call commit
marker. Text and reasoning can flow earlier. Cancellation, truncation or errors
do not commit unfinished Ollama calls. OpenAI consumers must likewise wait for
successful completion before executing a streamed call.

Invalid declarations, choices and incomplete/inconsistent histories fail with
HTTP 400 before model loading. Argument JSON must describe an object; duplicate
keys and nonfinite values are refused without repair. Native JSON/schema support
remains authoritative; TLlama does not interpret arbitrary parameter schemas.

Call IDs are sent by default through both public APIs. OpenAI histories require
them. Ollama also accepts histories replayed by SDKs that discard IDs, matching
results by function name and original call order, or overall order when unnamed.
Use explicit IDs when results can arrive in a different order. The same function
may be called multiple times; a function name is not a unique call identifier.

## Offline capability advertisement

`/api/show` includes `tools` when the selected Jinja template references an
external `tools` input. A nonempty TOML `[template].jinja` overrides the GGUF
templates. Otherwise, `tokenizer.chat_template.tool_use` takes precedence over
the default template, matching native tool-template selection. An empty override
retains native's GGUF fallback. Changes to TOML apply immediately over cached
GGUF metadata. Header cache version 4 adds the named template; weight digests
are not invalidated.

The template is parsed, never rendered or executed. Comments, literals and local
variables named `tools` do not advertise support. Model names/families do not
decide it. Unknown native-only syntax/filters, malformed or oversized templates
remain unadvertised. This conservative hint does not certify model quality or
the native output parser and never blocks explicit tool requests. Native
`/props.chat_template_caps` supplies separate live template analysis after load;
`/api/show` does not start a worker to obtain it.

## Verification

See [tools-api-smoke.md](../scripts/tools-api-smoke.md) for independent raw HTTP
and actual SDK cycles, streaming, cancellation, repeated-function matching and
runtime cleanup. Optional SDK packages are test dependencies only. The captured
Qwen runtime can preserve an escaped key differently from the requested key;
this is recorded as a native/model semantic mismatch, never repaired by TLlama.
