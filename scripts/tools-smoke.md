# Tools: native reference checkpoint 0044

Working smoke instructions; final user documentation remains deferred.
This checkpoint changes diagnostics only. The public Tools adapter is not yet
accepted; do not interpret successful native inference as public API parity.

Run from the TLlama checkout in the same Python environment as pytest. The
script resolves registered model TOML/GGUF files, starts one private worker per
model and calls its authenticated native /v1/chat/completions directly. It does
not use OLLAMA_HOST or the running TLlama service. Release the service's loaded
model first to avoid two competing loads. Output directories must be new.

```bash
pytest -q tests/test_tools_smoke.py

export PYTHONPATH=src
export TLLAMA_RUNTIME_DEVICE=Vulkan0
tools_results="/tmp/tllama-tools-reference-$(date +%Y%m%d-%H%M%S)"
python scripts/tools-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --output "$tools_results"
tar -cJf "$tools_results.tar.xz" -C "$tools_results" .
```

The existing prepared Qwen is the first target. For a second template/parser
family, repeat --model with the registered reference
`unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL`, if available
in the chosen models directory. Each worker is unloaded before loading the next.
No new download is currently required. Keep the same native runtime as the
accepted memory run: b11552-3144c167ef. Its warn/strict policy still applies.

Optional --models-dir and --executable select an alternate repository/binary.
Launch uses registered device, KV, mmap, flash attention, template and other
runtime options. Only context (default 4096), batch/ubatch (512) and request
sampling are diagnostic overrides. TOML files are never edited. The private
server's effective context and launch argv are saved. If the model TOML contains
its own device choice, that choice retains its normal precedence over env.

Request sampling is temperature=0, seed=42 and --predict=256. Thinking is
explicitly disabled except for thinking-auto, where enable_thinking=true is a
probe rather than an assumption that every model supports it. --predict,
--ctx, --load-timeout (600s) and --request-timeout (180s) are configurable.
Do not silently increase limits after a failed probe; retain the first evidence.

## Matrix and interpretation

Six probes run with both stream=false and stream=true:

- single-auto: choose the declared lookup_record among two tools;
- single-required: required call with parallel calls disabled;
- none: declared tools with tool_choice=none and an exact NO_TOOL answer;
- parallel-required: two independent keys, the same function twice;
- escaped-required: a Czech key containing quotes, backslash and newline;
- thinking-auto: separate reasoning/content/tool fields with thinking requested.

Every matching call is followed by fixed in-memory fixture results and another
request using the same stream mode. The final response must contain the actual
returned markers. This exercises call -> result -> answer, not merely a plausible
tool-call header. The diagnostic dispatches no arbitrary functions, code or
network requests; it supplies only its fixed lookup fixture after exact name
and argument checks pass. Partial/invalid/length-limited calls are never used.

Finally close a deliberately long-argument stream after its first argument
fragment, then run another tool request on the same worker. Save native logs:
connection closure and successful reuse alone do not prove native task
cancellation. Review the runtime task/cancel log before claiming it.

`raw.jsonl` saves outgoing payloads, complete responses, every native SSE JSON
event (with local elapsed_ms), independently assembled messages, semantic checks,
stream/non-stream comparisons, follow-up calls and cleanup. Each request starts
its own elapsed clock; elapsed_ms is local receipt timing, not GPU time or a
controlled benchmark. Worker transport independently verifies SSE [DONE].
`summary.jsonl` and stdout omit raw requests/events/responses and verbose props.
`worker.log` retains runtime logging; `NN-load.log` keeps each worker's load log.
Send the whole tarball, especially raw.jsonl and worker.log.

Random call IDs are compared only within a request. Cross-request comparison
uses function names and decoded argument objects, never ID equality or JSON
whitespace. Multiple call indices may interleave; do not concatenate different
calls. In this native revision IDs are cached per call, while function names
can arrive as growing full-name revisions. The diagnostic replaces those
prefix revisions rather than concatenating them; incompatible revisions fail
explicitly. Native argument strings remain exactly as received in saved messages.
Malformed JSON is recorded invalid, never repaired or semantically trimmed.

Exit 0 means the harness, recognized native responses and cleanup completed.
It does not certify model quality or public Tools support. review_required=true
retains model-semantic mismatches, unsupported combinations/HTTP 4xx and
stream differences for analysis. HTTP 5xx, broken streams/protocol or resource
failures produce exit 1. A thinking probe exhausting its token budget is review
evidence, not automatically a TLlama regression.

## Public adapter work after native evidence

The current OpenAI stream can preserve argument-string deltas but still needs
normalization if native full-name revisions arrive. A standard client may
concatenate function-name deltas and duplicate the name. The current
Ollama adapter merely copies them too, whereas Ollama's function.arguments is
an object. The production implementation must assemble one complete call per
index for Ollama, preserve text/thinking streaming, validate terminal/truncated
calls and maintain identity/history without replaying calls on errors.
The diagnostic assembler is independent and will not be imported as production
logic. Actual native evidence comes first; public mapping, client compatibility
and public cancellation tests follow in separate patches.

Sources inspected for this checkpoint:

- Native 3144c167ef: common/chat.cpp (message/history and diff parsing),
  common/chat.h (cached call IDs), tools/server/server-task.h/.cpp
  (unfiltered default diff emission), server-chat.cpp
  (index-keyed SSE deltas), server-common.cpp (tool_choice, template capability).
- https://docs.ollama.com/capabilities/tool-calling
  (argument objects and tool-result continuation, including streaming).

Accepted next feature order: Tools -> MTP -> embeddings -> decision. Change it
only when an actual dependency or runtime finding warrants discussion.
