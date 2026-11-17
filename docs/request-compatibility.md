# Public request compatibility

## OpenAI chat controls

Assistant messages accept `reasoning` as the canonical reasoning-history field,
matching Ollama's OpenAI-compatible chat interface. `reasoning_content` is an
input-only alias. Both names become one internal `reasoning` value; equal
non-null values are accepted, different values return HTTP 400 before model
admission. Null means absent; an explicit empty string is preserved and
conflicts with nonempty text. Values must be strings or null.

**Upgrade note for 1.6.0:** full replies now return `message.reasoning` and
streams return `delta.reasoning`, replacing `reasoning_content`. Only the
canonical name is emitted; clients reading the old response field must update.
Reasoning remains separate from answer `content`, including reasoning-only
and tool-call turns. A returned assistant message can be replayed unchanged.
This is an Ollama-compatible extension, not an official OpenAI Chat Completions
message field. The pinned native worker still uses `reasoning_content` privately.
Ollama `/api/chat` and `/api/generate` continue to use `thinking`.

Message reasoning text is distinct from the request-level `reasoning` object
and `reasoning_effort` thinking controls. A nonempty `reasoning.effort` wins;
missing/null/empty/whitespace falls through to `reasoning_effort`. Different
explicitly supplied controls warn, equal normalized controls do not. Unset
controls preserve defaults. Unsupported values in an exactly qualified model
contract use its default, and unavailable off uses its lowest level, with
warnings; unknown templates/custom levels pass through to the backend.
See [thinking-controls.md](thinking-controls.md) for qualification boundaries.

`POST /v1/chat/completions` accepts request-local `seed`, `frequency_penalty`
and `presence_penalty`, including explicit zero values. They override model
sampling defaults without editing the definition or reloading its worker.
Omitting them preserves the configured defaults. Native seed bounds are -1
(random) through uint32; penalties must be finite numbers.

For streams, `stream_options: {"include_usage": true}` adds one final usage
frame with empty `choices`, after the finish frame and before `[DONE]`. Omitted
or false leaves that frame absent. Native statistics are still collected and
validated internally in either case. Non-stream results retain their usage.

The Ollama-compatible `keep_alive` extension controls the admitted generation's
residency, not native sampling. Zero retires it after inference and existing
leases drain; negative values pin it. Missing values retain the server's
existing defaults. Invalid controls are rejected before model admission.

## Ollama embedding load controls

Both `/api/embed` and legacy `/api/embeddings` accept `options.num_ctx`,
`num_batch`, `num_gpu`, `main_gpu`, `use_mmap` and `num_thread`. They use the same
validation and immutable worker reload/reuse policy as generation requests.
The overrides also apply to empty-input preload requests; they do not edit TOML.
Sampling, context-shifting and MTP options are rejected for embeddings, including
before an empty-input unload. Vector dimensions and normalization are unchanged.

## Unintegrated generation controls

Generation routes recognize `logprobs`, `top_logprobs` and
`_debug_render_only`. Enabling them returns HTTP 400 before loading, unloading
or generating; they are not silently discarded. False/null flags and zero
`top_logprobs` remain compatible with clients sending disabled defaults.
Boolean controls are strict, and `top_logprobs` must be an integer from 0 to 20.

Ollama chat/generate also reject explicit top-level `truncate` and `shift`
booleans until their public policy is integrated. Null/omitted values leave
the existing `num_keep` and runtime `context_shift` contract unchanged; see
[ollama-options.md](ollama-options.md). These refusals do not emulate a native
feature or alter the accepted upstream `num_predict=0` behavior. Other unknown
SDK annotations retain the existing permissive schema behavior.

## Offline model information

`HEAD /api/version` and `HEAD /api/tags` provide the corresponding GET headers
without a response body. Listing and show do not load inference workers.

`/api/show` includes the current TOML system prompt, the definition's modification
timestamp, compact metadata for the configured projector, and the GGUF
`general.license` identifier when present. This identifier is not fetched full
license text; absent licenses remain empty. License name/link metadata is
available in `model_info`. TOML edits take effect without rewriting a GGUF cache.
Header cache schema 8 refreshes old metadata; expensive content digests retain
their independent cache version. `modelfile` remains the original TLlama TOML.

This is still compact model information, not full Ollama show parity: verbose
tensor inventories, arbitrary show-time options and multi-runner manifests
remain separate work. No new API route is claimed by these metadata additions.
