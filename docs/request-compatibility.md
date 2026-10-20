# Public request compatibility

## OpenAI chat controls

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
