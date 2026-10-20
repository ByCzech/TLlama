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
