# API compatibility

TLlama implements the following public routes using private native workers.
This table describes the current application, not every endpoint provided by
Ollama, OpenAI or the native server. The default listener is
`http://127.0.0.1:54800`; OpenAI-compatible clients use its `/v1` base URL.

## Route inventory

| Method | Path | Contract |
| --- | --- | --- |
| GET, HEAD | `/` | Empty HTTP 200 reachability response |
| GET | `/health` | Frontend health; not a loaded-model/kernel qualification |
| GET, HEAD | `/api/version` | Installed TLlama Python distribution version, or `0.0.0-dev` without metadata |
| GET, HEAD | `/api/tags` | Registered model definitions and offline GGUF details |
| POST | `/api/show` | Model details, original TOML, template, capabilities and license identifier |
| GET | `/api/ps` | Owned worker generations and memory provenance |
| POST | `/api/generate` | Raw/rendered text or supported nonempty FIM suffix, full or streamed |
| POST | `/api/chat` | Chat, thinking, modern tools and configured inline images, full or streamed |
| POST | `/api/embed` | Modern pooled embeddings, batch/load controls and supported inline-image extension |
| POST | `/api/embeddings` | Legacy single text embedding with native legacy normalization |
| POST | `/api/pull` | Hugging Face GGUF download, metadata/digest preparation and definition registration |
| POST | `/api/copy` | Copy a definition while sharing weight files |
| DELETE | `/api/delete` | Retire workers and delete a definition, retaining weights |
| HEAD, POST | `/api/blobs/{digest}` | Immutable SHA-256-checked GGUF upload staging |
| POST | `/api/create` | TLlama TOML definition creation and uploaded-file import |
| GET | `/v1/models` | Registered models in OpenAI list shape |
| POST | `/v1/chat/completions` | Chat, reasoning, modern tools and inline images, full or streamed |
| POST | `/v1/embeddings` | Pooled vectors, exact local token input, float/base64 and supported media extension |
| POST | `/v1/systemone` | Text Decision extension; no standard high-level OpenAI/Ollama SDK method |

`/v1/responses`, `/v1/completions` and audio routes are not implemented.
Framework `/docs`, `/redoc` and `/openapi.json` describe the application schema;
they are not inference APIs. A private worker's `/completion`, `/infill`,
`/props` or `/tllama/memory` is not automatically a public TLlama route.

## Generation and chat

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","prompt":"Explain GGUF.","stream":false}'

curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

Replace `MODEL` with a registered name from `tllama list`. Model mode admission
happens before eviction/loading; embedding or Decision models are not generic
chat models. Empty Ollama prompt/messages are preload/unload control. Nonempty
inference with `keep_alive=0` finishes before retirement; existing leases drain.

Ollama uses NDJSON for streams, OpenAI uses SSE ending in `[DONE]`. Unknown
statistics stay absent; measured zero stays zero. Prompt/generation durations
come from native per-request timings, converted from milliseconds to nanoseconds
for Ollama. OpenAI input usage includes reused tokens; Ollama separately reports
evaluated and cached prompt counts. `load_duration` includes admission/lookup,
not just reading weights. See [native-api-adapter.md](native-api-adapter.md).

Ollama `format` accepts `"json"` or a JSON Schema object; omitted/null/empty
string leaves output unconstrained. Invalid formats fail before model work.
OpenAI uses `response_format`. Native owns constrained sampling and schema
support. Thinking selection controls templates; it cannot make an unsupported
model honor a level. Thinking and answer text have separate response channels.
See [native-public-api.md](native-public-api.md) and
[request-compatibility.md](request-compatibility.md) for controls and refusals.

A nonempty generate `suffix` selects [FIM](native-fim.md). Ordinary token-history
`context` input is unsupported. Enabled `logprobs`, `top_logprobs` and
`_debug_render_only`, and explicit Ollama top-level `truncate`/`shift`, are
refused until integrated. Accepted runner options and reload semantics are in
[ollama-options.md](ollama-options.md). `num_predict=0` retains native behavior,
including its current one-token result; it is not a no-inference sentinel.

## Runner and model management

Ollama chat/generate/show/pull may select `runner`. Omitted, null, empty string,
`llamacpp` and alias `ggml` use the same engine. Other strings return HTTP 400
unless `TLLAMA_IGNORE_UNSUPPORTED_RUNNER=true`; nonstrings always fail. Backend
names such as CUDA, ROCm and Vulkan are not runner names. `/api/tags` and
`/api/show` report `details.runner`; `/api/ps` reports top-level `runner`.
OpenAI routes do not add a runner field.

Pull accepts `namespace/repository/path/to/file[.gguf]`, adding `.gguf` when
omitted. It is not an Ollama registry tag or a Hugging Face browser URL. See
[model-repositories.md](model-repositories.md) for storage and permissions.
`/api/create` accepts TLlama TOML plus its explicit upload map; it does not parse
Ollama Modelfiles or the full Ollama structured create schema. Copy/import,
conflict, staging and deletion contracts are in [cli.md](cli.md).

## Memory and capabilities

`/api/ps` does not load a model, hash weights or extend idle TTL. `size` and
`size_vram` drive the compatibility SIZE/PROCESSOR display. With complete native
weight evidence, they use occupied weight payload plus nonweight buffers;
proved full non-input offload applies explicit display exclusions. Sparse mmap
holes/padding are not occupied payload. Detailed inventory retains those display
exclusions. Older native/log sources keep their own capacity metric. Partial
or missing evidence remains unknown; there is no fabricated physical VRAM/RSS
measurement or capacity cap. See [native-memory.md](native-memory.md).

`/api/show` capabilities are offline definition/header/template hints. They do
not prove live backend kernels, image decoding or output quality. Image calls
require live worker vision support. Runtime identity admission is shared across
all worker modes: `warn` continues with a warning, `strict` refuses unaudited
identities; malformed protocols still fail under both. Exact diagnostic pins
remain independent. See [runtime-audit-070.md](runtime-audit-070.md).

## Tools, media, embeddings and Decision

- [Tools](native-tools.md): modern declarations/choices/history; callers execute
  functions. OpenAI emits appendable argument-string deltas; Ollama commits
  complete validated objects after native completion. Deprecated functions and
  malformed/ambiguous histories are refused. Escaped-key limitations are recorded
  in the runtime audit without speculative unescaping.
- [Vision](native-vision.md): inline base64/data URLs on chat APIs with a configured
  projector and live capability. Remote URLs, local paths and unsupported media
  parts fail before admission. Generate images and audio are unsupported.
- [Embeddings](embedding-api.md): explicit embedding mode and verified pooled
  semantic metadata, literal input, exact token limits and checked native vectors.
  Modern routes support their documented image extension; legacy is text-only.
  Arbitrary dimension reduction is unsupported.
- [Decision](decision-api.md): typed choice/score/noul requests and validated
  native probabilities/usage. This is protocol equivalence, not classifier
  accuracy; transport cancellation does not prove early GPU compute stop.
- [MTP](native-mtp.md): embedded heads are opt-in and model/backend dependent.
  Public statistics report measured draft activity, not an assumed speedup.

## Clients and exposure

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Official Ollama/OpenAI Python SDK paths have reviewed capture coverage as listed
in the runtime audit. A custom base URL permits compatible clients to try the
implemented contracts; it does not certify every command or GUI/IDE version.
See [usage.md](usage.md) for examples and client boundaries.

A nonempty placeholder API key may satisfy an SDK constructor. TLlama does not
use it as public authentication. Native worker bearer tokens are private process
transport credentials. Listener, CORS and deployment controls are documented in
[configuration.md](configuration.md) and [systemd.md](systemd.md).
