# Embedding API — text 0058 / inline images 0064

Requires an explicitly configured embedding worker from
[native-embeddings.md](native-embeddings.md). Public text adapters now use its
managed lifecycle; native build b11552-3144c167ef remains the hardware reference.
No native patch, rebuild, download or automatic TOML edit is included.

0063 also permits text requests on explicitly configured embedding workers with
an attached projector. Known pooled semantic output is still required; a
projector does not select mode/pooling or bypass Decision-head rejection.
`/api/show` reports `embedding` for their semantic output contract. In0064 the
modern routes also accept inline images, as specified below. Offline capability
reporting does not claim a loaded vision encoder; media inference additionally
checks the admitted projector and live native `modalities.vision is true`.
See [native-embeddings.md](native-embeddings.md) for explicit configuration.

## Public contracts

| Endpoint | Input | Output | Length handling |
| --- | --- | --- | --- |
| `POST /api/embed` | `input`: text, `{text?, image?}`, or a mixed array | `model`, normalized `embeddings`, nanosecond durations, `prompt_eval_count` | Text defaults to `truncate=true`; images require `truncate=false` |
| `POST /api/embeddings` | `prompt`: one text | One unnormalized `embedding` | Over-limit input reports 400 |
| `POST /v1/embeddings` | `input`: text(s), token-ID array(s), or the explicit image content extension below | OpenAI list/data/index/model/usage envelope | Over-limit input reports 400; no truncation option |

All are non-streaming JSON endpoints. Ollama also accepts `runner`, `keep_alive`
and `options.num_ctx`. Other options and unknown fields are rejected rather
than ignored. OpenAI accepts `user` as caller metadata, without altering the
embedding input; it has no request-level runner, keep_alive or generation options.
Malformed requests use the addressed API's 400 error shape before model work.
Missing registered models return 404. Native request failures keep their status;
malformed upstream vectors or accounting become 500, never partial success.

Ollama scalar empty/null/omitted input and empty arrays are load control, with
an empty vector array and zero evaluated tokens. Empty legacy prompt is load
control with `embedding: []`. Zero keep_alive unloads idempotently without
starting a worker, including a missing name. Nonempty inference with zero
keep_alive finishes before retiring the worker. OpenAI refuses empty inputs.
An empty text *inside* an Ollama batch remains an item, using the tokenizer's
actual special-token behavior; it is not converted into load control.

## Inline image inputs — 0064

Ollama follows the image subset of its current Go API request contract:

```json
{"model":"registered-embedding-model","input":{"text":"a literal query","image":"<base64 image bytes>"},"truncate":false}
```

Each object owns one image, with optional text placed before it. Bare base64
and inline image data URLs are accepted. A mixed array of strings and objects
returns one vector per entry, in request order. A text-only object is equivalent
to its literal text; empty objects/text objects are invalid, not load control.
Audio/video and unknown object fields are refused, never discarded.

The OpenAI-compatible route has an explicit **TLlama/native extension**;
OpenAI's standard embeddings API and SDK input type describe text/token input:

```json
{"model":"registered-embedding-model","input":{"content":[{"type":"text","text":"before\n"},{"type":"image_url","image_url":{"url":"data:image/png;base64,<bytes>"}},{"type":"text","text":"\nafter"}]}}
```

Content parts preserve order and literal text, including multiple images and
interleaved text. Arrays may mix nonempty strings and content objects. Raw
native `prompt_string`/`multimodal_data`, client media markers, remote/file URLs,
audio/video, image detail hints and other fields are rejected. No fetch,
chat template, instruction, prefix or separator is inserted by TLlama. The
loaded native media marker is reserved inside media content text. Text-only
content concatenates its text parts and follows the existing text tokenizer.
Legacy `/api/embeddings` remains one text prompt.

Image requests allow at most32 entries,32 images,4MiB decoded bytes per image,
16MiB decoded media in total and64 content parts per entry. Encoded length is
bounded before decoding; image signatures and strict base64 are checked before
model admission. These are byte/item bounds, **not** pixel, image-feature or
peak-memory estimates. Native remains the decoder and image preprocessor, using
the resolved projector options (including image token settings); no launch
setting is silently changed. Malformed image bytes that pass signature checks
may still be refused by native. The pinned decoder's exact HTTP500 message
`Failed to load image or audio file` becomes public400 on an image item only.
Unrelated upstream500s remain500.

There is no literal multimodal token-count preflight in native3144c167ef:
`/tokenize` is text/token-only, and chat count routes apply a chat template.
Consequently image entries go unchanged to native, one entry per private
request. Native enforces its own context checks; afterward its mandatory,
positive, consistent `usage` is checked against the same conservative per-item
bound below and the300000-token request bound. An excessive returned count is
public400 **after native work**, without a partial batch response. Missing or
malformed media usage is500; text tokens never substitute for image counts.
No exact multimodal preflight or automatic truncation is claimed. Ollama image
requests require explicit `truncate=false`, including mixed text/image batches.
Any ordinary text entries still use exact native tokenization and preflight.

Use OpenAI Python SDK `extra_body={"input": ...}` for this provider extension;
it does not establish hosted OpenAI image-embedding support. An older Ollama
SDK may reject objects locally despite the current server contract. The public
diagnostic records installed versions and high-level support, and separately
labels fallback SDK HTTP transport/response parsing. It does not call that
fallback high-level `embed()` compatibility.

## Tokenization and limits

Text is tokenized by the admitted worker with add_special/parse_special enabled.
The exact token arrays go to native, avoiding a decode/re-tokenize cycle and
duplicate special tokens. Caller text, literal escapes, Unicode and query/
document prefixes are preserved; no model-name-dependent prompt is inserted.

The current per-item limit is the minimum of the effective worker context,
the trained context when known, configured n_batch and n_ubatch. This is a
deliberately conservative bound for all pooled models: native does not expose
whether this loaded model can split a sequence across physical batches.
For LAST pooling, the effective worker context contributes `n_ctx - 1`: the
pinned server's memory-backed splitting path rejects `n_tokens >= n_ctx`.
Mean/CLS retain the inclusive context bound. Native props do not expose memory
presence, so the one-token reserve is conservative for any configured LAST
output. It is selected from the verified pooling contract, never a family name.
Only the worker context is reduced; a smaller trained/batch/ubatch bound stays
inclusive. Thus context 256 permits 255 LAST tokens or 256 mean/CLS tokens when
the other bounds allow it. Ollama truncation and reported usage use that exact
submitted limit; token-array, OpenAI text and legacy inputs above it get 400
before native inference. This corrects the boundary found in hardware run 0059.
A causal decoder may support more than this bound natively. Increase the
configured physical batches when appropriate; no silent launch tuning or
architecture-name heuristic is used here.

Ollama truncation removes the tail of content tokens while retaining the
tokenizer-added prefix/suffix (for example CLS/SEP). If those affixes cannot be
identified unambiguously, truncation reports 400. It does not silently remove
the required terminal special token. OpenAI and legacy requests never take
this truncation path.

OpenAI token IDs are exact final local-model token sequences. They receive no
extra BOS/EOS; callers own those tokens. IDs must be nonnegative integers below
the GGUF vocabulary size, with no booleans, floats or mixed input shapes.
Token arrays require known vocabulary metadata; native decode never receives
an unvalidated caller-supplied ID. Batch limits are 2048 inputs and 300000
submitted tokens. The input token limit is the local model/load limit above,
not a hardcoded token limit borrowed from a hosted OpenAI model.

Private text inference batches are split to keep each upstream JSON response within
a conservative byte budget. Indices are validated and restored to input order
across those batches. Every result has the full configured dimension and finite
float32 values. Modern outputs must already have unit L2 norm; TLlama validates
that property and does not repair invalid vectors. Legacy requests explicitly
select native embd_normalize=-1; modern/OpenAI select 2. Reported native token
counts must match submitted tokens. If native omits usage for a text-only
item, counts come from those exact submitted IDs, never character estimates.
Image items instead require authoritative usage as described above.

## Dimensions and encoding

`dimensions`, when supplied, must equal the full GGUF output dimension.
Use a positive architecture-specific `embedding_length_out` when declared;
otherwise absent/zero uses `embedding_length`, matching native n_embd_out().
These can differ: the supplied EmbeddingGemma 2 has hidden width 512 and a
learned projection to 768 output values. Positive explicit output width is
authoritative across preflight, vector validation, chunk sizing and base64;
the hidden width is not an alternative supported `dimensions` value.
Malformed explicit output metadata is rejected rather than ignored. Metadata
cache schema 6 refreshes older snapshots missing this key; digest caches stay
valid. This is declared full native output, with no response-based size guessing.
Reduction is explicitly unsupported at this checkpoint: no arbitrary vector
slicing, padding or unverified Matryoshka claim. Other positive dimensions
receive 400 before embedding inference. A mismatched native output dimension
is a protocol error.

OpenAI `encoding_format` supports float (default) and base64. Base64 contains
the same values as little-endian float32 bytes. Native is always queried for
float vectors so shape, finiteness, normalization and accounting are validated
before encoding. Both explicit formats and the Python SDK's automatic base64
decoding are covered.

## Admission, capabilities and metadata

Public calls use operation=text_embedding: explicit embedding mode plus a
known GGUF pooled output/width and no decision-head metadata. These checks run
on the resolved immutable definition before admission or replacement. Raw
internal operation=embedding remains the 0057 transport-mode contract. Neither
a backend embedding flag nor a family name makes decision scores into retrieval
vectors. Auto pooling requires a known model pooling key; an explicit verified
TOML pooling selection can supply it. None/rank pooling is outside this API.

`/api/show` advertises only `embedding` for supported configured text models;
unknown or incompatible output metadata gets no embedding capability. The GGUF
reader adds architecture-specific pooling, width and decision keys, plus a
compact vocabulary count. It does not expose the token array. Metadata cache
schema advances to 6 so older header snapshots are rebuilt automatically;
content digest caches stay valid.

Real ASGI disconnects cancel in-flight calls and slot waiters, close private
responses and release leases. Hardware reference 0054 shows that native can
still finish a cancelled embedding batch. This API does not promise early
compute cancellation. Same-worker recovery, explicit unload and owned-directory
cleanup retain the existing lifecycle.

## Verification and hardware scope

```bash
pytest -q tests/test_embedding_api.py tests/test_native_embeddings.py
pytest -q tests/test_embedding_media_api.py tests/test_embeddings_multimodal_smoke.py
pytest -q
```

Tests use real GGUF headers, owned TCP native stubs, public ASGI/TCP requests
and optional official Ollama/OpenAI Python SDKs. They cover input/limit/encoding
contracts, normalization differences, error integrity, order/duplicates,
metadata cache refresh, admission, queue cancellation and actual disconnects.
They certify protocol/lifecycle behavior, not embedding-model quality.

0059 supplies [the public TCP/SDK diagnostic](../scripts/embeddings-api-smoke.md)
against independent MiniLM/Qwen native calls on the maintainer's hardware.
It uses temporary definitions with ordinary production resolution and checks
registered TOMLs unchanged. Keep those two models; additional architectures
have been accepted separately. Keep all six accepted text models and the
accepted0062/0063 image references. The0064 public image gate is
[`embeddings-multimodal-smoke.py --public`](../scripts/embeddings-multimodal-smoke.md),
with real loopback HTTP, independent raw-native controls, native usage and SDK
wire capture. These historical recipes are diagnostics, not instructions to
repeat accepted captures. Native 0.7.0 MiniLM/Qwen text embeddings and SDKs
passed on MI50/Vulkan; this does not establish 0.7.0 image-embedding acceptance.
Use [runtime-audit-070.md](runtime-audit-070.md) for the current reviewed scope.

Primary API references checked 2026-10-19 (the current Ollama Go type/docs also
describe image/audio objects;0064 integrates only inline images):

- https://docs.ollama.com/api/embed
- https://github.com/ollama/ollama/blob/main/api/types.go
- https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
- https://github.com/ollama/ollama/blob/main/server/routes.go
- https://developers.openai.com/api/reference/resources/embeddings/methods/create/

Pinned native source: common/arg.cpp (server flag availability),
common/common.cpp (normalization), tools/server/server-context.cpp
(embedding token input, normalization override and response construction).
