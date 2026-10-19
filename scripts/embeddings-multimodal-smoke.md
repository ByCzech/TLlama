# Image embedding references — independent 0062 / managed 0063 / public 0064

This is the independent native gate before public multimodal embedding support.
The six accepted text-model runs remain valid. The default0062 reference uses
an owned launch subclass and the existing private authenticated, bounded worker
transport. After0063, `--managed` exercises the normal production embedding
configuration and NativeWorker instead. These two modes open no public
listener. The0064 `--public` mode adds a loopback public listener and SDK checks
after the same native controls. No mode edits the registered TOML or native binary.

The maintainer's registered reference is:

```text
mradermacher/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B.Q8_0
```

Its existing `[mmproj].model` resolves to:

```text
HuggingFace/mradermacher/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf
```

The diagnostic preserves that resolved projector and its options, device and
placement. It temporarily chooses context/batch/ubatch1024, F16 K/V, FA off,
speculation none and explicit LAST pooling. It discards generation template
overrides; all submitted content is literal, with no implicit instruction or
chat wrapper. These are protocol controls, not a retrieval-quality benchmark
or a Transformers-vector equivalence claim. The source model's
[pooling configuration](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/1_Pooling/config.json)
uses lasttoken and includes the prompt. The publisher's
[usage guide](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
describes caller instructions; semantic prompt policy stays a separate decision.

Pinned native source3144c167ef, not a newer upstream assumption:

- `tools/server/server-context.cpp::handle_embeddings_impl` accepts a string,
  exact token IDs, an object with typed `content`, or
  `prompt_string`/`multimodal_data`; it queues embedding tasks, not generation.
- `tools/server/server-common.cpp::tokenize_oai_content_array` concatenates text
  as-is, loads media and inserts the live `/props.media_marker`. It does not
  run a chat template. `tokenize_input_subprompt` independently accepts the
  matching raw marker/media representation. No image IDs are fabricated here.
- `send_embedding` reads the declared native output width. `/v1/embeddings`
  returns pooled, normalized vectors; base64 encodes float32 values.

Run from the applied TLlama checkout, in the same model-store/device environment
as the successful text inventory (for example existing
`TLLAMA_RUNTIME_DEVICE=Vulkan0`). No additional model download/build is needed:

```bash
result_dir=/tmp/tllama-embeddings-qwen3vl-$(date +%Y%m%d-%H%M%S)
PYTHONPATH=src python3 scripts/embeddings-multimodal-smoke.py \
  --model mradermacher/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B.Q8_0 \
  --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

Use a new output directory; an existing one is refused before launch. `--ctx`
accepts512..4096 and is an explicit diagnostic setting; the native projector
may cap its image token budget to the physical ubatch, which stays visible in
the retained load log. No image truncation or private token estimate is hidden
in Python. Output width comes from GGUF metadata, never a returned vector.

Two locally generated, valid PNGs swap red/blue panels. There are seven literal
content controls: text, either image, either image with the same text, and two
images in either order separated by text. Each typed entry is compared with
the independently constructed raw marker/media entry. Further checks cover
plain text/exact IDs, single/batch/repeat/reordered duplicates, float/base64,
four errors, same-worker recovery, transport cancellation, explicit unload
and cleanup. Images are inline data URLs; no remote or file media is requested.

All16 equality comparisons should pass within atol1e-4/rtol1e-3. Three separate
sensitivity observations must show measurable effects from changed images and
image order. Indistinguishable output is flagged for review, never accepted as
vision support just because HTTP succeeded. Missing usage, wrong normalization,
native500s, differences or incomplete memory telemetry are also review notes.
The deliberately undecodable PNG is one known exception: the pinned native
decoder throws `runtime_error("Failed to load image or audio file")`, which its
HTTP wrapper maps to500. That exact error is recorded as a known native
limitation and followed by recovery, not a model incompatibility. Unrelated
500s are still review notes; public error mapping remains a later gate.
Media token usage is retained exactly as returned; no text-only tokenizer is
used to count image tokens. Shape/index/nonfinite/zero-vector/capability failures
fail the protocol. Exit0 means protocol completion; acceptance also requires
`review_required=false` and review of the complete archive.

`raw.jsonl` includes requests, vectors, usage, metadata, argv/props and native
memory telemetry. `summary.jsonl` and stdout contain compact observations.
The archive retains PNGs, metadata cache, registered definition text/hash,
worker load log and full `worker.log`. Known buffer capacities are not physical
RSS/VRAM peaks; partial projector telemetry remains partial. Cancellation
records pending transport and same-worker reuse, with inference start and
early native compute stop unknown. No SDK or public multimodal compatibility
is claimed by this private-native checkpoint.

Local regression checks:

```bash
pytest -q tests/test_embeddings_multimodal_smoke.py tests/test_embeddings_smoke.py \
  tests/test_embeddings_api_smoke.py tests/test_native_embeddings.py
```

Owned TCP stubs independently decode PNG pixels and both native request forms.
They test missing vision, ignored images, drift, malformed output, missing usage,
normalization, recovery and cleanup. Synthetic vectors do not certify hardware
or model quality. Return pytest and the complete hardware tarball for review.

## Production resolution and transport — 0063

The accepted independent0062 result has16/16 equality comparisons with zero
difference and three distinct image-sensitivity observations. After applying
0063, verify the same controls through production admission and transport:

```bash
result_dir=/tmp/tllama-embeddings-managed-qwen3vl-$(date +%Y%m%d-%H%M%S)
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/embeddings-multimodal-smoke.py --managed --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

The default exact model reference is the maintainer's Qwen3-VL Q8_0 above.
`--managed` writes `managed-model.toml` in the new results directory, preserving
the registered model/projector references and projector options. Its explicit
runtime uses embedding=true, pooling_type=3, ctx/batch/ubatch1024, F16KV,
FAoff and speculationnone. Generation sections and draft/FIM companions are
omitted. Device/placement precedence is unchanged. Native memory reporting is
selected explicitly for this diagnostic; no native compute/capacity tuning is
performed to hide the accepted projector CPU fallback.

The temporary definition is resolved and fingerprinted through ordinary
ModelManager admission. The worker factory creates `NativeWorker(spec)` without
changing the resolved spec, command or embedding method. The admitted semantic
contract is checked before loading, then the live vision capability before media
requests. `raw.jsonl` records diagnostic version2/checkpoint0063 and
launch_path=production; it also retains the temporary definition. The registered
TOML's original bytes/hash and final unchanged check remain separate evidence.
The default independent recipe stays available as0062; compare retained vectors
and lifecycle evidence after the managed run. This gate still makes no public
multimodal API or SDK claim.

Focused checks for the production integration:

```bash
pytest -q tests/test_native_embedding_projector.py \
  tests/test_embeddings_multimodal_smoke.py tests/test_native_embeddings.py \
  tests/test_embedding_api.py tests/test_native_config.py
```

`tests/native_worker_stub.py` is an executable fixture launched by these tests,
not a pytest module. Its subprocess environment is supplied by the fixtures.

## Public HTTP and SDK comparison — 0064

Both private reference runs are accepted. After applying0064, run only this
new public gate; no repeat of the accepted text-model matrix is needed:

```bash
result_dir=/tmp/tllama-embeddings-public-qwen3vl-$(date +%Y%m%d-%H%M%S)
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src \
  python3 scripts/embeddings-multimodal-smoke.py --public --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

This requires the optional official `ollama` and `openai` Python packages,
already used by the public text diagnostic. `--public` implies `--managed`;
the exact registered model, projector and explicit runtime controls stay as
above. Production admission uses the temporary TOML's context directly, so
public/native requests resolve the same configuration and generation.

The version3/checkpoint0064 run retains all16 original comparisons and three
sensitivity checks, then adds35 comparisons (51 total). Seven fresh raw-native
controls reproduce the accepted content vectors and supply per-item usage.
Five Ollama controls independently follow its text-before-one-image shape;
that shape cannot express the interleaving of the OpenAI/native content extension.
Public singles, batches, repeats, reordered duplicates, full width, OpenAI
float/base64 and three SDK results must match their corresponding native
vectors and usage. There are25 successful public response comparisons,
eight public400 probes and two deliberate public TCP cancellations with
same-worker recovery. Both SDKs also parse a public400. SDK wire payloads and
responses are retained; older Ollama SDKs may use a clearly labelled transport
fallback if their high-level input validator rejects media objects. OpenAI
uses `extra_body` for the explicit provider extension, not a standard image
embedding parameter. See [the public contract](../docs/embedding-api.md).

Media requests require `truncate=false`; byte/item bounds are checked before
admission. Mandatory native image usage is checked after inference because
the pinned runtime has no exact literal-media counting endpoint. No image token
guess, implicit template, Python image truncation or silent native setting
change is used. The exact pinned undecodable-image500 becomes public400;
unrelated500s remain errors. The diagnostic keeps early compute cancellation
and inference-start observations unknown, even when a TCP request was pending.

Public unload is issued while the independent reference lease is retained;
closing that lease then completes native exit and owned-directory cleanup.
Memory is captured after public work as well, and the registered TOML must
remain byte-identical. Successful completion needs `protocol_ok=true`,
`review_required=false`, `public_api=exercised`, all51 comparisons within
tolerance, three distinct sensitivity checks and clean unload/exit. This is
still a protocol/reference comparison, not a semantic-quality benchmark or
Transformers parity claim. Video/audio are outside this checkpoint.

```bash
pytest -q tests/test_embedding_media_api.py tests/test_embedding_api.py \
  tests/test_native_embedding_projector.py tests/test_embeddings_multimodal_smoke.py \
  tests/test_native_embeddings.py tests/test_native_config.py
```
