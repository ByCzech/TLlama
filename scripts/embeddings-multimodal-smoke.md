# Native multimodal embedding reference — 0062

This is the independent native gate before public multimodal embedding support.
The six accepted text-model runs remain valid. Production embedding workers
still refuse projectors; this diagnostic uses an owned launch subclass and the
existing private authenticated, bounded worker transport. It opens no public
listener and changes no production API, registered TOML or native binary.

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
