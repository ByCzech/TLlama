# Native vision

Image inference is restored on `/api/chat` and `/v1/chat/completions`, using the
same native lease, transport and result/stream adapters as text. The Python
process validates and maps input; the pinned native worker owns image decoding,
projector execution and model templates. No Python inference handler or second
engine is introduced. Images in `/api/generate` remain a separate checkpoint.

## Input and errors

Ollama message `images` accepts bare base64 or base64 image data URLs. OpenAI
`image_url` parts accept the same representations, including the existing
string-valued compatibility shape. Normalization produces canonical image data
URLs with MIME types read from the decoded bytes. Data URLs are now checked as
strictly as bare base64: exactly one comma, an image MIME header, base64 encoding
and a recognizable image signature. A mismatched image label is corrected from
the bytes. The worker remains the authoritative decoder for actual image data.

Only text and image_url content parts reach chat inference. Remote URLs,
file:// paths, malformed base64, missing URLs and unrecognized signatures return
HTTP 400 before model admission. Other media types and externally supplied
media_marker parts are refused rather than forwarded to native media loading.
No TLlama or worker image fetch is performed for these refused inputs.

Text/image part order and message history are preserved. If an Ollama message
contains both content parts and an images list, the attachments follow its
existing parts. Each image occurs exactly once for each supplied part/attachment;
no deduplication changes the caller's requested image sequence. Plain text-only
parts retain the existing flattening behavior. Name, tool history and structured
reasoning history keep their previous mappings.

Valid image requests acquire a model lease, then require the admitted worker's
`/props.modalities.vision` to be the boolean true. Missing, false or malformed
capability values return HTTP 400 before public stream headers, with no inference
POST. A configured projector cannot bypass this check. There is no silent text
fallback. Header-recognizable but undecodable image data can fail later in native
inference: non-stream calls preserve the upstream HTTP error; streams already
started return the public error frame without a successful terminal.

## Configuration and reporting

The lease retains its TOML template, default system prompt, sampling and stop
configuration. Native chat renders the image markers using its selected template;
TLlama does not flatten media into a text prompt or tokenize image data itself.
Thinking and structured output use the existing native mappings. Token counts,
cache counts and durations come from native request-local results, with no
Python estimate of image tokens or binding counter correction.

`/api/show` restores configured vision without loading weights: it requires a
valid projector definition and does not advertise an explicitly audio-only
projector. This describes the definition, not a successful hardware/runtime
probe. Actual inference always checks the selected generation's live capability.

Memory uses the selected native inventory when available; complete projector
allocations can contribute known totals. A load-log-only source may report just
a projector tensor summary and must retain partial/null totals. Neither source
measures physical residency. See [native-memory.md](native-memory.md) and
[runtime-audit-070.md](runtime-audit-070.md) for current evidence and fallback
limits. The Python inference binding has been removed; FIM has its separate
[text suffix contract](native-fim.md).

## Verification

Tests cover strict data URLs, MIME canonicalization, mixed parts/attachments and
history order, text-worker refusal before headers, live capability versus config,
offline `/show`, TOML template/system/stop preservation, native usage, changing
images on one generation, decoder errors, stream/non-stream cancellation,
protected retirement and the actual public TCP smoke entry point. Local fake
workers decode the generated RGB PNG panels to distinguish payloads; this proves
transport and mapping, not real model recognition.

Run complete pytest, then the installed runtime on Vulkan:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 TLLAMA_CONTEXT_LENGTH=4096 PYTHONPATH=src \
  python scripts/vision-smoke.py \
    --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-mmproj-IQ3_S \
    --log vision-vulkan.log
```

The supplied vision TOML has no active n_ctx, so this transient global context
setting selects 4096 for all requests without editing the definition. Its K/V
types remain q8_0/q4_0 and its projector remains Local/qwen3.6-mmproj-Q8_0.gguf.
The ready stage prints requested/effective context, fingerprint, actual
modalities, argv and PID/generation.

The script generates valid PNG panels (red/blue, then blue/red) in memory, with
no new image dependency. For both APIs it checks non-stream and stream JSON
answers. The prompt names no colors and the schema accepts arbitrary strings;
the grammar cannot force the expected answer. It also exercises thinking,
stream cancellation, another image request on the same generation, honest
projector memory reporting and final unload/process/directory cleanup.

Expected stages include show-without-load, ready, ollama-image-0/1,
openai-image-0/1 and their streams, vision-thinking, vision-stream-cancel,
after-vision-cancel, ps-vision, worker-closed and manager-closed. A thinking-only
length response is valid. The swapped image must produce the swapped color
answer. Send pytest and JSON output; attach vision-vulkan.log on failure.
