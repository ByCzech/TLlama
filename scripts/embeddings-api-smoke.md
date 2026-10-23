# Public text embedding / SDK comparison — 0059 / 0060 / 0061 / 0073

Run from the checkout after applying 0061 on top of 0058–0060. This gate compares real loopback TCP
requests and the official Ollama/OpenAI Python SDKs against independent private
native requests. The default models are the existing MiniLM Q8_0 and Qwen3
Embedding 0.6B Q8_0 references, sequentially, each with one fresh managed worker.
It requires both optional Python packages `ollama` and `openai` in the selected
Python environment. Missing packages stop the CLI before it loads a model;
there is no silently skipped SDK gate. No package installation, model download,
native source edit or native rebuild is performed.

```bash
result_dir=/tmp/tllama-embeddings-api-$(date +%Y%m%d-%H%M%S)
PYTHONPATH=src python3 scripts/embeddings-api-smoke.py --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

Use the same model-store/device/executable environment as the accepted 0054
reference (native fingerprint b11552-3144c167ef). `--models-dir` and
`--executable` override their usual environment settings. `--model` can be
repeated for a separately reviewed selection; defaults need no additional
models. The temporary public listener uses a kernel-selected loopback port;
it does not contact or change a running TLlama service. Its workers nevertheless
share the hardware with other workloads. Stop other inference before this run
if needed for the same memory budget as 0054.

## Exact diagnostic configuration

The tool reads each registered definition to locate its existing GGUF and
preserve model runtime placement/settings. It writes a separate `model-N.toml`
under the new results directory. Only that private manager redirects definition
lookup to this file; ordinary production parsing, fingerprinting, configuration
resolution, operation admission and worker launch remain in use. The worker
factory does not alter the resolved launch. Metadata cache writes also stay in
the results directory; registered TOML bytes are checked unchanged at teardown.
No changes are applied to the registered definitions or another server.

The temporary definition has the existing `[llm].model` and `[runtime]`, with:

```toml
embedding = true
pooling_type = -1
n_ctx = 256
n_batch = 512
n_ubatch = 512
type_k = "f16"
type_v = "f16"
flash_attn = false
spec_type = "none"
```

It omits projector, template, sampling, system and FIM sections and removes
`draft_n_max` / `spm_infill` companions from model and global overrides. No task
prefix is added. Device and remaining runtime settings follow normal global /
model precedence. Memory reporting is explicitly native; resident TTL is
unlimited until the public unload probe. Pooling stays model-defined; the
expected headers are BERT mean/384 and Qwen3 last/1024. There is no heuristic
pooling fix if the supplied artifact reports something else.

`--ctx` permits 64..16384 for a separately reviewed run; physical batch/ubatch
are at least 512 and at least that context. The default 256 matches 0054.
All successful public calls, direct native controls and SDK calls must retain
one PID and generation per model. The preload/unload paths are public too.

After boundary correction 0060, a LAST output reserves one token against the
effective native context (255 submitted tokens for context 256); mean/CLS keep
the inclusive context boundary (256). This follows the pinned server's
memory-backed `can_split()` path, which rejects `n_tokens >= n_ctx`. Native props
do not expose memory presence, so the LAST reserve is deliberately conservative.
The ready record includes pooling and context reserve. The smaller trained
limit remains inclusive. Context and physical batch sizes are not increased
to hide the boundary.

The 0059 hardware result already accepts MiniLM's full public/SDK run. Repeat
only Qwen after 0060; the original Qwen run stopped before SDK and recovery
probes at a direct native 256-token request:

```bash
result_dir=/tmp/tllama-embeddings-api-qwen-$(date +%Y%m%d-%H%M%S)
PYTHONPATH=src python3 scripts/embeddings-api-smoke.py \
  --model Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0 \
  --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

## Comparisons

Diagnostic version 3 / checkpoint 0061 reads explicit `embedding_length_out`
independently of hidden `embedding_length`, following native n_embd_out(). A
positive output value wins; absent/zero falls back to hidden width. Ready rows
record both hidden_width and output dimension. Metadata cache schema 6 exposes
this key even for a GGUF previously scanned under schema 5. This corrects the
EmbeddingGemma 2 reference (512 hidden -> 768 output) without resizing vectors.

The inventory run already accepts EmbeddingGemma 300M and Qwen8B completely.
After 0061, repeat only the existing EmbeddingGemma 2 text path. The Nomic name
from that run did not resolve to an imported definition; use its actual local
registered reference once established, without guessing a quantization/name.
No completed MiniLM/Qwen0.6B/EmbeddingGemma300M/Qwen8B repeat is needed.

- Native text versus independently measured exact final token arrays, then
  public scalar/batch/repeat/reordered/duplicate inputs, Unicode, real newlines,
  tabs and literal backslash escapes. No de-escaping or prefix insertion.
- `/api/embed` and `/v1/embeddings` versus native `embd_normalize=2`, including
  full `dimensions`, explicit float and little-endian float32 base64.
- Legacy `/api/embeddings` and Ollama SDK legacy method versus native
  `embd_normalize=-1`, preserving raw magnitude.
- Actual Ollama SDK modern call, and OpenAI SDK default automatic base64
  decoding, explicit float, explicit base64 and local final-token arrays.
  Wire request/response bodies and SDK objects are recorded separately.
- Empty string inside an Ollama batch, exactly-at-limit token input, default
  Ollama truncation with native special-token affixes retained, and 400 errors
  for over-limit inputs, malformed IDs, invalid encoding, unsupported dimension
  reduction and empty OpenAI input. Both SDK error types are exercised.
- A private native 400 and same-worker public recovery; public transport
  cancellation and recovery on all three endpoints, followed by public unload,
  process exit, private-directory removal and empty residency.

Vector inspection and comparisons reuse the independent 0054 measurement
helpers, without importing public embedding preparation/validation functions.
The direct native text path supplies an independent tokenizer/input comparison
against the public adapter's exact-token path. The truncation probe independently
measures native decorated/raw token sequences and compares the native result
for the expected retained sequence, including any suffix special token.

Default comparison tolerances are absolute `1e-4`, relative `1e-3`, unit-norm
`1e-3`; `--atol`, `--rtol`, `--norm-tolerance` change review thresholds, not the
vectors. There is no normalization repair, dimension slicing or invented usage.
Native missing usage is recorded as unknown; public usage must equal measured
submitted IDs. No retrieval quality, model ranking or multimodal claim is made.

## Cancellation and output

Diagnostic version 4 / checkpoint 0073 fixes a cancellation race exposed by
synthetic TCP tests. Public embedding work now runs in an owned AnyIO cancel
scope: disconnect or caller cancellation keeps cancellation active across
transport checkpoints, including when a transport consumes the first request
to cancel. Cleanup still waits for work and residency lease release. This does
not establish early GPU compute termination. Existing accepted vector/SDK
hardware references remain accepted; no repeated model matrix is required.

Each cancellation probe owns a direct loopback TCP connection. The tool sends
one HTTP/1.1 JSON request and explicitly closes its socket after the observation
delay, then joins its local response reader. A pooled HTTPX request task can
still wait for headers after cancel(), including a race while connecting; task
cancellation alone therefore does not prove transport closure. Completed
responses are parsed with h11 and validated; incomplete response headers and
bodies are bounded. Recovery uses the separate ordinary HTTPX client. Socket
cleanup and subsequent lease draining have separate `--request-timeout` bounds;
`transport-close` records distinguish these stages. This changes diagnostic
transport only; ordinary public and SDK requests retain their existing paths.

These responses are non-streaming and expose no reliable inference-start
event. A request may already have completed before `--cancel-after-ms` (default
5). A pending request can still be in public tokenization or native work.
`pending_at_cancel` and `leases_at_cancel` record what the harness can observe;
inference start and early native compute stop remain unknown. As seen in 0054,
native can finish a whole batch after transport closure. Recovery and cleanup
are checked, but neither is presented as proof of an early GPU stop. Do not
repeat the run merely because a fast request completed before cancellation.

`raw.jsonl` retains request bodies, complete vectors, SDK wire responses/objects,
token IDs, build/argv/props/metadata and native memory reports. Authentication
headers and native key contents are not recorded. `summary.jsonl` is also
printed to stdout. `worker.log`, per-worker load logs and temporary definitions
are retained alongside private metadata caches. Memory reports after load and
requests preserve owners and deduplicated totals; these capacities are not OS
RSS or physical VRAM peaks. Timings are not an inference benchmark.

The CLI refuses an existing output path. Exit 0 means hard protocol/lifecycle
checks completed; always inspect `review_required` and `observations` for
numerical differences or incomplete memory/usage evidence. Exit 1 means a hard
failure; exit 2 means invalid CLI/output or missing SDK prerequisites. Return the
whole `.tar.xz` result for review. Stub tests validate the harness only; hardware
acceptance of both default models precedes broader architectures, multimodal
inputs and Decision.
