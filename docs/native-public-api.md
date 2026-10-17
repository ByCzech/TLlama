# Public native text API checkpoint (05b)

The three existing inference endpoints now use native workers exclusively:
/api/chat, /api/generate and /v1/chat/completions. There is no public legacy
fallback or selectable second engine. The backend's remaining binding imports
and reference tests are removed in block 07. Do not deploy this checkpoint as a
complete production migration: binding/application cleanup remains block 07.
Native process reporting is restored in [06a](native-memory.md).
Chat image inference is restored in [06b](native-vision.md).

## Ownership and configuration

Native text serving uses the 05a request/result adapters and ModelManager leases.
Admission happens before returning a streaming response, so a missing model or
broken definition still has its pre-header HTTP status. The response then owns
the lease through slot wait, stream iteration, early disconnect and failures
while sending headers/body. Cleanup closes the upstream iterator and releases
the lease even if the body never starts. Cancellation does not replay inference.

Equal configurations share an owned load. Explicit unload removes a generation
from selection but already-admitted requests, including slot waiters, finish.
Zero keep_alive with inference retires after the last lease. Empty Ollama
messages/prompt preload or unload without generating; zero unload of an absent
model does not spawn a worker. TTL remains completion-based as established in 04.

ModelDefinition now freezes header metadata alongside its resolved TOML and
file fingerprints. Header/TOML/file changes during resolution trigger bounded
retry. Request serving never rereads mutable definitions after acquiring its
lease. Existing valid persistent metadata caches are reused; otherwise headers
are read without writing a potentially changed file's scan into its cache.
Optional header scan failure leaves native loading authoritative and does not
invent GGUF recommendations. Storage/show/list metadata readers remain independent.

Definition deletion retires native generations both before and after the file
operation, covering overlapping acquisitions. Its owned operation completes
retirement even if the caller cancels. Existing users can finish on the retired
worker; the GGUF weights remain on disk as before.

## Generate rendering and tokenization

Generate retains raw input and the existing sandboxed Python Jinja renderer.
The renderer receives BOS/EOS strings directly from the selected worker, with
no Llama-shaped proxy. Request template/system overrides and the former aliases
(Prompt/System/Messages/Think/ThinkLevel/IsThinkSet, developer_instructions and
model_identity) retain their meaning. An explicit empty system value suppresses
the TOML default. Stop precedence remains request > TOML plus native EOS.

The effective worker template or TOML override comes from the lease, not a later
metadata lookup. Raw mode bypasses rendering and still refuses template/system/
context combinations. Both modes tokenize on that same worker with special
token parsing. The former initial duplicate-BOS rule is retained; completion
receives the resulting integer token array, which the pinned native mixed-input
parser does not prepend with another BOS. This is prompt preparation, not an
estimate substituted for request-local token statistics.

Chat uses the native template/parser. Ollama exposes separate thinking; OpenAI
exposes separate reasoning_content. Raw output uses the shared completion
reasoning splitter. JSON/schema fields reach native grammar-constrained sampling
without constructing Python grammar objects. Model-specific constrained output
and reasoning control still require the real-runtime smoke below.

## Agreed development limitations

| Input/endpoint | Checkpoint behavior |
|---|---|
| Nonempty generate suffix | 501: native FIM is a separately verified feature |
| Nonempty generate context | 400: token-history input is not silently ignored |
| Deprecated functions/function_call fields or history | 400: use modern tools/tool_choice/tool messages |
| Images in chat | Native vision since 06b; the selected worker must support images |
| Images in generate | 501: separate extension |
| /api/show vision capability | Configured vision since 06b, without loading a worker |
| /api/ps | Native generation reporting since 06a; missing memory totals are null with provenance |

Image validation precedes model loading and does not fetch URLs. Chat image
mapping and live capability checks are documented in native-vision.md.
Model listing, definitions, version/runner rules and
storage/pull remain available. Tools cycles, admission, streaming and client
compatibility are documented in [native-tools.md](native-tools.md).

Statistics keep the 05a measured-source rules: OpenAI full input usage, Ollama
evaluated input count, separate measured cache count, and ms-to-ns durations.
Unknown measurements stay absent. Native chat empty-choice usage frames are
handled internally. Public OpenAI streams retain the existing request schema
and terminal SSE shape; no new stream_options request extension is introduced.

## Verification and user hardware check

The complete local suite passes 1422 tests. Public contract tests now use native
leases/results instead of injected Python inference objects or C-counter resets.
Real subprocess/socket coverage checks the six public result/stream paths,
preload reuse, zero keep_alive, active/queued unload, client/load cancellation,
missing-model errors before streaming headers, native HTTP/stream errors, crash
without replay, snapshot edits and deletion. Send failures before body start and
during body output both release resources. The smoke CLI's success/error paths
also run over actual loopback HTTP with the native test subprocess.

Run on the refactor checkout with its dependencies installed:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/api-smoke.py \
  --model unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S \
  --log api-vulkan.log
```

The script starts its own temporary loopback TLlama frontend and one model
generation. It uses the registered model's TOML context for every endpoint,
including OpenAI, to avoid differing request overrides and reloads. With the
supplied text profile this is 4096, not the earlier 2048 worker-only test.
No model definitions are edited, extra GGUFs loaded or SDKs built. Device/TOML
precedence is unchanged. Native and public ports are dynamically assigned.

The check exercises Ollama/OpenAI chat with and without streaming, thinking
control, raw/rendered generate with and without streaming, both structured
output schemas, closing a live stream and using the same worker afterward,
then empty-prompt unload/reap. Reported stages include ready, ollama-chat,
api-chat-stream, openai-chat, v1-chat-completions-stream, generate-rendered,
generate-raw, api-generate-stream, ollama-schema, openai-schema, stream-cancel,
after-stream-cancel, worker-closed and manager-closed.

A reasoning-only length response is valid, not an inference failure. The script
prints reasoning separately; inspect the Qwen thinking-enabled stage to confirm
actual model behavior. Both schema checks must decode to exactly {"answer":42}.
worker-closed must have a reaped return code, directory_removed=true and empty
residency. Send JSON output and pytest summary; attach api-vulkan.log on failure.

The new adapters/text routers need no further hardware/backend variants for this
checkpoint. Native allocation reporting is documented in native-memory.md;
vision is documented in native-vision.md (06b). Physical-residency comparisons
are a later test campaign.
