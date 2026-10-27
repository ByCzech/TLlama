# Public native generation API

`/api/chat`, `/api/generate` and `/v1/chat/completions` use native workers
exclusively. The Python inference binding and its fallback path have been
removed. The supported routes are listed in [api-compatibility.md](api-compatibility.md);
current hardware scope is in [runtime-audit-070.md](runtime-audit-070.md).

## Ownership and configuration

Admission precedes streaming response headers, so model/configuration/startup
errors retain their ordinary HTTP status. The response owns a ModelManager lease
through slot wait, upstream iteration, disconnect and send failures. Cleanup
closes the upstream iterator and releases the lease even if the body never starts.
There is no automatic replay of inference.

Equal immutable load configurations share a generation. Changes retire the old
generation; already-admitted requests keep their snapshot and finish before
cleanup. Explicit unload removes a generation from selection without killing
protected requests. Zero keep_alive with inference retires after use. Empty
Ollama prompt/messages preload or unload; unloading an absent model starts no
worker. Idle TTL begins when active use finishes, not while inference is running.
See [native-residency.md](native-residency.md).

Definitions freeze TOML, GGUF header metadata and file fingerprints at admission.
Changes during resolution trigger bounded retry. A request does not reread mutable
definitions after acquiring its lease. Storage/list/show remain independent of
inference loading. Definition deletion retires workers and retains weight files.

## Generate rendering and tokenization

Ordinary generate supports raw text or TLlama's sandboxed Jinja renderer.
Request template/system overrides and established aliases remain supported.
BOS/EOS strings and tokenizer behavior come from the admitted worker. An explicit
empty system override suppresses the model default. Raw mode bypasses rendering
and rejects template/system/context combinations.

Rendered/raw prompt preparation sends exact integer tokens to native completion,
avoiding duplicate BOS insertion. Prompt preparation is not a substitute for
native request-local usage or timing measurements. Chat uses the selected native
chat template/parser; template overrides belong to the immutable generation.

A nonempty `suffix` instead selects the literal [FIM contract](native-fim.md).
Chat system/template defaults are bypassed for FIM; no text is silently appended
as a suffix to ordinary completion. Model quality is evaluated separately from
public/native output parity.

## Controls and restrictions

| Input | Current behavior |
| --- | --- |
| JSON/JSON Schema output | Native constrained sampling via Ollama `format` or OpenAI `response_format` |
| Thinking | Native template controls; separate thinking/reasoning and answer channels |
| Modern tools | Validated declarations/history and stream assembly; [tools](native-tools.md) |
| Chat images | Validated inline images, configured projector and live capability; [vision](native-vision.md) |
| Generate images | HTTP 501; no image silently converted into text |
| Nonempty generate context | HTTP 400; token-history input is not silently ignored |
| Deprecated functions/function_call/history | HTTP 400; use modern tools |
| Enabled logprobs/top_logprobs/debug render | HTTP 400 until integrated |
| Explicit top-level Ollama truncate/shift | HTTP 400 until integrated |

See [request-compatibility.md](request-compatibility.md) for exact disabled/default
value handling and [ollama-options.md](ollama-options.md) for accepted request
runner controls. Those controls do not edit stored definitions.

Statistics preserve measured provenance: OpenAI full input, Ollama evaluated
input, separate cache count and native ms-to-ns timing conversion. Unknown values
stay absent; zero remains zero. Public OpenAI usage frames are optional through
`stream_options.include_usage`; native usage is collected internally either way.
See [native-api-adapter.md](native-api-adapter.md) for translation details.

## Verification

Tests cover public result/stream paths, pre-header errors, no-replay failures,
lease ownership, disconnects, configuration snapshots, thinking, schema output,
FIM, tools and images. Controlled subprocesses verify protocol and cleanup;
real-model captures supply separate numerical/semantic evidence.

The text diagnostic starts its own loopback frontend and worker, without changing
the service or model definitions:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/api-smoke.py \
  --model REGISTERED_TEXT_MODEL --log api-vulkan.log
```

It checks public chat/generate streams and full replies, raw/rendered input,
thinking, JSON schemas, cancel/recovery, memory reporting and unload/cleanup.
Use an independent log/output for a new experiment and release competing service
loads first. Scripts and model names in historical qualification records are
recipes, not instructions to repeat already accepted GPU captures. Runtime/model
changes require review within their actual tested configuration.
