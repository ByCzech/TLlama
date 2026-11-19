# Client SDK compatibility and direct HTTP recipes

This guide describes existing TLlama APIs. The reviewed third-party clients
are **Python ollama 0.6.3**, **JavaScript ollama 0.6.4** and Go `api` sources at
Ollama 0.40.16. An additional Python openai 3.52.0 mock-transport probe covered
extensions. These are dated references, not claims about the latest packages.
Serialization/parser checks do not qualify model inference. Start with the
[client examples](client-examples.md) for ordinary sync/async/stream calls.

Server-side model settings such as device placement, KV or MTP/DFlash already
apply when a standard client calls that model. A client need not expose their
configuration to benefit from it. Complete management and diagnostic access
has additional SDK limits:

| Operation | Python ollama 0.6.3 | Direct HTTP / alternative |
| --- | --- | --- |
| Chat/generate, tools, images, FIM and ordinary embeddings | Available in the compatible documented shapes | Existing examples remain valid |
| Supported `options` extensions such as min_p/spec_type/draft_num_predict | Plain dict is forwarded; `Options(...)` silently drops these unknown fields | Use a dict with server-supported keys |
| `tllama.speculative`, cached prompt count and extended show/ps fields | Response models discard these fields, including in stream chunks | Read raw JSON/NDJSON; model_dump cannot restore discarded data |
| Detailed ps | `ps()` has no `detailed` argument | `GET /api/ps?detailed=true` |
| TOML create/import | No definition/uploads arguments | TLlama CLI create or the documented `/api/create` body |
| Copy with mmproj/draft/force; delete with remove_unused_helpers | Extended arguments are absent | Existing management HTTP payloads |
| Blob upload | create_blob inherits JSON Content-Type for raw bytes | POST octet-stream to `/api/blobs/sha256:HEX` |
| Image embedding objects / list message content | Narrow SDK input models reject some valid TLlama shapes | Direct documented HTTP contract |
| System One | Method exists; scalar JSON states such as bool may fail SDK typing | Text-only `/v1/systemone` with model/state/questions; no keep_alive |
| Health/version/HEAD | Some public Python methods are absent | Existing HTTP routes |

JavaScript 0.6.4 preserves more response JSON at runtime, but its TypeScript
types do not describe all TLlama data. Some methods rebuild request bodies:
delete drops remove_unused_helpers, ps omits the detailed query, and create
adds `stream=false`, which TLlama create rejects. A type cast cannot repair a
request the method did not send. Go options are a map, while its response
structs omit TLlama extensions. Do not infer the behavior of unaudited Java,
.NET or Rust wrappers from another language's client.

## Python options: use a mapping

```python
from ollama import Client

client = Client(host="http://127.0.0.1:54800", timeout=300.0)
reply = client.chat(
    model="MODEL",
    messages=[{"role": "user", "content": "Explain GGUF briefly."}],
    options={"min_p": 0.05, "num_predict": 128},
    stream=False,
)
print(reply.message.content)
```

For a profile with a configured compatible external DFlash helper, the same
mapping can contain `spec_type="draft-dflash"` and `draft_num_predict=3`.
The helper must actually exist; passing a client field does not create it.
For complete speculative/cached statistics, use direct HTTP rather than this
SDK's parsed reply. Only keys accepted by the selected public route are valid;
TOML-only settings do not automatically become request fields.

## Preserve extended diagnostics with raw HTTP

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:54800", timeout=300.0) as client:
    response = client.get("/api/ps", params={"detailed": "true"})
    response.raise_for_status()
    print(response.json())
    response = client.post("/api/show", json={"model": "MODEL"})
    response.raise_for_status()
    print(response.json())
    response = client.post("/api/chat", json={
        "model": "MODEL",
        "messages": [{"role": "user", "content": "Explain GGUF briefly."}],
        "stream": False,
    })
    response.raise_for_status()
    reply = response.json()
    print(reply.get("prompt_eval_cached_count"))
    print(reply.get("tllama", {}).get("speculative"))
```

Missing measurements stay absent/None; do not turn them into measured zero.
The [existing streaming examples](client-examples.md) check NDJSON done=true or
SSE completion and close streams on failure/cancellation. Raw HTTP does not
remove the need for those checks or make automatic inference retries safe.

## Management and binary upload

For read-only detailed ps:

```sh
curl --fail-with-body -sS 'http://127.0.0.1:54800/api/ps?detailed=true'
```

To stage a real GGUF as bytes, replace the file path and use its actual digest:

```sh
file=model.gguf
digest=$(sha256sum -- "$file")
digest=${digest%% *}
curl --fail-with-body -sS "http://127.0.0.1:54800/api/blobs/sha256:$digest" \
  -H 'Content-Type: application/octet-stream' --data-binary "@$file"
```

Staging a blob does not register a model. The subsequent TOML definition and
ordered uploads map must match complete GGUF groups, including required
projector/draft files. Prefer `tllama create --file model.toml` for that workflow;
see [model management compatibility](model-management-compatibility.md) and
[CLI import/upload rules](cli.md). Direct create uses TLlama `definition` and
`uploads`, not an Ollama Modelfile or registry create request. The current
sections remain `[llm]`, `[mmproj]` and `[draft]`.

## OpenAI clients and scope

Use Chat Completions for the implemented OpenAI-compatible chat API. The
reviewed Python OpenAI client can forward supported extensions through
`extra_body` and retains unknown response fields, including `reasoning` and
`tllama.speculative`. This does not create support for a server field or route:
Ollama `options.spec_type` is not an OpenAI top-level request option, and public
Responses/audio/media APIs are not present in this release.

Model files/server.toml, local repository maintenance and native backend
installation remain local administration. There is no remote repo-maintenance
route. No own TLlama SDK is introduced here; standard SDKs remain useful within
their compatible scope. For remote daemons replace the base URL, never pass a
client-local filesystem path as an arbitrary daemon path.

## Reviewed client sources

- [ollama-python v0.6.3](https://github.com/ollama/ollama-python/tree/v0.6.3):
  `_client.py` signatures/request assembly and `_types.py` Pydantic models.
- [ollama-js](https://github.com/ollama/ollama-js): published npm 0.6.4
  browser HTTP implementation and interfaces; not every Node filesystem helper.
- [Go API at Ollama 0.40.16](https://github.com/ollama/ollama/tree/d63b702609c4300a2ca07be2bdb0adb9834725e5/api).
