# Usage with clients and SDKs

Start `tllamad`, then select a registered model from `tllama list`.
The default Ollama-compatible base URL is `http://127.0.0.1:54800`;
OpenAI-compatible clients use `http://127.0.0.1:54800/v1`.
The [route inventory](api-compatibility.md) defines the supported contracts.
For copyable basic and advanced recipes, see [client examples](client-examples.md):
Python `ollama`, `openai`, `requests`, `httpx`/`httpx2`, curl and JavaScript.

## TLlama client

```bash
tllama list
tllama show MODEL
tllama run MODEL
tllama run MODEL 'Explain GGUF in two sentences.'
cat source.py | tllama run MODEL 'Review this code.'
tllama ps --json
tllama --host http://server:54800 list
```

`run` chooses generation, embeddings or Decision from the model's advertised
capabilities. With a terminal and no prompt, it opens the REPL. One-shot/stdin
calls remain one request. The daemon does not automatically download a missing
model. See [cli.md](cli.md) for import/copy, slash commands, reasoning display,
verbose timings, cancellation and exit codes.

## Ollama CLI

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
OLLAMA_HOST=127.0.0.1:54800 ollama show MODEL
OLLAMA_HOST=127.0.0.1:54800 ollama run MODEL 'Explain GGUF.'
```

Use the exact public name from the model list. Core show/chat/generate paths
are supported, not every Ollama command. `show --modelfile` returns original
TLlama TOML rather than an Ollama Modelfile conversion. Use TLlama's client for
its native TOML create/import workflow.

Pull uses a Hugging Face reference, not an Ollama registry tag:

```bash
tllama pull namespace/repository/model-file.gguf
```

The filename can include repository subdirectories; omitting `.gguf` adds it.
Do not paste a `https://huggingface.co/.../blob/main/...` URL as the reference.
The optional `huggingface` dependency is required on the server. Downloaded
weights, metadata/digest cache and definition handling are described in
[model-repositories.md](model-repositories.md).

## HTTP examples

Replace `MODEL` with a registered generation model:

```bash
curl -s http://127.0.0.1:54800/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Say hello."}],"stream":false}'

curl -s http://127.0.0.1:54800/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","prompt":"Explain GGUF.","stream":false}'

curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

Ollama streaming returns NDJSON; OpenAI chat streaming returns SSE. A stream
can fail after partial output; consumers must check its successful terminal
rather than treating EOF as success. See [native-public-api.md](native-public-api.md)
for thinking, structured output, ownership and unsupported controls.

## Python SDKs

Install `ollama` and/or `openai` in the **client's** Python environment. These
SDKs are optional clients, not TLlama inference dependencies. Native diagnostic
scripts that exercise SDK paths need them in the interpreter running the script.

```python
import ollama as tllama

client = tllama.Client(host="http://127.0.0.1:54800", timeout=300.0)
response = client.chat(
    model="MODEL",
    messages=[{"role": "user", "content": "Explain GGUF in one sentence."}],
    stream=False,
)
print(response["message"]["content"])
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:54800/v1", api_key="tllama")
response = client.chat.completions.create(
    model="MODEL",
    messages=[{"role": "user", "content": "Say hello."}],
    stream=False,
)
print(response.choices[0].message.content)
```

The placeholder key satisfies SDK construction; TLlama does not use it as
public API authentication. It is unrelated to private native worker credentials.
`responses.create()` and `completions.create()` are unsupported routes.
SDK transport parsing does not imply every optional hosted-service feature is
implemented. The [runtime audit](runtime-audit-070.md) records actually reviewed
SDK paths and hardware settings.

For a registered embedding model:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:54800/v1", api_key="tllama")
response = client.embeddings.create(model="EMBEDDING_MODEL", input=["literal text"])
print(response.data[0].embedding)
```

Embedding inputs are literal: add any model-required query/document prefix
explicitly. Exact local token arrays, bounds, normalization, dimensions, base64
and image extensions have the [embedding API](embedding-api.md) contract.
A text Decision model instead uses HTTP JSON at `/v1/systemone`; see
[decision-api.md](decision-api.md). No standard SDK Decision method is claimed.

## GUI and IDE clients

For a client supporting a custom compatible endpoint, configure:

| Setting | Ollama mode | OpenAI-compatible mode |
| --- | --- | --- |
| Base URL | `http://127.0.0.1:54800` | `http://127.0.0.1:54800/v1` |
| Model | Exact registered name | Exact registered name |
| API key | As required by the client | Nonempty placeholder if required |

These are generic integration settings, not a certification of a particular
Alpaca, Jan, Continue, Cline or Roo release. Client configuration names vary;
use its custom-provider settings and test one ordinary chat request first.
An agent that requires Responses, unsupported controls or automatic server-side
tool execution needs a different supported client mode.

For tools, the client executes functions and returns results with the required
history/IDs. OpenAI supplies appendable argument strings; Ollama supplies complete
validated objects after native completion. Wait for successful completion before
executing calls. See [native-tools.md](native-tools.md). Inline images need a
configured projector and live native support; remote image URLs are refused.
FIM autocomplete has the separate [suffix contract](native-fim.md) and model
quality qualification. Ordinary coding chat does not prove FIM support.

## Troubleshooting

Connection failures: check the listener and selected address:

```bash
curl -s http://127.0.0.1:54800/health
curl -s http://127.0.0.1:54800/api/tags
```

The TLlama client does not read server TOML or `TLLAMA_PORT`; include a changed
port in `--host`/client `TLLAMA_HOST`. A source/install update may require
regenerating console scripts; `python -m tllama` and `python -m tllama.daemon`
provide checkout entry points with dependencies available.

For missing models, check the exact registered name and valid TOML. For slow
first listings, retain the wait while metadata/digests are prepared and check
cache permissions. A permission-denied pull must be diagnosed under the actual
daemon account and failing path; see [configuration.md](configuration.md).
For tool failures, inspect declarations, histories and native/model output,
without automatically unescaping or repairing arguments.

Use [daemon.md](daemon.md) for configuration inspection, [systemd.md](systemd.md)
for service permissions and [native-memory.md](native-memory.md) for honest
SIZE/PROCESSOR interpretation. Public authentication/TLS belongs to the deployment
controls described in those guides; a placeholder SDK key does not provide it.
