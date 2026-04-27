# Usage

This document shows practical ways to use TLlama with existing clients, tools, libraries, and IDE integrations.

TLlama exposes:

```text
Ollama-compatible API: http://127.0.0.1:54800
OpenAI-compatible API: http://127.0.0.1:54800/v1
```

The default port is:

```text
54800
```

---

## Using TLlama with the Ollama CLI

The official Ollama CLI can be pointed to TLlama with `OLLAMA_HOST`.

Example:

```bash
export OLLAMA_HOST=127.0.0.1:54800
```

Then use normal Ollama commands.

---

### List models

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Example output:

```text
NAME                                                     ID              SIZE      MODIFIED
unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B...       e88c661ec170    4.7 GB    3 days ago
DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L                  486cfb4d3607    10 GB     13 seconds ago
```

---

### Show model details

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Example:

```text
Model
  architecture    llama
  parameters      8B
  quantization    IQ4_NL
```

---

### Show model Modelfile/template

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show --modelfile unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

This can be useful for checking the chat template extracted from GGUF metadata.

---

### Run a model interactively

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama run unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Then type prompts directly in the Ollama CLI.

---

### Run a one-shot prompt

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama run unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL "Explain what GGUF is in two sentences."
```

---

### Pull a model

TLlama supports a pull workflow for Hugging Face GGUF models.

Exact pull syntax depends on the current TLlama `/api/pull` implementation and supported model reference format.

After a successful pull, TLlama stores the model under the `HuggingFace` repository area and creates metadata cache for it.

Then it appears in:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

## Using TLlama with curl

### Ollama-compatible generate

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
    "prompt": "Explain what a local LLM server is.",
    "stream": false
  }' | jq .
```

---

### Ollama-compatible chat

```bash
curl -s http://127.0.0.1:54800/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
    "messages": [
      {
        "role": "user",
        "content": "Give me three benefits of local LLM inference."
      }
    ],
    "stream": false
  }' | jq .
```

---

### OpenAI-compatible chat completion

```bash
curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from TLlama."
      }
    ],
    "temperature": 0.7
  }' | jq .
```

---

## Using TLlama with Python Libraries

### Python `ollama` package

```python
import ollama

client = ollama.Client(host="http://127.0.0.1:54800")

response = client.chat(
    model="unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
    messages=[
        {
            "role": "user",
            "content": "Explain TLlama in one sentence.",
        }
    ],
)

print(response["message"]["content"])
```

---

### OpenAI-compatible Python clients

For OpenAI-compatible clients, use:

```text
base_url = http://127.0.0.1:54800/v1
api_key  = any non-empty value if the client requires one
```

Example using the OpenAI Python client style:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:54800/v1",
    api_key="tllama",
)

response = client.chat.completions.create(
    model="unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
    messages=[
        {
            "role": "user",
            "content": "Say hello from TLlama.",
        }
    ],
)

print(response.choices[0].message.content)
```

---

## GUI Clients

TLlama can be used with GUI clients that support either Ollama-compatible servers or OpenAI-compatible local servers.

---

### Alpaca

Alpaca can work with TLlama in two possible modes:

```text
Ollama-compatible mode:
  Base URL: http://127.0.0.1:54800

OpenAI-compatible mode:
  Base URL: http://127.0.0.1:54800/v1
  API key:  any non-empty value if required
```

Use one of the model names shown by:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

### Jan.ai

Jan.ai can use TLlama through the OpenAI-compatible API.

Typical settings:

```text
Provider:  OpenAI-compatible / custom OpenAI endpoint
Base URL:  http://127.0.0.1:54800/v1
API key:   any non-empty value if required
Model:     one of the TLlama model names
```

Example model:

```text
unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

---

### Other OpenAI-Compatible GUI Clients

Any GUI client that supports a custom OpenAI-compatible base URL should be able to use TLlama for basic chat/generation workflows.

Use:

```text
Base URL: http://127.0.0.1:54800/v1
API key:  any non-empty value if required
Model:    <model-name from TLlama>
```

---

## IDE and Coding Assistants

Many IDE extensions and coding assistants support OpenAI-compatible custom endpoints.

TLlama can be used as a local endpoint for these tools when they allow setting:

```text
Base URL
API key
Model name
```

Use:

```text
Base URL: http://127.0.0.1:54800/v1
API key:  any non-empty value if required
Model:    one of the model names returned by /v1/models or ollama list
```

---

### Generic OpenAI-Compatible IDE Setup

Typical configuration:

```text
Provider: OpenAI-compatible
Base URL: http://127.0.0.1:54800/v1
API key:  tllama
Model:    unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Use a model suitable for code tasks, for example a coding-focused GGUF model stored in the `Local`, `TLlama`, or `HuggingFace` repository area.

---

### Continue-style Configuration Example

For IDE tools that use a JSON/YAML-style model configuration, the shape is usually similar to:

```json
{
  "title": "TLlama",
  "provider": "openai",
  "model": "unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL",
  "apiBase": "http://127.0.0.1:54800/v1",
  "apiKey": "tllama"
}
```

Exact field names depend on the client.

---

### Cline / Roo Code / Similar Tools

For tools that support an OpenAI-compatible provider:

```text
API Provider: OpenAI-compatible
Base URL:     http://127.0.0.1:54800/v1
API Key:      tllama
Model:        <TLlama model name>
```

Tool-call support may be required by some agentic IDE workflows. Because automatic tool calls are currently limited by upstream `llama-cpp-python` behavior, some advanced agent workflows may not work correctly yet.

Basic chat/completion workflows should be the first thing to test.

---

## Model Names

Use model names exactly as TLlama exposes them.

List available models:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

or:

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .
```

Examples:

```text
HuggingFace:
  unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL

TLlama:
  official/example-model

Local:
  DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L
```

---

## Notes About Tool Calls

Some clients and IDE agents rely heavily on tool calls.

Current TLlama behavior:

```text
Basic chat/generation: supported
Forced tool calls: may work depending on backend/model
Automatic tool calls: currently limited by upstream llama-cpp-python behavior
Ollama API tool calls: planned
OpenAI API tool calls: planned
```

For now, prefer simple chat/completion clients or IDE workflows that do not require automatic tool calls.

---

## Troubleshooting

### Client cannot connect

Check that TLlama is running:

```bash
curl -s http://127.0.0.1:54800/api/tags | jq .
```

If using OpenAI-compatible mode:

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .
```

---

### GUI client requires an API key

Use any non-empty placeholder value:

```text
tllama
```

TLlama does not currently require a real OpenAI API key for local usage.

---

### Model not found

Check the exact model name:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Then copy the model name exactly.

---

### IDE agent fails when using tools

This may be caused by current tool-call limitations.

Try a simpler chat/completion mode first.

---

## Related Files

- `README.md`
- `docs/configuration.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
- `docs/development.md`
