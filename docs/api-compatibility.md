# API Compatibility

TLlama exposes two API families:

- Ollama-compatible API endpoints,
- OpenAI-compatible API endpoints.

The goal is to make TLlama usable with existing clients and libraries while keeping the implementation Python-first and easy to extend.

TLlama is not yet a complete implementation of every Ollama or OpenAI API feature. This document describes the intended compatibility surface and known limitations.

---

## Default Server Address

By default, TLlama listens on:

```text
127.0.0.1:54800
```

Ollama-compatible base URL:

```text
http://127.0.0.1:54800
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:54800/v1
```

---

## Ollama-Compatible API

TLlama implements important Ollama-style endpoints used by the official Ollama CLI and compatible clients.

Common workflows include:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

---

## Common Ollama Endpoints

### `/api/tags`

Lists available models.

Example:

```bash
curl -s http://127.0.0.1:54800/api/tags | jq .
```

CLI equivalent:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

### `/api/show`

Returns model details and template information.

Example:

```bash
curl -s http://127.0.0.1:54800/api/show \
  -H "Content-Type: application/json" \
  -d '{"name":"namespace/repo/model-file"}' | jq .
```

CLI equivalent:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show namespace/repo/model-file
```

---

### `/api/generate`

Generates text from a prompt.

Example:

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "namespace/repo/model-file",
    "prompt": "Explain what GGUF is.",
    "stream": false
  }' | jq .
```

Streaming responses are supported where implemented.

---

### `/api/chat`

Generates a chat response from a message list.

Example:

```bash
curl -s http://127.0.0.1:54800/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "namespace/repo/model-file",
    "messages": [
      {
        "role": "user",
        "content": "Explain why metadata caching is useful."
      }
    ],
    "stream": false
  }' | jq .
```

---

### `/api/ps`

Returns currently loaded models.

Example:

```bash
curl -s http://127.0.0.1:54800/api/ps | jq .
```

---

### `/api/pull`

Downloads a model file into the TLlama model repository.

The exact supported pull format depends on the current implementation, but the intended workflow is to pull GGUF files from Hugging Face into the `HuggingFace` repository area.

After a successful pull, TLlama creates persistent metadata cache for the downloaded model.

---

## Ollama CLI Compatibility

The official Ollama CLI can be pointed to TLlama:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Examples:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show namespace/repo/model-file
OLLAMA_HOST=127.0.0.1:54800 ollama run namespace/repo/model-file
```

Compatibility depends on the specific Ollama command. Core listing, showing, chat, and generation workflows are the primary target.

---

## OpenAI-Compatible API

TLlama also exposes OpenAI-compatible endpoints under `/v1`.

Base URL:

```text
http://127.0.0.1:54800/v1
```

This is useful for GUI clients and libraries that support local OpenAI-compatible servers.

---

## Common OpenAI-Compatible Endpoints

### `/v1/models`

Lists available models.

Example:

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .
```

---

### `/v1/chat/completions`

Creates a chat completion.

Example:

```bash
curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "namespace/repo/model-file",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from TLlama."
      }
    ],
    "temperature": 0.7
  }' | jq .
```

Streaming may be supported depending on the current route behavior.

---

## Compatible Clients

Known compatible GUI clients include:

### Alpaca

Alpaca can work with TLlama through:

- Ollama-compatible API,
- OpenAI-compatible API.

Use:

```text
http://127.0.0.1:54800
```

for Ollama mode, or:

```text
http://127.0.0.1:54800/v1
```

for OpenAI-compatible mode.

### Jan.ai

Jan.ai can work with TLlama through the OpenAI-compatible API.

Use base URL:

```text
http://127.0.0.1:54800/v1
```

---

## Compatible Libraries

TLlama should work with common libraries that support Ollama or OpenAI-compatible local servers.

### Python `ollama` package

```python
import ollama

client = ollama.Client(host="http://127.0.0.1:54800")

response = client.chat(
    model="namespace/repo/model-file",
    messages=[
        {
            "role": "user",
            "content": "Say hello from TLlama.",
        }
    ],
)

print(response["message"]["content"])
```

### OpenAI-compatible Python clients

For OpenAI-compatible clients, configure the base URL as:

```text
http://127.0.0.1:54800/v1
```

Exact setup depends on the client library.

---

## Tool Calls

Tool call support is currently limited.

TLlama can accept OpenAI-compatible tool fields and pass them to `llama-cpp-python`.

However, full automatic tool call support depends on upstream `llama-cpp-python` behavior.

Observed behavior during testing:

```text
raw llama-server --jinja:
  tool_choice forced   -> message.tool_calls
  tool_choice auto     -> message.tool_calls
  tool_choice required -> message.tool_calls

llama-cpp-python server:
  tool_choice forced   -> message.tool_calls
  tool_choice auto     -> generated tool call returned as message.content
  tool_choice required -> generated tool call returned as message.content
```

Because TLlama uses `llama-cpp-python` as the backend, this limitation currently affects TLlama too.

TLlama intentionally avoids adding fragile heuristic parsing for model-generated tool calls as a workaround.

Planned work:

- investigate upstream `llama-cpp-python` parser behavior,
- support tool calls in OpenAI-compatible endpoints,
- support tool calls in Ollama-compatible endpoints.

---

## Vision and Audio

Vision and audio support are planned but not currently stable.

Future work may include:

- multimodal projector support,
- image input handling,
- audio input/output support,
- model-specific multimodal routing.

---

## Compatibility Goals

TLlama aims for practical compatibility, not blind byte-for-byte compatibility.

Priorities:

```text
1. Existing Ollama clients should work for common local model workflows.
2. OpenAI-compatible clients should work for chat/generation.
3. Model discovery should be fast and reliable.
4. Behavior should be inspectable and configurable from Python.
5. Known limitations should be documented rather than hidden.
```

---

## Testing Compatibility

### Ollama-compatible smoke test

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

### OpenAI-compatible smoke test

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .

curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [
      {
        "role": "user",
        "content": "Say hello."
      }
    ]
  }' | jq .
```

---

## Related Files

- `README.md`
- `docs/configuration.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/development.md`
