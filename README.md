# TLlama

**TLlama** is a Python-based local AI LLM server designed as a practical drop-in replacement for important parts of the Ollama API, while also exposing OpenAI-compatible endpoints.

It is built on top of `llama-cpp-python` / `llama.cpp` using FastAPI and Pydantic as well and focuses on local GGUF model serving, direct Hugging Face model usage, predictable model repository layout, and Python-friendly extensibility.

> TLlama is currently under active development. Core text generation, model listing, model details, Hugging Face model pulling, and metadata caching are usable. Some advanced features are still experimental or planned.

---

## Why TLlama?

TLlama was created for situations where the Ollama API model is convenient, but you also want more direct control from Python.

It aims to provide:

- Ollama-compatible API behavior.
- Compatibility with the official Ollama CLI and clients.
- OpenAI-compatible `/v1/...` endpoints.
- Direct use of GGUF models from Hugging Face.
- A simple local model repository layout.
- Fast repeated model listing through persistent metadata caching.
- A codebase that is easy to inspect, patch, package, and extend.

---

## Main Features

### Ollama API compatibility

TLlama implements key Ollama-compatible endpoints so existing Ollama clients can talk to TLlama by changing the host.

Example with the official Ollama CLI:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

The default TLlama port is:

```text
54800
```

because hex value 54 represents the ASCII code for T, but this is configurable freely.

Implemented or partially implemented Ollama-style workflows include:

- model listing,
- model details,
- model pulling,
- text generation,
- chat,
- currently loaded model information.

### OpenAI-compatible API

TLlama also exposes OpenAI-compatible endpoints for tools that expect `/v1/...` APIs.

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
    ]
  }' | jq .
```

### Direct Hugging Face model usage

TLlama can work directly with GGUF models from Hugging Face repositories.

Instead of requiring a separate model packaging layer, TLlama can use Hugging Face-style model references and store downloaded GGUF files in a predictable local layout.

Example model name:

```text
namespace/repository/model-file
```

### Multiple model repository areas

TLlama separates models into logical repository areas:

```text
HuggingFace: namespace/repo/model-file
TLlama:      collection/model-file
Local:       model-file
```

This allows TLlama to distinguish between:

- models downloaded from Hugging Face,
- models distributed specifically for TLlama,
- manually copied local GGUF files.

### Persistent GGUF metadata cache

TLlama reads GGUF metadata without initializing the full model through `llama.cpp`.

To avoid slow repeated metadata scans on large GGUF files, TLlama stores a persistent JSON metadata cache.

The cache is invalidated using a lightweight fingerprint:

```text
absolute model path + file size + mtime_ns + cache schema version
```

This makes repeated model listing fast, including after restarting TLlama.

### Direct GGUF metadata reader

Model metadata is read directly from GGUF files instead of using `Llama(..., vocab_only=True)`.

This avoids hangs observed with some model architectures and keeps model discovery independent from full model initialization.

### Reasoning output handling

TLlama includes helper logic for models that emit reasoning blocks, such as `<think>...</think>`.

Reasoning content can be split or preserved depending on endpoint behavior and model format.

---

## Advantages Compared to Ollama

TLlama is not intended to fully replace every Ollama feature immediately, but it provides several useful advantages for some workflows:

- Python-first implementation.
- Easy to inspect, patch, package, and extend.
- Direct Hugging Face GGUF model usage.
- Explicit local model repository layout.
- Persistent metadata cache for fast repeated model listing.
- OpenAI-compatible and Ollama-compatible endpoints in one service.
- Easier integration into Python-based tooling and custom deployment environments.
- Suitable for controlled offline or semi-offline environments where packaging and reproducibility matter.

---

## Current Limitations

### Tool calls

Tool call support is currently limited.

TLlama can accept OpenAI-compatible tool fields and pass them through to `llama-cpp-python`.

However, automatic tool choice is currently limited by upstream `llama-cpp-python` behavior. In testing, raw `llama-server --jinja` correctly returns OpenAI-compatible `message.tool_calls` for `tool_choice: "auto"` and `tool_choice: "required"`, while `llama-cpp-python` may return the model-generated tool call as plain `message.content`.

Explicitly forced tool calls may work depending on the model and backend behavior.

TLlama currently avoids adding fragile heuristic parsing as a workaround for upstream behavior. Proper tool call support is planned.

### Vision and audio

Image and audio support are planned, but not implemented as stable features yet.

Future work may include support for multimodal projector files and model-specific vision/audio pipelines.

### Full Ollama parity

TLlama is designed to be compatible with important Ollama API workflows, but full Ollama parity is not guaranteed yet.

Some advanced Ollama features may be missing, incomplete, or implemented differently.

---

## Installation

Installation depends on how TLlama is packaged in your environment.

For development from source:

```bash
git clone https://github.com/ByCzech/TLlama.git
cd TLlama

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

TLlama depends on `llama-cpp-python`.

For acceleration, build `llama-cpp-python` with the backend flags appropriate for your system. The exact flags may depend on your platform and the current `llama-cpp-python` / `llama.cpp` version, but the common scenarios are:

### CPU-only

```bash
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### CUDA

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### ROCm / HIP

```bash
export CMAKE_ARGS="-DGGML_HIP=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Depending on your ROCm setup, additional environment variables may be required, for example `HIP_VISIBLE_DEVICES` or architecture-specific build options.

### Metal / Apple Silicon

```bash
export CMAKE_ARGS="-DGGML_METAL=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Vulkan

```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

If you maintain your own patched `llama-cpp-python` wheel, install that wheel before installing or running TLlama.

---

## Running TLlama

Start TLlama using your installed entry point or module command.

Example:

```bash
tllama
```

or, depending on your development setup:

```bash
python -m tllama
```

By default, TLlama listens on:

```text
127.0.0.1:54800
```

Then point the Ollama CLI to TLlama:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

## Basic Usage

## Compatible Clients and Libraries

TLlama is designed to work with clients and libraries that speak either the Ollama API or the OpenAI-compatible API.

Known working GUI clients include i.e.:

- **Alpaca** — works through both the Ollama-compatible API and the OpenAI-compatible API.
- **Jan.ai** — works through the OpenAI-compatible API.

TLlama should also work with standard libraries that use Ollama or OpenAI-compatible APIs.

Example with the Python `ollama` package:

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

For OpenAI-compatible clients, point the base URL to:

```text
http://127.0.0.1:54800/v1
```

---

### List models

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Equivalent API call:

```bash
curl -s http://127.0.0.1:54800/api/tags | jq .
```

### Show model information

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show namespace/repo/model-file
```

Equivalent API call:

```bash
curl -s http://127.0.0.1:54800/api/show \
  -H "Content-Type: application/json" \
  -d '{"name":"namespace/repo/model-file"}' | jq .
```

### Generate text

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "namespace/repo/model-file",
    "prompt": "Write a short explanation of what GGUF is.",
    "stream": false
  }' | jq .
```

### Chat

```bash
curl -s http://127.0.0.1:54800/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "namespace/repo/model-file",
    "messages": [
      {
        "role": "user",
        "content": "Explain why local LLM metadata caching is useful."
      }
    ],
    "stream": false
  }' | jq .
```

### OpenAI-compatible chat completion

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

---

## Model Repository Layout

TLlama uses a structured model repository under its configured models directory.

Typical layout:

```text
models/
  HuggingFace/
    namespace/
      repository/
        model-file.gguf

  Local/
    manually-copied-model.gguf

  TLlama/
    collection/
      model-file.gguf

  .tllama/
    metadata-cache/
      <cache-key>.json
```

Public model names are derived from the repository area:

```text
HuggingFace model: namespace/repository/model-file
TLlama model:      collection/model-file
Local model:       model-file
```

---

## Metadata Cache

TLlama creates persistent JSON cache files for model metadata.

The cache is used to make repeated model listing fast, especially for large GGUF models.

Cache invalidation is based on:

```text
absolute model path
file size
file mtime_ns
cache schema version
```

When a model file changes, the cache is regenerated automatically.

This is especially important for large local models or models stored on slower disks or network storage.

---

## Configuration

Configuration can be provided through environment variables.

Common options include:

```text
TLLAMA_HOST
TLLAMA_CONTEXT_LENGTH
TLLAMA_MODELS
```

These are commonly used to configure the listening address, default context length, and model repository location.

Additional configuration options should be documented in `docs/configuration.md`.

---

## Development Notes

Useful development checks:

```bash
python -m compileall src
```

Run Ollama-compatible checks:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

Run OpenAI-compatible checks:

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .
```

---

## Roadmap

Planned or under investigation:

- Better OpenAI-compatible tool call support.
- Ollama API tool call support.
- Investigation of upstream `llama-cpp-python` tool-call parsing.
- Vision model support.
- Audio model support.
- Documentation improvements.
- Debian packaging workflow.
- More structured logging.
- Generated API documentation from docstrings.

---

## Status

TLlama is usable for local text generation workflows, Ollama-compatible model listing/showing, Hugging Face GGUF model management, and OpenAI-compatible chat/generation workflows.

Advanced features such as automatic tool calls, vision, and audio are still under development.

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
