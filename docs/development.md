# Development

This document contains practical notes for TLlama development.

It focuses on local setup, useful checks, compatibility testing, and recommended development workflow.

---

## Development Goals

TLlama is intended to be:

- easy to inspect,
- easy to patch,
- compatible with important Ollama API workflows,
- compatible with OpenAI-style clients where possible,
- suitable for local and controlled deployment environments,
- maintainable without hidden magic.

When adding features, prefer small, testable changes over large speculative rewrites.

---

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install development dependencies:

```bash
pip install -U pip
pip install -e .
```

If your environment uses a custom or patched `llama-cpp-python` build, install it before running TLlama.

Example:

```bash
pip install ./dist/llama_cpp_python-*.whl
pip install -e .
```

---

## Building `llama-cpp-python`

TLlama depends on `llama-cpp-python`, which must be built with the correct backend for your hardware.

Examples:

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

### Metal

```bash
export CMAKE_ARGS="-DGGML_METAL=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Vulkan

```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

---

## Running TLlama Locally

Example development configuration:

```bash
export TLLAMA_HOST=127.0.0.1
export TLLAMA_CONTEXT_LENGTH=4096
export TLLAMA_MODELS=/var/lib/tllama/models

tllama
```

or:

```bash
python -m tllama
```

Default address:

```text
127.0.0.1:54800
```

---

## Basic Smoke Tests

### Compile Python files

```bash
python -m compileall src
```

### Ollama-compatible model listing

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

### Ollama-compatible model details

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

### Ollama-compatible generation

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "prompt": "Say hello from TLlama.",
    "stream": false
  }' | jq .
```

### Ollama-compatible chat

```bash
curl -s http://127.0.0.1:54800/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from TLlama."
      }
    ],
    "stream": false
  }' | jq .
```

### OpenAI-compatible models

```bash
curl -s http://127.0.0.1:54800/v1/models | jq .
```

### OpenAI-compatible chat completion

```bash
curl -s http://127.0.0.1:54800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from TLlama."
      }
    ]
  }' | jq .
```

---

## Testing Metadata Cache

Remove persistent metadata cache:

```bash
rm -rf /var/lib/tllama/models/.tllama/metadata-cache
```

First listing should recreate cache:

```bash
time OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Second listing should be fast:

```bash
time OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Restart TLlama and run again:

```bash
time OLLAMA_HOST=127.0.0.1:54800 ollama list
```

If persistent cache works, listing after restart should remain fast.

Inspect cache files:

```bash
find /var/lib/tllama/models/.tllama/metadata-cache -type f -name '*.json' | head
python -m json.tool /var/lib/tllama/models/.tllama/metadata-cache/<cache-file>.json | less
```

---

## Testing Manual Local Models

Copy a GGUF file into the Local repository:

```bash
cp ./model.gguf /var/lib/tllama/models/Local/my-model.gguf
```

List models:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Expected public model name:

```text
my-model
```

Show model details:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show my-model
```

Update file modification time to simulate a model update:

```bash
touch /var/lib/tllama/models/Local/my-model.gguf
```

Then list or show again. TLlama should invalidate and refresh metadata cache.

---

## Testing Hugging Face Pull Workflow

After pulling a model through TLlama, verify that metadata cache was created immediately:

```bash
find /var/lib/tllama/models/.tllama/metadata-cache -type f -name '*.json' | wc -l
```

Then list models:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

A newly pulled model should not require a slow metadata scan during the first later listing.

---

## Tool Call Investigation

Tool calls are currently limited by upstream `llama-cpp-python` behavior.

Recommended comparison setup:

```text
1. raw llama.cpp server
2. llama-cpp-python OpenAI-compatible server
3. TLlama
```

### Raw llama.cpp server

```bash
./llama-server \
  --model /path/to/model.gguf \
  --alias test-model \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --jinja
```

### llama-cpp-python server

```bash
python -m llama_cpp.server \
  --model /path/to/model.gguf \
  --model_alias test-model \
  --host 127.0.0.1 \
  --port 8001 \
  --n_ctx 4096
```

### TLlama

```bash
tllama
```

Then compare `/v1/chat/completions` behavior for:

```text
tool_choice: forced function
tool_choice: auto
tool_choice: required
```

Observed upstream behavior during development:

```text
raw llama-server --jinja:
  forced   -> message.tool_calls
  auto     -> message.tool_calls
  required -> message.tool_calls

llama-cpp-python server:
  forced   -> message.tool_calls
  auto     -> generated tool call returned as message.content
  required -> generated tool call returned as message.content
```

This is why TLlama currently documents automatic tool calls as a known limitation.

---

## Code Style Guidelines

Prefer:

- small focused patches,
- explicit contracts between helpers and routers,
- deterministic tests before large refactors,
- clear error handling,
- structured logging instead of `print`,
- docstrings for public or architectural helpers.

Avoid:

- hidden magic,
- broad catch-all compatibility hacks,
- speculative large rewrites,
- parsing generated model content with unsafe heuristics unless absolutely necessary.

---

## Logging

TLlama should use the standard Python `logging` module.

Recommended pattern:

```python
import logging

logger = logging.getLogger(__name__)
```

Then use:

```python
logger.debug("...")
logger.info("...")
logger.warning("...")
logger.exception("...")
```

Debug `print(...)` calls should be replaced over time.

---

## Recommended Commit Style

Use small commits with clear messages.

Examples:

```text
Add persistent JSON cache for model metadata
Normalize and limit GGUF metadata extraction
Use prefixless model names for Local and TLlama repositories
Accept OpenAI tool call fields in chat completion requests
```

A good commit should be easy to test and easy to revert.

---

## Useful Files

Common areas to inspect:

```text
src/tllama/backend.py
src/tllama/config.py
src/tllama/routers/ollama.py
src/tllama/routers/openai.py
src/tllama/helpers/gguf_metadata.py
src/tllama/helpers/metadata_cache.py
src/tllama/helpers/prompt_render.py
src/tllama/helpers/reasoning_split.py
```

---

## Related Files

- `README.md`
- `docs/configuration.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
