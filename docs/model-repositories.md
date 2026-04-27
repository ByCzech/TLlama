# Model Repositories

TLlama stores models in a structured repository directory.

The goal is to support three common sources of GGUF models while keeping public model names simple and predictable:

- models downloaded from Hugging Face,
- models distributed specifically for TLlama,
- manually copied local models.

---

## Repository Root

The repository root is configured with:

```bash
export TLLAMA_MODELS=/var/lib/tllama/models
```

Typical structure:

```text
/var/lib/tllama/models/
  HuggingFace/
    namespace/
      repository/
        model-file.gguf

  TLlama/
    collection/
      model-file.gguf

  Local/
    manually-copied-model.gguf

  .tllama/
    metadata-cache/
      <cache-key>.json
```

---

## Public Model Name Rules

TLlama exposes model names without internal storage prefixes.

| Repository area | Public model name format | Example |
|---|---|---|
| HuggingFace | `namespace/repository/model-file` | `unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL` |
| TLlama | `collection/model-file` | `official/example-model` |
| Local | `model-file` | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L` |

Internal directory names such as `HuggingFace`, `TLlama`, and `Local` are not normally part of the public model name.

---

## Hugging Face Models

Hugging Face models are stored under:

```text
<TLLAMA_MODELS>/HuggingFace/<namespace>/<repository>/<model-file>.gguf
```

Example on disk:

```text
/var/lib/tllama/models/HuggingFace/unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL.gguf
```

Public model name:

```text
unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Example:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Equivalent API call:

```bash
curl -s http://127.0.0.1:54800/api/show \
  -H "Content-Type: application/json" \
  -d '{"name":"unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL"}' | jq .
```

---

## Pulling Models from Hugging Face

TLlama can download GGUF files from Hugging Face repositories.

The exact request format depends on the current `/api/pull` implementation, but the resulting model is stored in the `HuggingFace` repository area.

After a successful pull, TLlama creates metadata cache for the downloaded model so later `ollama list` and `ollama show` calls can be fast.

Example target layout after pull:

```text
/var/lib/tllama/models/HuggingFace/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q8_0.gguf
```

Public model name:

```text
Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q8_0
```

---

## Local Models

Local models are intended for manually copied GGUF files.

They are stored under:

```text
<TLLAMA_MODELS>/Local/
```

Example on disk:

```text
/var/lib/tllama/models/Local/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf
```

Public model name:

```text
DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L
```

Example:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L
```

This keeps the local repository simple and close to the early flat-directory model layout.

### Adding a Local Model Manually

Copy a GGUF file into the `Local` directory:

```bash
sudo cp ./model.gguf /var/lib/tllama/models/Local/my-local-model.gguf
```

Then list models:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

If metadata cache does not exist yet, TLlama creates it lazily when the model is listed or shown.

---

## TLlama Models

The `TLlama` repository area is reserved for models or model bundles distributed specifically for TLlama.

They are stored under:

```text
<TLLAMA_MODELS>/TLlama/<collection>/<model-file>.gguf
```

Example on disk:

```text
/var/lib/tllama/models/TLlama/official/example-model.gguf
```

Public model name:

```text
official/example-model
```

Example:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show official/example-model
```

The TLlama repository area is intended for future workflows where models may be distributed directly for TLlama instead of downloaded from Hugging Face.

---

## `model.gguf` Directory Form

TLlama supports directory-style model references for repositories that use a fixed file name such as `model.gguf`.

Example on disk:

```text
/var/lib/tllama/models/TLlama/official/example/model.gguf
```

Public model name:

```text
official/example
```

This is useful for model bundles where additional files may live next to the GGUF file in the future.

---

## Metadata Cache

TLlama stores persistent metadata cache under:

```text
<TLLAMA_MODELS>/.tllama/metadata-cache/
```

The cache is keyed by the absolute model path and validated using model file size and modification time.

You normally do not need to manage these files manually.

If needed, the cache can be removed safely:

```bash
rm -rf /var/lib/tllama/models/.tllama/metadata-cache
```

TLlama will recreate metadata cache files on demand.

See `docs/metadata-cache.md` for details.

---

## Name Collision Notes

The public naming scheme is intentionally simple:

```text
Local:       one segment
TLlama:      two segments
HuggingFace: three or more segments
```

Examples:

```text
my-model
official/my-model
unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-IQ4_NL
```

Avoid creating ambiguous names across repository areas.

For example, do not use a TLlama collection/name pair that could be confused with a manually managed Local nested path.

---

## Quick Checks

List all models:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

Show one model:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

Generate with one model:

```bash
curl -s http://127.0.0.1:54800/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "prompt": "Say hello from TLlama.",
    "stream": false
  }' | jq .
```

---

## Related Files

- `README.md`
- `docs/configuration.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
