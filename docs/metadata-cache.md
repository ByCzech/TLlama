# Metadata Cache

TLlama uses a persistent JSON metadata cache for GGUF model metadata.

The cache exists because reading metadata from large GGUF model files can be slow, especially on large models, network storage, or slower disks.

The goal is simple:

```text
First metadata scan: may be slow
Next listing/show calls: fast
Next listing/show after restart: still fast
```

---

## Why Metadata Cache Exists

TLlama needs model metadata for several operations:

- `/api/tags`
- `/api/show`
- `/api/ps`
- prompt template handling
- reasoning format detection
- Ollama-compatible model details
- OpenAI-compatible model handling

Earlier metadata discovery could use:

```python
Llama(model_path=model_path, vocab_only=True, verbose=False)
```

That approach is fast for some models, but it can hang or fail for some architectures.

TLlama now reads metadata directly from GGUF files and stores the normalized result in a persistent JSON cache.

---

## Cache Location

The metadata cache is stored under the model repository root:

```text
<TLLAMA_MODELS>/.tllama/metadata-cache/
```

Example:

```text
/var/lib/tllama/models/.tllama/metadata-cache/
```

Each model gets one JSON cache file.

The cache file name is derived from the absolute model path.

Example:

```text
/var/lib/tllama/models/.tllama/metadata-cache/03db1a2600145898458d5ccbf09bbd3bdbe9a995a2a530a6738b400f59f77e0d.json
```

The hash-like file name is not intended to be human-readable. The JSON content contains the actual model path and metadata.

---

## What Is Cached

The cache stores normalized metadata used by TLlama.

Example structure:

```json
{
  "schema_version": 1,
  "created_at": "2026-04-26T02:20:27.679666+00:00",
  "model": {
    "name": "unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M",
    "path": "/var/lib/tllama/models/HuggingFace/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf",
    "size": 2740937888,
    "mtime_ns": 1776855409859723725
  },
  "metadata": {
    "arch": "qwen35",
    "params": 0,
    "parameter_size": "4B",
    "size_label": "4B",
    "bits": "Q4_K_M",
    "template": "...",
    "context_length": 262144,
    "display_name": "Qwen3.5-4B",
    "metadata_raw": {
      "general.architecture": "qwen35",
      "general.file_type": 15,
      "general.size_label": "4B"
    }
  }
}
```

The exact fields may change as the cache schema evolves.

---

## Cache Validation

TLlama validates cache entries using a lightweight file fingerprint:

```text
schema_version
absolute model path
file size
file mtime_ns
```

If any of these values no longer match, the cache entry is ignored and regenerated.

This means the cache is automatically invalidated when:

- the model file is replaced,
- the model file is updated,
- the model file size changes,
- the model modification time changes,
- the cache schema version changes.

TLlama does not hash the full model file by default, because hashing large GGUF files would require reading many gigabytes of data and would defeat the purpose of the cache.

---

## When Cache Is Created

Metadata cache can be created in several situations.

### During model listing

When `ollama list` or `/api/tags` needs metadata for a model and no valid cache exists, TLlama creates it lazily.

Example:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

The first listing after adding large models may be slower. Later listings should be fast.

### During model details

When `ollama show` or `/api/show` is called for a model without a valid cache, TLlama creates the cache.

Example:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama show <model-name>
```

### After model pull

When a model is downloaded through TLlama's pull workflow, TLlama creates metadata cache for the downloaded file.

This avoids paying the metadata scan cost during the first later listing.

### After manual Local model copy

If a GGUF file is copied manually into the `Local` repository area, TLlama creates metadata cache when the model is first listed or shown.

---

## In-Memory Cache vs Persistent Cache

TLlama uses two metadata cache layers:

```text
1. In-memory cache
2. Persistent JSON cache
```

### In-memory cache

The in-memory cache is fastest, but it only lives while TLlama is running.

It is controlled by:

```bash
export TLLAMA_METADATA_CACHE_TTL=18000
```

### Persistent JSON cache

The persistent JSON cache survives restarts.

It is stored under:

```text
<TLLAMA_MODELS>/.tllama/metadata-cache/
```

After the persistent cache exists, model listing can be fast even after restarting TLlama.

---

## Timeout Behavior

First-time metadata scanning of large GGUF files can be slow.

The scan time depends on:

- model file size,
- disk speed,
- network storage speed,
- operating system cache,
- GGUF reader behavior.

On fast NVMe storage, a large model may take seconds to scan. On network storage, a large model can take much longer.

For this reason, TLlama's persistent-cache creation path may avoid short default scan timeouts. Otherwise, large models might never get their metadata cache created.

Some code paths may still support an explicit scan timeout through:

```bash
export TLLAMA_MODEL_SCAN_TIMEOUT=300
```

---

## Removing the Cache

It is safe to remove the metadata cache.

```bash
rm -rf /var/lib/tllama/models/.tllama/metadata-cache
```

TLlama will recreate cache files on demand.

This can be useful after:

- changing metadata parsing code,
- changing cache schema,
- debugging model discovery,
- moving models manually.

---

## Inspecting Cache Files

Cache files are plain JSON.

Example:

```bash
python -m json.tool /var/lib/tllama/models/.tllama/metadata-cache/<cache-file>.json | less
```

Useful fields to inspect:

```text
schema_version
model.name
model.path
model.size
model.mtime_ns
metadata.arch
metadata.parameter_size
metadata.bits
metadata.context_length
metadata.template
```

---

## Troubleshooting

### First `ollama list` is slow

This is expected if metadata cache does not exist yet.

The first scan may read metadata from each GGUF file. Later calls should be fast.

### `ollama list` is slow after every restart

Check whether persistent cache files are being created:

```bash
find /var/lib/tllama/models/.tllama/metadata-cache -type f -name '*.json' | head
```

If no files are created, check permissions on the model directory.

### Cache does not update after replacing a model

Check the model file size and modification time.

TLlama invalidates cache using `size` and `mtime_ns`. If a file is replaced while preserving both size and modification time exactly, TLlama may not detect the change.

This is unusual in normal workflows.

### Cache contains old fields after a code update

Remove the cache directory:

```bash
rm -rf /var/lib/tllama/models/.tllama/metadata-cache
```

Then restart TLlama or call `ollama list` again.

---

## Design Notes

The metadata cache intentionally uses JSON instead of a binary format.

Reasons:

- easy to inspect manually,
- easy to debug,
- stable enough for small metadata payloads,
- safe compared to pickle,
- simple schema versioning,
- easy atomic writes through temporary files and replace.

The cache is not a substitute for the model file. It only stores small metadata needed by TLlama.

---

## Related Files

- `README.md`
- `docs/configuration.md`
- `docs/model-repositories.md`
- `docs/api-compatibility.md`
