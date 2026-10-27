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

Use a repository reference, with optional subdirectories and `.gguf` suffix:

```bash
tllama pull namespace/repository/path/to/model-file.gguf
```

```json
{"model":"namespace/repository/path/to/model-file.gguf","stream":false}
```

The JSON goes to `POST /api/pull`; `name` is also accepted. Omitted `stream`
defaults to NDJSON progress. The server appends `.gguf` to a filename without
that extension. This is not an Ollama registry tag or a Hugging Face browser
`blob/main`/`resolve/main` URL. Downloaded files use the `HuggingFace` area and
successful definition preparation makes them visible as registered models.
The server needs the optional `huggingface` extra. Pull downloads the requested
file; do not infer client import's complete split-shard upload behavior here.

The daemon account needs suitable model-store and Hugging Face cache permissions,
including parent traversal. Service and interactive accounts can differ; inspect
the actual error path before changing ownership/ACLs. See [systemd.md](systemd.md).

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

Local models hold manually copied GGUF files and client-imported weights.
Client imports use `Local/imports/`; their registered TOML remains independent
of the physical weight path. See [cli.md](cli.md#create-and-import-models).

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

Register the missing definition, then list models:

```bash
tllamad rebuildrepo --dryrun
tllamad rebuildrepo
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

A bare GGUF is not listed until a TOML definition points at it. Use the same
server configuration/model store for `rebuildrepo` and the running daemon;
global daemon flags precede the maintenance subcommand. Alternatively,
`tllama create MODEL -f FILE` uploads client-side `from` files and writes the
definition through HTTP; see [CLI create/import](cli.md#create-and-import-models).
If metadata cache does not exist yet, listing/show creates it lazily.

---

## Writing the Missing `.toml` Definitions

A `.gguf` that no `.toml` points at is not a model: it is not listed and cannot be loaded. `tllamad rebuildrepo` writes the definitions that are missing, for a repository that predates them and equally for a file just copied into `Local/` by hand.

```bash
tllamad rebuildrepo --dryrun     # report only, writes nothing
tllamad rebuildrepo              # add the definitions that are absent
```

Without `--force` it only adds; a `.toml` that already exists is left exactly as it is, so the command is safe to run at any time.

### `--force` and the backup

`--force` also replaces definitions that already exist. Whatever a person wrote into one — an uncommented `n_ctx`, a `[system].prompt`, their own comments — is copied to a `.toml.bak` beside it first, without being asked for: anyone reaching for `--force` is thinking about what they want to gain, not about what they are about to lose.

A second `--force` therefore finds a `.toml.bak` already there. It **skips that file** rather than replacing the backup with a copy of the generated one, which would destroy the only thing protecting against the first run. The skip is per file: everything without a backup is still done, and each skipped file is named with the reason.

Two ways forward from a skip. Move the `.toml.bak` somewhere else, which keeps it and lets the file be rebuilt; or say it is expendable:

```bash
tllamad rebuildrepo --force --overwrite-backups
```

That flag does nothing on its own and is refused without `--force`. `--dryrun` reports the skips exactly as the real run performs them.

Note that `rebuildrepo` regenerates a definition whole rather than refreshing it in place, so `--force` is the wrong tool for keeping an edited `.toml` and updating the commented block around it.

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
OLLAMA_HOST=127.0.0.1:54800 ollama show MODEL
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
