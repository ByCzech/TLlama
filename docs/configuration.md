# Configuration

TLlama is configured primarily through environment variables.

This keeps local development simple and also makes TLlama easy to run from systemd, containers, shell scripts, or Debian packages.

---

## Quick Example

Common local development setup:

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models
export TLLAMA_CONTEXT_LENGTH=4096

tllama
```

Then test the Ollama-compatible API:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

## Configuration Overview

| Variable | Default | Type | Purpose |
|---|---:|---|---|
| `TLLAMA_HOST` | `127.0.0.1:54800` | string | Server host and port |
| `TLLAMA_MODELS` | `/var/lib/tllama/models` | path | Model repository root |
| `TLLAMA_CONTEXT_LENGTH` | `0` | int | Default model context length |
| `TLLAMA_KEEP_ALIVE` | `5m` | string/int/float | Loaded model keep-alive duration |
| `TLLAMA_MAX_LOADED_MODELS` | `1` | int | Maximum number of simultaneously loaded models |
| `TLLAMA_JANITOR_INTERVAL` | `10.0` | float | Background model janitor interval in seconds |
| `TLLAMA_MODEL_SCAN_TIMEOUT` | `5.0` | float | Explicit metadata scan timeout where applicable |
| `TLLAMA_METADATA_CACHE_TTL` | `300.0` | float | In-memory metadata cache TTL in seconds |
| `TLLAMA_FLASH_ATTENTION` | `false` | bool | Enable llama.cpp flash attention |
| `TLLAMA_KV_CACHE_TYPE` | unset | string | KV cache type override |
| `TLLAMA_APP_RELOAD` | `false` | bool | Enable application reload mode |
| `TLLAMA_DEBUG` | `false` | bool | Enable debug mode |

Boolean values accept:

```text
1, true, yes, on
0, false, no, off
```

Invalid values fall back to defaults.

---

## Application Server Options

### `TLLAMA_HOST`

Configures the listening host and port.

Default:

```bash
export TLLAMA_HOST=127.0.0.1:54800
```

You can provide host and port:

```bash
export TLLAMA_HOST=127.0.0.1:54800
```

Host only is also accepted; the default port is used:

```bash
export TLLAMA_HOST=127.0.0.1
```

Listen on all interfaces:

```bash
export TLLAMA_HOST=0.0.0.0:54800
```

> Do not expose TLlama directly to untrusted networks without a reverse proxy, authentication, TLS, and access controls.

---

### `TLLAMA_APP_RELOAD`

Enables application reload mode.

Default:

```bash
export TLLAMA_APP_RELOAD=false
```

Typical development use:

```bash
export TLLAMA_APP_RELOAD=true
```

Use this only for development.

---

### `TLLAMA_DEBUG`

Enables debug mode.

Default:

```bash
export TLLAMA_DEBUG=false
```

Example:

```bash
export TLLAMA_DEBUG=true
```

Debug mode is intended for development and diagnostics.

---

## Model Repository Options

### `TLLAMA_MODELS`

Path to the TLlama model repository root.

Default:

```bash
export TLLAMA_MODELS=/var/lib/tllama/models
```

Expected structure:

```text
/var/lib/tllama/models/
  HuggingFace/
  Local/
  TLlama/
  .tllama/
    metadata-cache/
```

Repository areas:

```text
HuggingFace: namespace/repo/model-file
TLlama:      collection/model-file
Local:       model-file
```

See `docs/model-repositories.md` for details.

---

## Model Loading Options

### `TLLAMA_CONTEXT_LENGTH`

Default context length passed when loading models.

Default:

```bash
export TLLAMA_CONTEXT_LENGTH=0
```

A value of `0` means TLlama lets the backend/model metadata decide where applicable.

Common examples:

```bash
# Small tests
export TLLAMA_CONTEXT_LENGTH=2048

# Common chat setup
export TLLAMA_CONTEXT_LENGTH=4096

# Larger context if memory allows
export TLLAMA_CONTEXT_LENGTH=8192
```

The effective context length depends on:

- model architecture,
- model metadata,
- available RAM / VRAM,
- selected backend,
- KV cache type,
- GPU offload behavior.

---

### `TLLAMA_KEEP_ALIVE`

Default keep-alive duration for loaded models.

Default:

```bash
export TLLAMA_KEEP_ALIVE=5m
```

Examples:

```bash
# Unload quickly
export TLLAMA_KEEP_ALIVE=30s

# Keep model loaded for 10 minutes
export TLLAMA_KEEP_ALIVE=10m

# Keep model loaded for 1 hour
export TLLAMA_KEEP_ALIVE=1h
```

Depending on endpoint behavior, keep-alive may also be overridden by request payloads.

---

### `TLLAMA_MAX_LOADED_MODELS`

Maximum number of simultaneously loaded models.

Default:

```bash
export TLLAMA_MAX_LOADED_MODELS=1
```

With the default value, loading a new model unloads the previous one.

Allow two loaded models:

```bash
export TLLAMA_MAX_LOADED_MODELS=2
```

If the limit is reached and TLlama cannot unload automatically, loading another model may fail with a message suggesting increasing `TLLAMA_MAX_LOADED_MODELS`.

Memory usage grows quickly when several models are loaded at the same time.

---

### `TLLAMA_JANITOR_INTERVAL`

Interval in seconds for the background loaded-model janitor.

Default:

```bash
export TLLAMA_JANITOR_INTERVAL=10.0
```

The janitor checks loaded models and unloads expired ones based on keep-alive behavior.

Example:

```bash
export TLLAMA_JANITOR_INTERVAL=5.0
```

---

## llama.cpp Runtime Options

### `TLLAMA_FLASH_ATTENTION`

Enables flash attention when supported by the current `llama-cpp-python` / `llama.cpp` build.

Default:

```bash
export TLLAMA_FLASH_ATTENTION=false
```

Enable:

```bash
export TLLAMA_FLASH_ATTENTION=true
```

This maps to the `flash_attn` load option.

Whether it improves performance depends on:

- backend,
- model,
- quantization,
- context length,
- GPU support,
- current `llama.cpp` behavior.

---

### `TLLAMA_KV_CACHE_TYPE`

Overrides KV cache type.

Default: unset.

Supported values:

```text
f16
q8_0
q4_0
```

Examples:

```bash
export TLLAMA_KV_CACHE_TYPE=f16
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_KV_CACHE_TYPE=q4_0
```

Lower-precision KV cache types can reduce memory usage, but may affect quality or compatibility.

The selected value must be supported by the installed `llama-cpp-python` build. If the required constant is unavailable, TLlama raises an error.

---

## Metadata and Model Discovery

TLlama reads GGUF metadata directly and stores persistent JSON metadata cache.

### `TLLAMA_MODEL_SCAN_TIMEOUT`

Timeout for metadata scan code paths that explicitly use a timeout.

Default:

```bash
export TLLAMA_MODEL_SCAN_TIMEOUT=5.0
```

Example:

```bash
export TLLAMA_MODEL_SCAN_TIMEOUT=300
```

Important: first-time persistent metadata cache creation may intentionally avoid a short timeout. Large GGUF files can take much longer than 5 seconds to scan, especially on network storage.

---

### `TLLAMA_METADATA_CACHE_TTL`

TTL for the in-memory metadata cache, in seconds.

Default:

```bash
export TLLAMA_METADATA_CACHE_TTL=300.0
```

Example:

```bash
export TLLAMA_METADATA_CACHE_TTL=18000
```

There are two metadata cache layers:

```text
1. In-memory cache
2. Persistent JSON cache
```

The in-memory cache is fastest, but it only lives while TLlama is running.

The persistent JSON cache survives restarts and is stored under:

```text
<TLLAMA_MODELS>/.tllama/metadata-cache/
```

See `docs/metadata-cache.md` for details.

---

## Example Configurations

### Minimal local setup

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models

tllama
```

### Larger context setup

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models
export TLLAMA_CONTEXT_LENGTH=8192

tllama
```

### Lower memory setup

```bash
export TLLAMA_CONTEXT_LENGTH=4096
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_MAX_LOADED_MODELS=1

tllama
```

### Flash attention setup

```bash
export TLLAMA_FLASH_ATTENTION=true
export TLLAMA_CONTEXT_LENGTH=8192

tllama
```

### LAN-accessible setup

```bash
export TLLAMA_HOST=0.0.0.0:54800
export TLLAMA_MODELS=/var/lib/tllama/models

tllama
```

Use a reverse proxy and access controls when exposing TLlama beyond localhost.

---

## Example systemd Environment

Example environment file:

```bash
# /etc/default/tllama
TLLAMA_HOST=127.0.0.1:54800
TLLAMA_MODELS=/var/lib/tllama/models
TLLAMA_CONTEXT_LENGTH=4096
TLLAMA_KEEP_ALIVE=5m
TLLAMA_MAX_LOADED_MODELS=1
TLLAMA_METADATA_CACHE_TTL=18000
```

Example service snippet:

```ini
[Service]
EnvironmentFile=/etc/default/tllama
ExecStart=/usr/bin/tllama
```

---

## Troubleshooting

### TLlama listens on the wrong address

Check:

```bash
echo "$TLLAMA_HOST"
```

Expected format:

```text
host:port
```

Example:

```bash
export TLLAMA_HOST=127.0.0.1:54800
```

---

### Model repository is empty

Check:

```bash
echo "$TLLAMA_MODELS"
find "$TLLAMA_MODELS" -maxdepth 3 -type f -name '*.gguf' | head
```

---

### Model listing is slow

The first listing can be slow if persistent metadata cache does not exist yet.

Check cache files:

```bash
find "$TLLAMA_MODELS/.tllama/metadata-cache" -type f -name '*.json' | head
```

---

### KV cache type fails

Check that the selected value is supported:

```bash
echo "$TLLAMA_KV_CACHE_TYPE"
```

Supported values:

```text
f16
q8_0
q4_0
```

Also make sure the current `llama-cpp-python` build exposes the required GGML type constants.

---

## Related Files

- `README.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
- `docs/usage.md`
- `docs/development.md`
