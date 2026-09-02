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
| `TLLAMA_RUNTIME_*` | unset | varies | Any `Llama()` load parameter, server-wide |
| `TLLAMA_FLASH_ATTENTION` | unset | bool | Alias for `TLLAMA_RUNTIME_FLASH_ATTN` |
| `TLLAMA_KV_CACHE_TYPE` | unset | string | KV cache type override, both sides |
| `TLLAMA_K_CACHE_TYPE` | unset | string | K cache type override |
| `TLLAMA_V_CACHE_TYPE` | unset | string | V cache type override |
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

An IPv6 address works either bracketed or bare; a bare one cannot carry a port, so it takes the default:

```bash
export TLLAMA_HOST=[::]:54800
export TLLAMA_HOST=[::1]:54800
export TLLAMA_HOST=::1
```

A `http://` or `https://` prefix is accepted and ignored, since that is a documented way to write `OLLAMA_HOST`. The scheme says nothing about how the server listens — TLS belongs to a reverse proxy in front of it, and `https://` here does not enable any. Any other scheme is refused rather than ignored:

```bash
export TLLAMA_HOST=http://0.0.0.0:54800
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

### `TLLAMA_RUNTIME_*`

Sets any load parameter of `llama-cpp-python`'s `Llama()` for every model, as a server-wide default.

The variable name is the parameter name with a `TLLAMA_RUNTIME_` prefix:

```bash
export TLLAMA_RUNTIME_N_THREADS=8
export TLLAMA_RUNTIME_USE_MLOCK=true
export TLLAMA_RUNTIME_MAIN_GPU=1
export TLLAMA_RUNTIME_TENSOR_SPLIT=0.6,0.4
```

Which names exist, and what each one takes, is read from `Llama()`'s signature in the installed build. There is no list here that could disagree with it, and a parameter a future `llama-cpp-python` adds is settable without a TLlama change. A name that is not a parameter, or a value of the wrong shape, stops the server at startup and names the variable.

Values are converted according to the parameter's declared type: whole numbers for `int`, `1/true/yes/on` and `0/false/no/off` for `bool`, comma-separated numbers for `tensor_split`, and either a word or a number for `numa`. Parameters taking an object — `chat_handler`, `draft_model`, `tokenizer` — cannot be set this way, since nothing written in a unit file can become one.

A model's `.toml` `[runtime]` table overrides these for that model.

Four parameters are set elsewhere and are refused here, with a message saying where:

| Parameter | Set by |
|---|---|
| `model_path` | `[llm]` in a model's `.toml` |
| `n_ctx` | `TLLAMA_CONTEXT_LENGTH` |
| `type_k` | `TLLAMA_K_CACHE_TYPE` or `TLLAMA_KV_CACHE_TYPE` |
| `type_v` | `TLLAMA_V_CACHE_TYPE` or `TLLAMA_KV_CACHE_TYPE` |

---

### `TLLAMA_FLASH_ATTENTION`

Enables flash attention for every model.

Default: unset, so `llama.cpp`'s own default applies.

This is an alias for `TLLAMA_RUNTIME_FLASH_ATTN`, kept because it is the spelling Ollama uses and the one TLlama had before the generic mechanism existed. Setting `TLLAMA_RUNTIME_FLASH_ATTN` as well overrides it, the specific spelling beating the general one.

Whether it improves performance depends on the backend, the model, the quantization, the context length and current `llama.cpp` behaviour.

Worth knowing: without flash attention, `llama.cpp` ignores a quantized KV cache, so `TLLAMA_KV_CACHE_TYPE` and its per-side variants have no effect.

---

### `TLLAMA_KV_CACHE_TYPE`

Overrides KV cache type.

Default: unset.

The value is a ggml type name, resolved against the `GGML_TYPE_*` constants the installed `llama-cpp-python` defines. It is not checked against a fixed list, so any type that build knows is accepted, and types a future `llama.cpp` adds work without a TLlama change. This is the same resolution `[runtime]` `type_k` / `type_v` / `type_kv` use in a model's `.toml`, so a name means the same thing in both places.

The types in common use:

```bash
export TLLAMA_KV_CACHE_TYPE=f16
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_KV_CACHE_TYPE=q4_0
```

Lower-precision KV cache types can reduce memory usage, but may affect quality or compatibility.

A name matching no `GGML_TYPE_*` constant is rejected at startup: the server refuses to come up and names the variable, rather than accepting the value and failing on the first request that needs a model. Note that resolving is not the same as being usable: `llama.cpp` supports only some ggml types as a KV cache, and one it does not support is refused when a model loads, which startup cannot check.

Quantized KV cache also requires flash attention (see `TLLAMA_FLASH_ATTENTION`); without it the setting is ignored silently by `llama.cpp`.

---

### `TLLAMA_K_CACHE_TYPE`, `TLLAMA_V_CACHE_TYPE`

Override the cache type for one side only.

Default: unset, meaning whatever `TLLAMA_KV_CACHE_TYPE` says.

`TLLAMA_KV_CACHE_TYPE` is the shorthand for setting both; either of these takes precedence over it for its own side. That is the same relationship `type_kv`, `type_k` and `type_v` have in a model's `.toml` `[runtime]`, and it is resolved by the same code, so a value means the same thing in both places.

```bash
# a smaller V cache than K, a common trade-off
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_V_CACHE_TYPE=q4_0
```

Ollama has no equivalent: `OLLAMA_KV_CACHE_TYPE` sets both sides together and there is no per-side variable.


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
