# Configuration

TLlama is configured primarily through environment variables.

This keeps local development simple and also makes TLlama easy to run from systemd, containers, shell scripts, or Debian packages.

---

## Common Options

### `TLLAMA_HOST`

Listening host/address used by the TLlama server.

Typical value:

```bash
export TLLAMA_HOST=127.0.0.1
```

Use `0.0.0.0` only when you intentionally want TLlama to listen on all network interfaces:

```bash
export TLLAMA_HOST=0.0.0.0
```

> TLlama is intended for local or controlled environments. Do not expose it directly to untrusted networks without an authentication and reverse-proxy layer.

---

### `TLLAMA_CONTEXT_LENGTH`

Default context length used when loading models.

Example:

```bash
export TLLAMA_CONTEXT_LENGTH=4096
```

Larger context lengths require more memory, especially when using GPU offload or large models.

Examples:

```bash
# Small local tests
export TLLAMA_CONTEXT_LENGTH=2048

# Common interactive setup
export TLLAMA_CONTEXT_LENGTH=4096

# Larger context, if memory allows
export TLLAMA_CONTEXT_LENGTH=8192
```
Without settings it or if 0 is set, then default from model is used.

The effective usable context length still depends on:

- model architecture,
- model metadata,
- available RAM / VRAM,
- backend support,
- runtime options passed to `llama-cpp-python`.

---

### `TLLAMA_MODELS`

Path to the TLlama model repository directory.

Example:

```bash
export TLLAMA_MODELS=/var/lib/tllama/models
```

TLlama expects a structured repository under this directory:

```text
/var/lib/tllama/models/
  HuggingFace/
  Local/
  TLlama/
  .tllama/
    metadata-cache/
```

---

## Metadata and Model Discovery

TLlama reads metadata directly from GGUF files and stores a persistent JSON cache.

The cache is used to make repeated model listing fast, especially for large models or models stored on slower disks.

### `TLLAMA_MODEL_SCAN_TIMEOUT`

Timeout for metadata scans in code paths that explicitly use a timeout.

Example:

```bash
export TLLAMA_MODEL_SCAN_TIMEOUT=300
```

In the current TLlama metadata-cache workflow, first-time persistent cache creation may intentionally avoid a short timeout, because large GGUF files can take a long time to scan on slow storage.

---

### `TLLAMA_METADATA_CACHE_TTL`

TTL for the in-memory metadata cache.

Example:

```bash
export TLLAMA_METADATA_CACHE_TTL=18000
```

The persistent JSON cache survives restarts. The in-memory cache only speeds up repeated metadata access while TLlama is running.

---

## Example Development Configuration

```bash
export TLLAMA_HOST=127.0.0.1
export TLLAMA_CONTEXT_LENGTH=4096
export TLLAMA_MODELS=/var/lib/tllama/models

tllama
```

Then test the Ollama-compatible API:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

## Example Systemd Environment

Example environment file:

```bash
# /etc/default/tllama
TLLAMA_HOST=127.0.0.1
TLLAMA_CONTEXT_LENGTH=4096
TLLAMA_MODELS=/var/lib/tllama/models
```

Example service snippet:

```ini
[Service]
EnvironmentFile=/etc/default/tllama
ExecStart=/usr/bin/tllama
```

---

## Example Network Setup

Local-only setup:

```bash
export TLLAMA_HOST=127.0.0.1
```

LAN-accessible setup:

```bash
export TLLAMA_HOST=0.0.0.0
```

When listening on a LAN interface, consider placing TLlama behind a reverse proxy with authentication, TLS, and access controls.

---

## Model Repository Notes

The model directory is not just a flat folder. TLlama uses several repository areas:

```text
HuggingFace: namespace/repo/model-file
TLlama:      collection/model-file
Local:       model-file
```

See `docs/model-repositories.md` for details.

---

## Related Files

- `README.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
