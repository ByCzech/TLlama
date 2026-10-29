# Daemon configuration

`tllamad --list-devices` forwards the configured native runtime's device table
and exits without starting the server or loading a model. `--config`,
`--no-config` and `--runtime-path` retain their normal precedence. See the
[device guide](device-inventory.md#one-shot-daemon-command-12) for scope and limits.

`tllamad` runs the server in the foreground. `tllama` is the HTTP client.
The source equivalents are `python -m tllama.daemon` and `python -m tllama`.

```sh
tllamad --config server.toml --check-config
tllamad --config server.toml --print-config
tllamad --config server.toml --device Vulkan0 --port 54801
tllamad --no-config --models-dir ./models --log-level debug
tllamad --config server.toml rebuildrepo --dryrun
```

Daemon options precede the maintenance subcommand. `--help` and `--version`
do not read server configuration or import the application. Inspection exits
without importing FastAPI, creating storage, discovering GPUs, or loading a
model. `--print-config` emits JSON with each effective value and its source;
it includes schema-owned settings only, never Hugging Face tokens or the full
process environment. Invalid configuration exits 2 with a diagnostic on stderr.
`--check-config` checks the configuration contract; runtime availability and
model/device compatibility are checked when actually used.

## Sources and file selection

For global settings: **CLI > environment > server TOML > defaults**.
Source syntax, keys and types are checked while reading; effective values are
validated before serving. Empty environment strings mean unset, except
`TLLAMA_BACKEND_PRIORITY=""`, which explicitly selects CPU-only policy.

Choose one server file, without merging several files:

1. `--config PATH`.
2. `TLLAMA_CONFIG`.
3. `$XDG_CONFIG_HOME/tllama/server.toml` (default `~/.config/tllama/server.toml`;
   on Windows, `%APPDATA%/tllama/server.toml` when available).
4. `/etc/tllama/server.toml`.

The first existing default file wins. An explicitly selected missing,
unreadable or malformed file fails; it never falls back to another file.
`--no-config` bypasses both file selection and discovery, including
`TLLAMA_CONFIG`. No file is required when none of the default paths exists.

Client address selection stays independent: when changing the listener port,
use `tllama --host HOST:PORT` or include that port in the client's `TLLAMA_HOST`.
The client does not read server TOML or `TLLAMA_PORT`.

Relative configuration and storage paths are relative to the daemon's **CWD**,
including paths written inside server TOML. `~` expands on the server host.
The native executable must be an absolute path. Global runtime `lora_path`
retains the existing model-store containment contract; see
[native configuration](native-config.md). The server TOML is separate from a
model definition; `[llm]`, `[mmproj]`, `[system]` and `[template]` belong in model
definitions. Unknown server tables/keys and wrong TOML types are refused.

## Server TOML and CLI reference

See [server.toml](../examples/server.toml) for a usable example. Omitted runtime
and sampling settings remain omitted, so native/model defaults keep applying.

| Server TOML key | CLI | Environment | Default |
| --- | --- | --- | --- |
| `server.host` | `--host` | `TLLAMA_HOST` | `127.0.0.1` |
| `server.port` | `--port` | `TLLAMA_PORT` | `54800` |
| `server.reload` | `--reload` / `--no-reload` | `TLLAMA_APP_RELOAD` | `false` |
| `server.log_level` | `--log-level` | `TLLAMA_LOG_LEVEL` | `info` |
| `server.origins` | `--origins` | `TLLAMA_ORIGINS` | no additions |
| `storage.models_dir` | `--models-dir` | `TLLAMA_MODELS` | `/var/lib/tllama/models` |
| `engine.executable` | `--runtime-path` | `TLLAMA_RUNTIME_EXECUTABLE` | `/usr/lib/tllama/server/tllama-server` |
| `engine.version_policy` | `--runtime-version-policy` | `TLLAMA_RUNTIME_VERSION_POLICY` | `warn` |
| `engine.backend_priority` | `--backend-priority` | `TLLAMA_BACKEND_PRIORITY` | `CUDA,ROCm,SYCL,Vulkan` |
| `engine.ignore_unsupported_runner` | `--ignore-unsupported-runner` / negative form | `TLLAMA_IGNORE_UNSUPPORTED_RUNNER` | `false` |
| `lifecycle.keep_alive` | `--keep-alive` | `TLLAMA_KEEP_ALIVE` | `5m` |
| `lifecycle.max_loaded_models` | `--max-loaded-models` | `TLLAMA_MAX_LOADED_MODELS` | `1` |
| `lifecycle.janitor_interval` | `--janitor-interval` | `TLLAMA_JANITOR_INTERVAL` | `10` seconds |
| `cache.metadata_ttl` | `--metadata-cache-ttl` | `TLLAMA_METADATA_CACHE_TTL` | `300` seconds |
| `telemetry.memory_source` | `--memory-source` | `TLLAMA_MEMORY_SOURCE` | `auto` |
| `telemetry.memory_compare` | `--memory-compare` / negative form | `TLLAMA_MEMORY_COMPARE` | `false` |
| `runtime.n_ctx` | `--context-length` | `TLLAMA_CONTEXT_LENGTH` | `0` = trained maximum |
| `runtime.device` | `--device` | `TLLAMA_RUNTIME_DEVICE` | automatic selection |
| `runtime.n_gpu_layers` | `--gpu-layers` | `TLLAMA_RUNTIME_N_GPU_LAYERS` | native fit when automatic |
| `runtime.n_threads` | `--threads` | `TLLAMA_RUNTIME_N_THREADS` | native automatic |
| `runtime.n_threads_batch` | `--threads-batch` | `TLLAMA_RUNTIME_N_THREADS_BATCH` | native automatic |
| `runtime.n_batch` | `--batch-size` | `TLLAMA_RUNTIME_N_BATCH` | `512` |
| `runtime.n_ubatch` | `--ubatch-size` | `TLLAMA_RUNTIME_N_UBATCH` | `512` |
| `runtime.flash_attn` | `--flash-attention` / negative form | `TLLAMA_RUNTIME_FLASH_ATTN` | `false` |
| `runtime.use_mmap` | `--mmap` / negative form | `TLLAMA_RUNTIME_USE_MMAP` | `true` |
| `runtime.use_mlock` | `--mlock` / negative form | `TLLAMA_RUNTIME_USE_MLOCK` | `false` |
| `runtime.type_kv` | `--kv-cache-type` | `TLLAMA_KV_CACHE_TYPE` | worker `f16` |
| `runtime.type_k` | `--k-cache-type` | `TLLAMA_K_CACHE_TYPE` | same-layer shorthand |
| `runtime.type_v` | `--v-cache-type` | `TLLAMA_V_CACHE_TYPE` | same-layer shorthand |

TOML uses native booleans, numbers and arrays; environment/CLI use strings,
with comma-separated values for origins, backend priorities and tensor splits.
Global `[runtime]` and `[sampling]` use TLlama's supported vocabulary from
[native-config.md](native-config.md), including typed runtime arrays/maps such as
`runtime.tensor_split` and `runtime.kv_overrides`. Global sampling accepts the
scalar environment vocabulary; `stop` and `logit_bias` remain per-model settings
and are refused in server TOML. Environment complex-value restrictions remain
unchanged. There is no arbitrary native CLI passthrough.

Within one layer, a dedicated port wins over a port inside `host`; a host-only
override retains the lower layer's port. Bracketed IPv6 and the legacy HTTP(S)
listener spelling work; an HTTP(S) spelling does not enable TLS.
Use `TLLAMA_LOG_LEVEL` and `TLLAMA_RUNTIME_FLASH_ATTN` for logging and flash
attention. Unknown nonempty `TLLAMA_*` environment settings stop startup rather
than being ignored. A K/V shorthand overrides both
lower-layer sides, then explicit same-layer sides override the shorthand.
`--print-config` shows those expanded effective sides and their origins.

These switches supply **global defaults**. A model's `[runtime].device`, context,
K/V types and other supported load settings can override them. Request settings
retain their existing precedence. `--device none` selects CPU as the global
default; it does not force every model to use CPU. Backend priority is used
when no explicit device was resolved; see [backend-priority.md](backend-priority.md).

## Logging and reload

Log levels are `critical`, `error`, `warning`, `info`, `debug` and `trace`.
Trace enables Uvicorn trace and TLlama debug logging. Routine outbound HTTP logs
are suppressed below debug; warnings remain visible subject to the selected
log threshold. Native subprocess diagnostics retain the worker's logging policy.

`--reload` is for development. The replacement process rereads the selected
TOML and replays CLI overrides, so it retains listener/storage/device defaults.
Uvicorn's watcher determines which changes trigger a reload; editing TOML alone
is not promised to trigger one. Restart the daemon to apply configuration
changes in normal operation. Loaded models are lost during process reload.
Applications embedding the ASGI `app` directly own their logging configuration;
the daemon's factory configures logging only when explicitly started.
