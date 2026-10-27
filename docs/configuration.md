# Configuration

TLlama has a server TOML file, daemon CLI flags, environment variables and a
separate `.toml` definition per model. Global precedence is **CLI > environment
> server TOML > defaults**. Model definitions override global load/sampling
defaults; supported request settings override those for the admitted request.
See [daemon.md](daemon.md) for server file selection, the full TOML/CLI/env
mapping, `--check-config` and `--print-config`.

This keeps a server easy to run from systemd, a container or a shell, while still letting one model be configured differently from the next without a second server.

---

## Quick example

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models
export TLLAMA_RUNTIME_FLASH_ATTN=true
export TLLAMA_KV_CACHE_TYPE=q8_0

tllamad
```

Then, against the Ollama-compatible API:

```bash
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

---

## How settings are resolved

### The layers

Weakest to strongest:

| Layer | Where | Scope |
|---|---|---|
| Native defaults | `tllama-server` / llama.cpp | settings TLlama leaves unset |
| TLlama's own baseline | built in | the few settings TLlama has a reason to differ on |
| Model header | the GGUF | that one model, where its author said something |
| Global daemon configuration | CLI > environment > server TOML | the whole server |
| Model definition | a model's `.toml` | that one model |
| Request | the API call | that one call |

A stronger layer wins only where it says something. TLlama-owned baselines and
native defaults are described in [native-config.md](native-config.md). Unset
threads/device defaults retain the native automatic policy.

GGUF sampling recommendations override TLlama's fallback values. Explicit
global, model and request settings override those recommendations. Values left
unset by TLlama retain native defaults where the launch contract permits it.

### Precedence per setting

Not every setting exists at every layer. Supported request runner options can
select a new immutable worker generation; admitted requests retain their old
generation while it drains. A GGUF cannot say how long to keep one in memory.

| Setting | Precedence, strongest first |
|---|---|
| Context length | request `num_ctx` → model `[runtime].n_ctx` → global configuration → `0`, meaning trained maximum |
| Load parameters | supported request runner options → model `[runtime]` → global configuration → TLlama/native defaults |
| K/V cache type | model `[runtime]` → global configuration → `f16`; expand shorthand within each layer |
| Sampling | request controls → model `[sampling]` → explicit global sampling → GGUF `general.sampling.*` → TLlama baseline |
| Stop strings | request `stop` → `.toml` `[sampling] stop` → none (no environment layer: a list is not a string) |
| Chat template | `.toml` `[template] jinja` → the GGUF's own template |
| System prompt | the request's own system message → `.toml` `[system] prompt` |
| Keep-alive | request `keep_alive` → global `lifecycle.keep_alive` |

A GGUF may carry the model author's recommended sampling values under `general.sampling.*` — `temp`, `top_k` and `top_p`, no others. TLlama applies them directly above its own baseline, so they take effect on a model whose `.toml` pins nothing and the server sets nothing, and are overridden by either. `/api/show` reports them regardless. It is a newer convention and most files do not carry it; its absence says nothing about a model.

### When a setting is wrong

A value TLlama cannot use stops the server at startup, with one line naming the variable and what was expected. It is not ignored and not silently replaced by a default: for a service started from a unit file, a setting that quietly does not apply is indistinguishable from one that applied and made no difference.

```
$ TLLAMA_MAX_LOADED_MODELS=lots tllamad
tllamad: TLLAMA_MAX_LOADED_MODELS (lifecycle.max_loaded_models): invalid literal for int() with base 10: 'lots'
$ echo $?
2
```

An unset or empty variable is not an error. `TLLAMA_CONTEXT_LENGTH=` in a unit file says "I set nothing", not "I set nonsense", and means the default.

Whether a value converts and whether it means anything are separate questions, and TLlama's own numeric settings are checked for both. `TLLAMA_JANITOR_INTERVAL=0` converts to a number and then asks for a wait that does not wait; `TLLAMA_MAX_LOADED_MODELS=0` converts but is below the accepted minimum of `1`. Each section below states the accepted range.

Sampling values are typed and must be finite. Disable modes such as `top_k <= 0` and `top_p >= 1` remain accepted. Negative `temperature` is refused: the native runtime would turn it into greedy sampling instead of reproducing the former binding chain.

`mirostat_mode` is an enum: only `0`, `1` and `2` are accepted. Other values
are refused by TLlama instead of falling through to another native sampler.

A model's `.toml` works the same way: a key in `[runtime]` or `[sampling]` that TLlama will not act on is refused when the file is read, and that model is not listed and cannot be loaded until the file is corrected. `tllamad rebuildrepo --dryrun` reports which file and which key, and writes nothing.

Booleans accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`, in any case. Anything else is refused.

---

## Server options

### `TLLAMA_IGNORE_UNSUPPORTED_RUNNER`

Defaults to `false`. Ollama requests may select `llamacpp` or its `ggml`
alias; another runner returns HTTP 400 before any model or download action.
Set this flag to `true` to silently ignore unsupported runner strings from
clients. Non-string values are still rejected. This never selects a different
inference engine or GPU backend; reported runner remains `llamacpp`.

### `TLLAMA_HOST`

Address and port to listen on. Default `127.0.0.1:54800`.

```bash
export TLLAMA_HOST=127.0.0.1:54800      # host and port
export TLLAMA_HOST=0.0.0.0              # host only, default port
export TLLAMA_HOST=:8080                # port only, default host
export TLLAMA_HOST=[::1]:54800          # IPv6, bracketed
export TLLAMA_HOST=::1                  # IPv6, bare, cannot carry a port
export TLLAMA_HOST=http://0.0.0.0:54800
```

A `http://` or `https://` prefix is accepted and discarded, that being a documented way to write `OLLAMA_HOST`. The scheme says nothing about how the server listens, and `https://` does **not** enable TLS; that belongs to a reverse proxy in front. Any other scheme is refused rather than ignored.

Brackets around an IPv6 address are how the written form separates address from port. They are not part of the address and are removed before binding.

The port has to be from `1` to `65535`. `0` is refused rather than passed on: it is the spelling for "any free port", so the server would start and listen somewhere nobody can predict.

> Do not expose TLlama to an untrusted network without a reverse proxy, authentication, TLS and access controls.

### `TLLAMA_ORIGINS`

Origins a browser is allowed to reach TLlama from, comma separated. Added to a built-in list, never replacing it, so a page on the local machine keeps working whatever else is configured.

```bash
export TLLAMA_ORIGINS=https://ui.example
export TLLAMA_ORIGINS=https://ui.example,http://192.168.1.10:*
```

A `*` stands in for any part of an origin, most usefully a port. Each entry is `scheme://host` with an optional port and **no path**; a browser never puts a path in an `Origin` header, so an entry carrying one could never match and stops the server instead of silently doing nothing.

Allowed without configuring anything:

| Origin | |
|---|---|
| `http://localhost`, `https://localhost`, and the same with any port | a page served locally |
| `http://127.0.0.1`, `http://[::1]`, likewise with any port and over `https` | the same, by address |
| `vscode-webview://*`, `vscode-file://*` | a VS Code extension drawing its UI in a webview |

The effective list is written to the log at startup.

This differs from Ollama's built-in list on purpose. `0.0.0.0` is an address to listen on, not one a browser connects to. `file://` is not allowed because a page opened off disk sends `Origin: null` in Chromium, and allowing `null` would allow every sandboxed iframe on the web along with it. `app://` and `tauri://` belong to Ollama's own desktop application. `[::1]` is added, which Ollama does not list, because TLlama binds to IPv6.

> **CORS is not access control.** It is enforced by browsers and governs whether a page may *read* a reply. It stops nothing sent by `curl`, a script, or any other client, and requests that need no preflight reach TLlama and run whatever the origin list says. What keeps TLlama private is the address it listens on and what sits in front of it.

Credentials are never allowed: TLlama has no cookie or session for a browser to send.

A request carrying an `Origin` header must send `Content-Type: application/json`; anything else is refused with `415`. Everywhere else TLlama accepts a JSON body whatever the content type says, because `curl -d` with no header sends `application/x-www-form-urlencoded` and the Ollama documentation shows exactly that. Those lenient content types are also the ones that let a browser skip the preflight entirely, so accepting them from a page would put the request beyond the reach of the origin list. `curl` never sends `Origin` and is unaffected; `fetch()` sends it always and should be declaring JSON in any case.

### `TLLAMA_LOG_LEVEL`

`TLLAMA_LOG_LEVEL` selects critical/error/warning/info/debug/trace. The default
is info. CLI `--log-level` overrides the environment. Trace enables Uvicorn trace and TLlama debug output.

Routine outbound HTTP request and transport logs are suppressed below debug.
Native worker memory polling remains enabled; quieter logging does not change
telemetry or `/api/ps`. Native worker diagnostics use the worker logger; errors
and load failures retain their diagnostics. There is no in-process binding
redirecting the Python server's file descriptors or printing model responses.
Direct ASGI embedders own their logging setup.

### `TLLAMA_APP_RELOAD`

Reload the application when its source changes. Default `false`. For development: it restarts the process, so loaded models are dropped.

---

## Model repository options

### `TLLAMA_MODELS`

The repository root, holding `Local/`, `TLlama/` and `HuggingFace/`. Default `/var/lib/tllama/models`.

A `.gguf` that no `.toml` points at is not a model: it is not listed and cannot be loaded. See `docs/model-repositories.md`; `tllamad rebuildrepo` writes the missing definitions.

### `TLLAMA_METADATA_CACHE_TTL`

How long in-memory metadata stays valid, in seconds. Default `300.0`. `0` or more; `0` makes every entry stale on arrival, which turns the in-memory cache off. The on-disk cache is separate and keyed by content rather than by time; see `docs/metadata-cache.md`.

---

## Model lifecycle options

### `TLLAMA_CONTEXT_LENGTH`

Default context length. Default `0`, meaning the model's trained maximum. `0` or more.

That differs from Ollama, whose `OLLAMA_CONTEXT_LENGTH` defaults to 4096. Whether a model's maximum fits depends on the KV cache type and flash attention: a large context with a quantized cache and flash attention on can take less memory than a modest one without.

Overridden per model by `[runtime] n_ctx`, per request by `num_ctx`.

### `TLLAMA_KEEP_ALIVE`

How long a model stays loaded after its last use. Default `5m`.

| Value | Meaning |
|---|---|
| `30s`, `5m`, `1h` | that duration |
| a bare number | that many seconds |
| `0` | unload as soon as the request finishes |
| any negative value | keep until the server stops |

Overridden per request by `keep_alive`, which is how Ollama treats `OLLAMA_KEEP_ALIVE`.

### `TLLAMA_MAX_LOADED_MODELS`

How many models may be resident at once. Default `1`, and `1` is the smallest accepted value. Asking for one more than fits unloads to make room.

### `TLLAMA_JANITOR_INTERVAL`

How often expired models are collected, in seconds. Default `10.0`. Must be greater than `0`: a delay of zero or less does not wait at all.

---

## Load parameters: `TLLAMA_RUNTIME_*`

Uses TLlama's pinned native runtime vocabulary, independent of the installed Python binding. A variable is the parameter name with the `TLLAMA_RUNTIME_` prefix:

```bash
export TLLAMA_RUNTIME_DEVICE=Vulkan0
export TLLAMA_RUNTIME_N_THREADS=8
export TLLAMA_RUNTIME_USE_MLOCK=true
export TLLAMA_RUNTIME_TENSOR_SPLIT=0.6,0.4
```

Native configuration and its defaults, CLI mapping and removal matrix are documented in [native-config.md](native-config.md). Only listed parameters are accepted. Unknown names, removed settings, invalid types and unrepresentable enums produce a configuration error.

Whole numbers, finite floats, boolean words (`1/true/yes/on`, `0/false/no/off`) and comma-separated tensor proportions are supported. `numa` accepts disabled/strategy numbers or a boolean. `kv_overrides` is TOML-only. The model path comes from `[llm]`; context and K/V types have dedicated environment variables rather than duplicate `TLLAMA_RUNTIME_*` spellings.

Public routes use managed native workers. Model TOML overrides the global device selector. If neither is set, [backend priorities](backend-priority.md) select the device at model load. A selector is not a stable physical GPU ID.

### `TLLAMA_BACKEND_PRIORITY`

Ordered automatic acceleration preferences, default `CUDA,ROCm,SYCL,Vulkan`.
For example, `TLLAMA_BACKEND_PRIORITY="Vulkan,ROCm"` tries only these GPU
backends, then CPU. An **unset** variable uses the default; an explicitly
**empty** value means CPU-only. `CPU` is a terminal entry and can appear first.
Explicit model/global `device` overrides the entire priority list, including its
allowlist. Names are case-insensitive, whitespace is trimmed, and duplicates
retain the first occurrence. CPU helper libraries such as BLAS/BLIS are not
independent placement choices. See [backend-priority.md](backend-priority.md)
for matching, failure handling and native fitting limits.

### `TLLAMA_RUNTIME_FLASH_ATTN`

Sets the global `runtime.flash_attn` default. A model definition can override it.

Compatibility depends on the backend, model, cache type and context. The
audited native runtime requires flash attention for a quantized **V** cache;
with FA explicitly off, that configuration fails loading rather than ignoring
the cache setting. K-only quantization has a separate native path and must not
be treated as the same requirement. FA also affects projector compute buffers.
A parsed setting does not prove that the selected model/backend supports it.

---

## KV cache type

### `TLLAMA_KV_CACHE_TYPE`, `TLLAMA_K_CACHE_TYPE`, `TLLAMA_V_CACHE_TYPE`

Quantization of the key/value cache. The global environment values are unset
by default; TLlama's resolved worker baseline is explicitly `f16` on both sides.

`TLLAMA_KV_CACHE_TYPE` sets both sides; the per-side variables override it for their own side. That is the relationship `type_kv`, `type_k` and `type_v` have in a `.toml` `[runtime]`, resolved by the same code, so a name means the same thing in both places.

```bash
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_V_CACHE_TYPE=q4_0     # a smaller V than K
```

Supported native KV names are `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`. TOML also accepts the corresponding pinned GGML enum integers. Other weight quantization types are not KV types and are refused. The native resolver explicitly defaults both sides to `f16`.

Resolving is not the same as being usable: `llama.cpp` supports only some ggml types as a KV cache, and one it does not support is refused when a model loads, which startup cannot check.

Ollama has no per-side equivalent; `OLLAMA_KV_CACHE_TYPE` sets both.

---

## Sampling: `TLLAMA_SAMPLING_*`

Sampling set for the whole server. A model's `.toml` `[sampling]` overrides it, and a request's own options override both.

```bash
export TLLAMA_SAMPLING_TEMPERATURE=0.6
export TLLAMA_SAMPLING_TOP_K=20
```

Global sampling values are unset by default. Explicit global values override
GGUF recommendations, while model `[sampling]` and request controls override
the global values. If none is configured, GGUF recommendations and then the
TLlama baseline supply the fallback.

Names and types come from TLlama's own sampling contract, also used for TOML and request validation. They do not change when a Python binding adds parameters. `logit_bias` and `stop` remain TOML-only because their structured values cannot be expressed through this environment interface.

A value that will not convert stops the server at startup rather than failing on the first request that generates anything.

Ollama has no equivalent. Its sampling is per-model or per-request only.

---

## Per-model configuration: the `.toml`

One file per model, named for the model. The repository layout and naming rules are in `docs/model-repositories.md`; what follows is the configuration each section carries.

```toml
[llm]
model = "HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"

[mmproj]
model = "HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B.gguf"

[runtime]
n_ctx = 8192
flash_attn = true
type_kv = "q4_0"

[sampling]
temperature = 0.7
top_p = 0.95
top_k = 20
stop = ["<|im_end|>"]

[system]
prompt = """
You are a careful assistant.
"""

[template]
jinja = """
{% for message in messages %}...{% endfor %}
"""
```

The six sections above and optional `[fim]` are the supported sections. A name outside them is refused when the file is read, and so is a key inside a section that nothing acts on. That includes a section named in the plans but not yet implemented, `[lora]` among them: a file asking for an adapter and getting none would run and answer, which is a wrong answer rather than a missing feature.

A misspelt section name is the mistake this catches that the per-section checks could not. `[smapling]` is valid TOML and reads as a section of its own, so every value under it used to be loaded and dropped in one go, leaving a file that looked obeyed. Shape counts as well: `[[runtime]]` is one bracket from `[runtime]` and means something else entirely, and a key written above the first section belongs to no section at all.

### `[llm]` and `[mmproj]`

`model` is a path relative to the repository root, including the category directory. `tllama create -f FILE` imports client-side `from` files into the server's `Local/imports/` storage and rewrites them to `model` in the stored TOML. Relative `from` paths are relative to the client's current working directory, regardless of FILE's location; the HTTP server does not read arbitrary server-side paths from this field. Exactly one of the two is required per section. A definition-only create with `model` reuses existing repository files without uploading them. See [CLI create/import](cli.md#create-and-import-models).

A path escaping the repository root is refused. An `[mmproj]` is checked to be a projector.

`[mmproj]` takes no keys beyond those two and the projector context settings below; an unrecognised one is refused when the file is read.

#### Projector context

The native resolver maps these projector settings to managed worker CLI
parameters. `warmup` affects the whole worker, including the text model.

| Key | Type | Notes |
| --- | --- | --- |
| `warmup` | boolean | Whole-worker warmup; false skips it |
| `image_min_tokens` | integer | Image processing lower limit; -1 uses native/model policy |
| `image_max_tokens` | integer | Image processing upper limit; -1 uses native/model policy |
| `batch_max_tokens` | integer | Native multimodal batch token limit |

Unknown projector keys are refused. Actual image decoding and inference belong
to the native runtime; valid configuration alone does not certify model/backend
compatibility. See [native-vision.md](native-vision.md) and
[native-config.md](native-config.md) for mapping, input and failure contracts.

Several models may point at one physical file; the weights are not copied.

### `[runtime]`

The supported parameters in [native-config.md](native-config.md), including `type_kv`. Overrides `TLLAMA_RUNTIME_*` for this model.

`model_path` is refused: the path comes from `[llm]`, and naming it here would mean the file defines one model and loads another.

`n_ctx` is read here, but as one layer of the context-length chain rather than as a raw argument. `type_k`, `type_v` and `type_kv` take either a ggml type name or a raw integer.

An unrecognised key is refused when the file is read.

### `[sampling]`

Overridden by a request's own `options`, and overriding `TLLAMA_SAMPLING_*`, the GGUF's recommended values and TLlama's baseline.

The baselines below apply only where nothing else does. On a model whose GGUF recommends `temperature`, `top_k` or `top_p`, that recommendation applies instead.

| Key | Baseline | Notes |
|---|---:|---|
| `temperature` | `0.8` | TLlama baseline; explicit global/model/request values and GGUF recommendations take precedence |
| `top_p` | `0.9` | Ollama's documented default, chosen over llama.cpp's `0.95` |
| `top_k` | `40` | |
| `min_p` | `0.05` | |
| `typical_p` | `1.0` | `1.0` disables |
| `presence_penalty` | `0.0` | |
| `frequency_penalty` | `0.0` | |
| `repeat_penalty` | `1.0` | `1.0` disables |
| `mirostat_mode` | `0` | `0`, `1` or `2`; anything else is refused. Ollama's client options spell this `mirostat` |
| `mirostat_tau` | `5.0` | |
| `mirostat_eta` | `0.1` | |
| `seed` | unset | |
| `logit_bias` | unset | Table of token id to bias, e.g. `{ "128009" = -100.0 }`. Keys are token ids; no request field carries this, so it is settable only here |
| `max_tokens` | unset | Ollama's client options spell this `num_predict` |
| `stop` | none | Array of strings |

`tfs_z` is refused even for `1.0`; it has no supported native equivalent.
Sampling types and enum values follow the pinned TLlama contract, rather than
every option upstream happens to provide. Public structured output is described
in [native-public-api.md](native-public-api.md); raw grammar passthrough is not
implicitly enabled by the server TOML.

Applied identically by `/api/generate`, `/api/chat` and `/v1/chat/completions`, so a model behaves the same whichever endpoint a client uses.

### `[system]`

`prompt` is the system prompt used when the request carries none of its own. A request always wins. It is the section's only key.

### `[template]`

`jinja` is a chat template overriding the GGUF's own, for text and vision models alike. Native chat renders the selected template. Completion prompt rendering uses
TLlama's sandboxed renderer where applicable. It is the section's only key.

---

### `[fim]`

`order` selects `auto` (default), `psm`, `training_spm`, `native_spm` or
`disabled`. It applies only to nonempty generate `suffix`; model chat defaults
are bypassed for that operation. See [native-fim.md](native-fim.md) for recipes,
validation, native stops and the separate model-quality qualification.

---

## Example configurations

### Minimal

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models

tllamad
```

### Lower memory

```bash
export TLLAMA_RUNTIME_FLASH_ATTN=true
export TLLAMA_KV_CACHE_TYPE=q4_0
export TLLAMA_CONTEXT_LENGTH=8192
export TLLAMA_MAX_LOADED_MODELS=1

tllamad
```

This example quantizes both sides, so its V cache requires flash attention.
Check model/backend support and output quality for the chosen precision.

### Two GPUs, split by proportion

```bash
export TLLAMA_RUNTIME_TENSOR_SPLIT=0.7,0.3
export TLLAMA_RUNTIME_N_GPU_LAYERS=-1

tllamad
```

### One GPU only, on a machine with several

```bash
export TLLAMA_RUNTIME_SPLIT_MODE=0
export TLLAMA_RUNTIME_MAIN_GPU=0

tllamad
```

`main_gpu` indexes the native configured device list; it is not a persistent
physical GPU ID. Select the backend/device explicitly with `device` when that
is the intent. Do not assume indices or discovery rows match across APIs; see
[device-inventory.md](device-inventory.md).

### CPU-heavy machine

```bash
export TLLAMA_RUNTIME_N_THREADS=16
export TLLAMA_RUNTIME_N_THREADS_BATCH=32
export TLLAMA_RUNTIME_USE_MLOCK=true

tllamad
```

### LAN-accessible

```bash
export TLLAMA_HOST=0.0.0.0:54800

tllamad
```

Use a reverse proxy and access controls beyond localhost.

---

## systemd

Install the supplied `systemd/tllamad.service` with the daemon account and
`/etc/tllama/server.toml` as described in [systemd.md](systemd.md). The service
runs `tllamad`, checks configuration before serving, and retains managed-worker
shutdown. CLI > environment > server TOML applies to service overrides too.

---

## Troubleshooting

### The server will not start

Read the last line: it names the variable and what was expected.

```
tllamad: TLLAMA_HOST='0.0.0.0:http' is not valid: expected HOST:PORT with a numeric port.
```

### A model disappeared from the listing

Its `.toml` no longer parses, most likely because it carries a key TLlama will not act on. Nothing is deleted; the model is invisible until the file is fixed.

```bash
tllamad rebuildrepo --dryrun
```

That names the file and the key, and writes nothing. A common case is a setting in the wrong section: context length is `n_ctx` in `[runtime]`, not `num_ctx` in `[sampling]`.

### A setting appears to do nothing

Check the layer above it. A model's `[runtime]` overrides `TLLAMA_RUNTIME_*`, and a request's `options` override `[sampling]`.

For a quantized V cache, enable supported flash attention or use `f16` V.
FA explicitly off with quantized V is a native load error. Inspect resolved
configuration with `tllamad --print-config` and retain the worker diagnostics.

### A browser cannot reach the server

The browser console reports a CORS failure, or a plain network error with nothing in TLlama's log. The page's origin is not on the list; the startup log line beginning `Browser origins allowed:` says what is. Add it with `TLLAMA_ORIGINS`.

An origin is the scheme, host and port exactly as the page was loaded, so `http://localhost:3000` and `http://127.0.0.1:3000` are two different origins even though they are one machine. Both are allowed by default, on any port.

A browser caches a preflight for ten minutes, so a corrected origin list can take that long to take effect in a tab that is already open.

### The repository looks empty

```bash
echo "$TLLAMA_MODELS"
find "$TLLAMA_MODELS" -name '*.gguf' | head
tllamad rebuildrepo --dryrun
```

A `.gguf` with no `.toml` is not a model. `rebuildrepo` writes the missing definitions.

### The first listing is slow

The persistent metadata cache is not built yet.

```bash
find "$TLLAMA_MODELS/.tllama/metadata-cache" -type f -name '*.json' | head
```

---

### Pull fails with permission denied

Check the actual failing path and the daemon account, including parent-directory
traversal, model-store writes and the Hugging Face cache. Under systemd, inspect
`systemctl cat tllamad.service` and its journal; the supplied service uses the
`tllama` account, not the interactive shell user. Run configuration inspection
as that account. Fix ownership/ACLs for the intended directories once identified;
a successful interactive pull does not verify service permissions.

### Runtime fingerprint warning

`engine.version_policy = "warn"` logs an unaudited identity and continues;
`"strict"` refuses it. This policy is shared by generation, embeddings and
Decision. Malformed native protocol remains an error under either policy.
Diagnostic scripts may require an exact build independently of ordinary service
admission. See [runtime-audit-070.md](runtime-audit-070.md).

---

## Related

- `README.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
- `docs/usage.md`
- `docs/development.md`
