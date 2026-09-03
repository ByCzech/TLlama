# Configuration

TLlama is configured from environment variables and from a `.toml` file per model. Environment variables say what is true for the whole server; a model's `.toml` says what is true for that model and overrides the server-wide value where the two overlap. An API request overrides both, for the things a request is allowed to decide.

This keeps a server easy to run from systemd, a container or a shell, while still letting one model be configured differently from the next without a second server.

---

## Quick example

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models
export TLLAMA_RUNTIME_FLASH_ATTN=true
export TLLAMA_KV_CACHE_TYPE=q8_0

tllama
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
| Library and model defaults | `llama-cpp-python`, the GGUF header | whatever nothing else sets |
| Environment | `TLLAMA_*` | the whole server |
| Model definition | a model's `.toml` | that one model |
| Request | the API call | that one call |

A stronger layer wins only where it says something. TLlama does not restate a default it agrees with: where nothing sets a value, the argument is not passed at all and `llama-cpp-python`'s own default applies. That is deliberate, since a restated default is a copy that can drift from what it copies.

### Precedence per setting

Not every setting exists at every layer. A request cannot change how a model was loaded, and a GGUF cannot say how long to keep one in memory. Exactly:

| Setting | Precedence, strongest first |
|---|---|
| Context length | request `num_ctx` → `.toml` `[runtime] n_ctx` → `TLLAMA_CONTEXT_LENGTH` → `0`, meaning the model's trained maximum |
| Load parameters | `.toml` `[runtime]` → `TLLAMA_RUNTIME_*` → `Llama()` default |
| K/V cache type | `.toml` `[runtime]` `type_k`/`type_v`/`type_kv` → `TLLAMA_K_CACHE_TYPE`/`TLLAMA_V_CACHE_TYPE`/`TLLAMA_KV_CACHE_TYPE` → unset |
| Sampling | request `options` → `.toml` `[sampling]` → `TLLAMA_SAMPLING_*` → TLlama's baseline |
| Stop strings | request `stop` → `.toml` `[sampling] stop` → none (no environment layer: a list is not a string) |
| Chat template | `.toml` `[template] jinja` → the GGUF's own template |
| System prompt | the request's own system message → `.toml` `[system] prompt` |
| Keep-alive | request `keep_alive` → `TLLAMA_KEEP_ALIVE` |

A GGUF may carry the model author's recommended sampling values under `general.sampling.*`. TLlama reads them and reports them through `/api/show`, but does not yet apply them; when it does they will sit between the environment and the `.toml`.

### When a setting is wrong

A value TLlama cannot use stops the server at startup, with one line naming the variable and what was expected. It is not ignored and not silently replaced by a default: for a service started from a unit file, a setting that quietly does not apply is indistinguishable from one that applied and made no difference.

```
$ TLLAMA_MAX_LOADED_MODELS=lots tllama
tllama: TLLAMA_MAX_LOADED_MODELS='lots' is not valid: expected a whole number.
$ echo $?
2
```

An unset or empty variable is not an error. `TLLAMA_CONTEXT_LENGTH=` in a unit file says "I set nothing", not "I set nonsense", and means the default.

A model's `.toml` works the same way: a key in `[runtime]` or `[sampling]` that TLlama will not act on is refused when the file is read, and that model is not listed and cannot be loaded until the file is corrected. `tllama rebuildrepo --dryrun` reports which file and which key, and writes nothing.

Booleans accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`, in any case. Anything else is refused.

---

## Server options

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


Debug logging. Default `false`.

### `TLLAMA_APP_RELOAD`

Reload the application when its source changes. Default `false`. For development: it restarts the process, so loaded models are dropped.

---

## Model repository options

### `TLLAMA_MODELS`

The repository root, holding `Local/`, `TLlama/` and `HuggingFace/`. Default `/var/lib/tllama/models`.

A `.gguf` that no `.toml` points at is not a model: it is not listed and cannot be loaded. See `docs/model-repositories.md`; `tllama rebuildrepo` writes the missing definitions.

### `TLLAMA_METADATA_CACHE_TTL`

How long in-memory metadata stays valid, in seconds. Default `300.0`. The on-disk cache is separate and keyed by content rather than by time; see `docs/metadata-cache.md`.

### `TLLAMA_MODEL_SCAN_TIMEOUT`

How long a metadata scan may take, in seconds. Default `5.0`.

---

## Model lifecycle options

### `TLLAMA_CONTEXT_LENGTH`

Default context length. Default `0`, meaning the model's trained maximum.

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

How many models may be resident at once. Default `1`. Asking for one more than fits unloads to make room.

### `TLLAMA_JANITOR_INTERVAL`

How often expired models are collected, in seconds. Default `10.0`.

---

## Load parameters: `TLLAMA_RUNTIME_*`

Sets any load parameter of `llama-cpp-python`'s `Llama()` for every model. The variable is the parameter name with a `TLLAMA_RUNTIME_` prefix:

```bash
export TLLAMA_RUNTIME_N_THREADS=8
export TLLAMA_RUNTIME_USE_MLOCK=true
export TLLAMA_RUNTIME_MAIN_GPU=1
export TLLAMA_RUNTIME_TENSOR_SPLIT=0.6,0.4
```

Which names exist, and what each takes, is read from `Llama()`'s signature in the installed build. No list in TLlama can disagree with it, and a parameter a future `llama-cpp-python` adds becomes settable without a TLlama change. A name that is not a parameter, or a value of the wrong shape, stops the server at startup.

Values are converted according to the declared type: whole numbers for `int`, `1/true/yes/on` and `0/false/no/off` for `bool`, comma-separated numbers for `tensor_split`, and either a word or a number for `numa`.

The table is that signature as of `llama-cpp-python` 0.3.35. Descriptions are TLlama's summary; `llama.cpp` is the authority on what each does.

| Parameter | Type | `Llama()` default | Notes |
|---|---|---:|---|
| `n_gpu_layers` | int | `0` | Layers offloaded to GPU, `-1` for all. **TLlama passes `-1`** unless overridden |
| `split_mode` | int | `1` | `0` none, `1` by layer, `2` by row |
| `main_gpu` | int | `0` | Index into the device list after llama.cpp merges duplicates |
| `tensor_split` | list of float | unset | Proportion per device, comma-separated |
| `vocab_only` | bool | `False` | Vocabulary only, no weights |
| `use_mmap` | bool | `True` | Map the file rather than read it |
| `use_mlock` | bool | `False` | Keep the model resident in RAM |
| `kv_overrides` | dict | unset | Not settable from the environment |
| `seed` | int | `4294967295` | Load-time seed |
| `n_ctx` | int | `512` | Set by `TLLAMA_CONTEXT_LENGTH`, not settable here |
| `n_batch` | int | `512` | Logical batch size |
| `n_ubatch` | int | `512` | Physical batch size |
| `n_threads` | int | unset | Generation threads |
| `n_threads_batch` | int | unset | Prompt-processing threads |
| `rope_scaling_type` | int | `-1` | `-1` unspecified |
| `pooling_type` | int | `-1` | For embedding models |
| `attention_type` | int | `-1` | `-1` unspecified |
| `rope_freq_base` | float | `0.0` | `0.0` takes the model's own |
| `rope_freq_scale` | float | `0.0` | `0.0` takes the model's own |
| `yarn_ext_factor` | float | `-1.0` | YaRN context extension |
| `yarn_attn_factor` | float | `1.0` | |
| `yarn_beta_fast` | float | `32.0` | |
| `yarn_beta_slow` | float | `1.0` | |
| `yarn_orig_ctx` | int | `0` | `0` takes the model's trained context |
| `logits_all` | bool | `False` | Logits for every token |
| `embedding` | bool | `False` | Embedding mode |
| `offload_kqv` | bool | `True` | KQV computation on GPU |
| `flash_attn` | bool | `False` | Also reachable as `TLLAMA_FLASH_ATTENTION` |
| `op_offload` | bool | unset | |
| `swa_full` | bool | unset | Full sliding-window attention cache |
| `no_perf` | bool | `False` | Skip performance counters |
| `last_n_tokens_size` | int | `64` | Window the repeat penalty looks back over |
| `lora_base` | str | unset | |
| `lora_scale` | float | `1.0` | |
| `lora_path` | str | unset | Turns `use_mmap` off by itself |
| `numa` | bool or int | `False` | A word for the bool, digits for a strategy number |
| `chat_format` | str | unset | Named prompt format; `[template]` is usually the better tool |
| `chat_handler` | object | unset | Not settable from the environment; comes from `[mmproj]` |
| `draft_model` | object | unset | Not settable from the environment |
| `tokenizer` | object | unset | Not settable from the environment |
| `type_k` | int | unset | Set by `TLLAMA_K_CACHE_TYPE`, not settable here |
| `type_v` | int | unset | Set by `TLLAMA_V_CACHE_TYPE`, not settable here |
| `spm_infill` | bool | `False` | SentencePiece infill |
| `verbose` | bool | `True` | **TLlama requires this on**: `/api/ps` takes every number it reports from the load log |
| `model_path` | str | required | Comes from `[llm]`, never settable |

`chat_handler`, `draft_model`, `tokenizer` and `kv_overrides` take values no string can express. Four more are refused with a message saying where they are set instead:

| Parameter | Set by |
|---|---|
| `model_path` | `[llm]` in a model's `.toml` |
| `n_ctx` | `TLLAMA_CONTEXT_LENGTH` |
| `type_k` | `TLLAMA_K_CACHE_TYPE` or `TLLAMA_KV_CACHE_TYPE` |
| `type_v` | `TLLAMA_V_CACHE_TYPE` or `TLLAMA_KV_CACHE_TYPE` |

### `TLLAMA_FLASH_ATTENTION`

An alias for `TLLAMA_RUNTIME_FLASH_ATTN`, kept because it is the spelling Ollama uses. An explicit `TLLAMA_RUNTIME_FLASH_ATTN` overrides it.

Whether it helps depends on the backend, the model, the quantization and the context length. Two consequences are worth knowing: without flash attention `llama.cpp` ignores a quantized KV cache, so the cache-type variables below have no effect; and a vision model's projector can need dramatically more compute buffer without it.

---

## KV cache type

### `TLLAMA_KV_CACHE_TYPE`, `TLLAMA_K_CACHE_TYPE`, `TLLAMA_V_CACHE_TYPE`

Quantization of the key/value cache. All unset by default, leaving the choice to `llama.cpp`.

`TLLAMA_KV_CACHE_TYPE` sets both sides; the per-side variables override it for their own side. That is the relationship `type_kv`, `type_k` and `type_v` have in a `.toml` `[runtime]`, resolved by the same code, so a name means the same thing in both places.

```bash
export TLLAMA_KV_CACHE_TYPE=q8_0
export TLLAMA_V_CACHE_TYPE=q4_0     # a smaller V than K
```

A value is a ggml type name resolved against the `GGML_TYPE_*` constants the installed build defines — `f16`, `q8_0`, `q4_0` and the rest. It is not checked against a fixed list, so a type a future `llama.cpp` adds works without a TLlama change. A name matching no constant stops the server at startup.

Resolving is not the same as being usable: `llama.cpp` supports only some ggml types as a KV cache, and one it does not support is refused when a model loads, which startup cannot check.

Ollama has no per-side equivalent; `OLLAMA_KV_CACHE_TYPE` sets both.

---

## Sampling: `TLLAMA_SAMPLING_*`

Sampling set for the whole server. A model's `.toml` `[sampling]` overrides it, and a request's own options override both.

```bash
export TLLAMA_SAMPLING_TEMPERATURE=0.6
export TLLAMA_SAMPLING_TOP_K=20
```

**Everything here is unset by default, and that matters.** A global sampling value overrules what a model's own definition recommends, for every model at once. Shipping numbers in this layer would silence those recommendations on a server nobody configured, so an unset variable stays genuinely unset and TLlama's baseline applies as before.

Which names exist is read off `create_completion()` and `create_chat_completion()` in the installed `llama-cpp-python`, narrowed to what TLlama actually applies — the same set a `.toml` `[sampling]` may use, listed under [`[sampling]`](#sampling) below. A name outside it stops the server at startup, and says which of the two reasons applies: TLlama does not apply this parameter, or the library has no such parameter at all.

The type each value has to become is read off the same signatures. `logit_bias` and `stop` are therefore refused here — a table and a list are not things an environment variable can hold — with a message pointing at the `.toml`, where they can be set.

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

### `[llm]` and `[mmproj]`

`model` is a path relative to the repository root, including the category directory. `from` imports a file from outside the repository instead, physically into `TLlama/`, and is rewritten to `model` afterwards. Exactly one of the two.

A path escaping the repository root is refused. An `[mmproj]` is checked to be a projector.

Several models may point at one physical file; the weights are not copied.

### `[runtime]`

Any `Llama()` parameter from the table above, plus `type_kv`. Overrides `TLLAMA_RUNTIME_*` for this model.

`model_path` is refused: the path comes from `[llm]`, and naming it here would mean the file defines one model and loads another.

`n_ctx` is read here, but as one layer of the context-length chain rather than as a raw argument. `type_k`, `type_v` and `type_kv` take either a ggml type name or a raw integer.

An unrecognised key is refused when the file is read.

### `[sampling]`

Overridden by a request's own `options`, and overriding `TLLAMA_SAMPLING_*` and TLlama's baseline.

| Key | Baseline | Notes |
|---|---:|---|
| `temperature` | `0.8` | Matches llama.cpp and Ollama, not `llama-cpp-python`, whose two completion calls disagree with each other |
| `top_p` | `0.9` | Ollama's documented default, chosen over llama.cpp's `0.95` |
| `top_k` | `40` | |
| `min_p` | `0.05` | |
| `typical_p` | `1.0` | `1.0` disables |
| `presence_penalty` | `0.0` | |
| `frequency_penalty` | `0.0` | |
| `repeat_penalty` | `1.0` | `1.0` disables |
| `tfs_z` | `1.0` | `1.0` disables |
| `mirostat_mode` | `0` | Ollama's client options spell this `mirostat` |
| `mirostat_tau` | `5.0` | |
| `mirostat_eta` | `0.1` | |
| `seed` | unset | |
| `logit_bias` | unset | Table of token id to bias, e.g. `{ "128009" = -100.0 }`. Keys are token ids; no request field carries this, so it is settable only here |
| `max_tokens` | unset | Ollama's client options spell this `num_predict` |
| `stop` | none | Array of strings |

This set is still narrower than what `llama-cpp-python`'s completion calls accept. `grammar` is the notable absentee, and deliberately so: GBNF that llama.cpp cannot parse takes the server process down with `SIGSEGV` rather than returning an error, and nothing in the installed library parses it early enough to catch that first. A name that is real to the library but unapplied here is refused rather than accepted and dropped.

Applied identically by `/api/generate`, `/api/chat` and `/v1/chat/completions`, so a model behaves the same whichever endpoint a client uses.

### `[system]`

`prompt` is the system prompt used when the request carries none of its own. A request always wins.

### `[template]`

`jinja` is a chat template overriding the GGUF's own, for text and vision models alike. Rendered by real Jinja2.

---

## Example configurations

### Minimal

```bash
export TLLAMA_HOST=127.0.0.1:54800
export TLLAMA_MODELS=/var/lib/tllama/models

tllama
```

### Lower memory

```bash
export TLLAMA_RUNTIME_FLASH_ATTN=true
export TLLAMA_KV_CACHE_TYPE=q4_0
export TLLAMA_CONTEXT_LENGTH=8192
export TLLAMA_MAX_LOADED_MODELS=1

tllama
```

Flash attention first: without it the cache type is ignored.

### Two GPUs, split by proportion

```bash
export TLLAMA_RUNTIME_TENSOR_SPLIT=0.7,0.3
export TLLAMA_RUNTIME_N_GPU_LAYERS=-1

tllama
```

### One GPU only, on a machine with several

```bash
export TLLAMA_RUNTIME_SPLIT_MODE=0
export TLLAMA_RUNTIME_MAIN_GPU=0

tllama
```

`main_gpu` indexes the device list *after* `llama.cpp` merges devices reporting the same PCI id, so one card visible under both CUDA and Vulkan counts once.

### CPU-heavy machine

```bash
export TLLAMA_RUNTIME_N_THREADS=16
export TLLAMA_RUNTIME_N_THREADS_BATCH=32
export TLLAMA_RUNTIME_USE_MLOCK=true

tllama
```

### LAN-accessible

```bash
export TLLAMA_HOST=0.0.0.0:54800

tllama
```

Use a reverse proxy and access controls beyond localhost.

---

## systemd

```bash
# /etc/default/tllama
TLLAMA_HOST=127.0.0.1:54800
TLLAMA_MODELS=/var/lib/tllama/models
TLLAMA_KEEP_ALIVE=5m
TLLAMA_MAX_LOADED_MODELS=1
TLLAMA_METADATA_CACHE_TTL=18000
TLLAMA_RUNTIME_FLASH_ATTN=true
TLLAMA_KV_CACHE_TYPE=q8_0
```

```ini
[Service]
EnvironmentFile=/etc/default/tllama
ExecStart=/usr/bin/tllama
```

A `systemctl edit tllama` drop-in with `Environment=` overrides the environment file. Both are the environment layer, so a model's `.toml` still wins over either.

A refused value exits 2 before the socket is opened, so `systemctl status` shows the message and the unit never comes up half-configured.

---

## Troubleshooting

### The server will not start

Read the last line: it names the variable and what was expected.

```
tllama: TLLAMA_HOST='0.0.0.0:http' is not valid: expected HOST:PORT with a numeric port.
```

### A model disappeared from the listing

Its `.toml` no longer parses, most likely because it carries a key TLlama will not act on. Nothing is deleted; the model is invisible until the file is fixed.

```bash
tllama rebuildrepo --dryrun
```

That names the file and the key, and writes nothing. A common case is a setting in the wrong section: context length is `n_ctx` in `[runtime]`, not `num_ctx` in `[sampling]`.

### A setting appears to do nothing

Check the layer above it. A model's `[runtime]` overrides `TLLAMA_RUNTIME_*`, and a request's `options` override `[sampling]`.

For a KV cache type, check flash attention: `llama.cpp` ignores a quantized cache without it, silently.

### A browser cannot reach the server

The browser console reports a CORS failure, or a plain network error with nothing in TLlama's log. The page's origin is not on the list; the startup log line beginning `Browser origins allowed:` says what is. Add it with `TLLAMA_ORIGINS`.

An origin is the scheme, host and port exactly as the page was loaded, so `http://localhost:3000` and `http://127.0.0.1:3000` are two different origins even though they are one machine. Both are allowed by default, on any port.

A browser caches a preflight for ten minutes, so a corrected origin list can take that long to take effect in a tab that is already open.

### The repository looks empty

```bash
echo "$TLLAMA_MODELS"
find "$TLLAMA_MODELS" -name '*.gguf' | head
tllama rebuildrepo --dryrun
```

A `.gguf` with no `.toml` is not a model. `rebuildrepo` writes the missing definitions.

### The first listing is slow

The persistent metadata cache is not built yet.

```bash
find "$TLLAMA_MODELS/.tllama/metadata-cache" -type f -name '*.json' | head
```

---

## Related

- `README.md`
- `docs/model-repositories.md`
- `docs/metadata-cache.md`
- `docs/api-compatibility.md`
- `docs/usage.md`
- `docs/development.md`
