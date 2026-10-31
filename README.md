# TLlama

TLlama is a Python server and client for local GGUF inference. It exposes
Ollama-compatible and OpenAI-compatible APIs and runs inference in private
`tllama-server` subprocesses based on llama.cpp. It uses FastAPI and Pydantic;
it does not require a Python inference binding.

TLlama combines an Ollama-style model service with direct control over the
native runtime. One daemon can keep model-specific device placement, cache
precision, context, sampling and speculative decoding settings, alongside
generation, vision, embedding and Decision models.

## At a glance

- **Configurable client, daemon and model profiles:** layer system/user client
  TOML preferences and override them from the CLI; inspect daemon CLI/env/TOML
  defaults and keep native load choices in each model profile. Client switches
  cover the remote server, HTTP timeouts, JSON output, request options, units,
  reasoning display and statistics. See the [configuration overview](docs/features.md#inspectable-configuration-and-model-profiles).
- **Independent K and V cache precision per model:** combine, for example,
  `type_k = "q8_0"` and `type_v = "q4_0"`, with global defaults and explicit
  model overrides. This goes beyond Ollama's single global K/V type.
- **Choose the device, including its backend:** assign one model to `ROCm0`,
  another to `Vulkan0` or `CUDA0`, and another to CPU. A main chat model can use
  CUDA/ROCm while auxiliary Decision/embedding models use Vulkan, on another
  card or a second backend view of the same card with shared physical memory;
  see the [placement example](docs/features.md#main-model-and-auxiliary-models-on-different-backends).
  Explicit comma-separated selectors also reach the native multi-device path,
  including mixed backends
  where the model and installed runtime support them.
- **Extended AMD support:** the development runtime includes native legacy
  Vega targets `gfx900`, `gfx906` and `gfx90c`, plus architecture-specific
  `gfx1250` fixes for Instinct MI455X. The hardware guide separates these
  extensions from completed inference qualification. Legacy HIP usually needs
  mmap disabled, or a Vulkan device instead; see the
  [workaround](docs/hardware.md#legacy-hip-mmap-workaround).
- **Experimental Instella-MoE:** the patched native runtime adds the converter,
  loader and model graph for `InstellaMoEForCausalLM`, initially
  Instella-MoE-16B-A3B-Think, beyond the reviewed official llama.cpp/Ollama sources.
- **A choice of native backends:** CPU and OpenBLAS, Vulkan, ROCm/HIP, CUDA
  12.4/12.9/13.3 and Intel SYCL have Debian runtime packaging. Other native
  backends depend on a suitable build; Windows/macOS builds and installers
  are planned.
- **More than chat:** compatible Ollama/OpenAI generation and tools, image
  input, semantic embeddings, text Decision models, FIM and opt-in embedded
  MTP use the same model lifecycle. Each has its own documented support scope.
- **Explainable memory reports:** native buffer inventory, deduplicated weight
  payload, CPU/GPU assignment and allocation locality remain separate. Unknown
  data stays unknown; `ps` does not guess placement from the GGUF file size.
- **Inspectable, reusable model definitions:** human-editable TOML profiles can
  share GGUF weights while keeping different templates, system prompts and
  runtime settings. Persistent metadata/digest caches support moved repositories.
- **Managed native processes:** immutable worker generations, protected active
  requests, idle TTL, cancellation and awaited cleanup keep model changes and
  failures local to their owned worker. No Python inference binding is required.

Start with the [documentation index](TOC.md), the detailed
[features and comparison](docs/features.md), [hardware support](docs/hardware.md)
and [packages and platform roadmap](docs/distribution.md).
TLlama 1.3.0 adds client image attachments for vision chat, image-aware
conversation saves and configurable image/session size limits. The daemon's
root status text also supports JetBrains AI Assistant's Ollama connection
check. See the [CLI guide](docs/cli.md#attach-images) and [changelog](CHANGELOG.md)
for release changes and upgrade notes.
These pages distinguish implemented features, tested configurations and planned
distribution work.

## Install and start

Install Python dependencies from a checkout:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[huggingface]'
```

The `huggingface` extra enables model pulling; local models do not require it.
Install the separate [tllama-server](https://github.com/ByCzech/tllama-server)
native runtime and backend libraries appropriate for the host. The default
executable is `/usr/lib/tllama/server/tllama-server`; select another absolute
path with `--runtime-path` or server TOML. Installing TLlama does not build
native backends or download weights.

```sh
tllamad --models-dir ./models
# Another terminal:
tllama list
tllama run MODEL
tllama run MODEL "Explain GGUF in two sentences."
```

Bare `tllama` shows client help. `tllamad` starts the foreground server.
From a checkout, use `python -m tllama` and `python -m tllama.daemon`.
The default listener is `127.0.0.1:54800`.

## Inference from Python

Install the optional client library with `python -m pip install ollama`, start
`tllamad` and replace `MODEL` with a registered name from `tllama list`:

```python
import ollama as tllama

client = tllama.Client(host="http://127.0.0.1:54800", timeout=300.0)
response = client.chat(
    model="MODEL",
    messages=[{"role": "user", "content": "Explain GGUF in one sentence."}],
    stream=False,
)
print(response.message.content)
```

The alias names the compatible client; inference runs in TLlama. No separate
Ollama daemon is needed. The explicit host selects TLlama's port.

See [client examples](docs/client-examples.md) for:

- **Python `ollama`:** basic chat, streaming, conversation history and async calls.
- **Python `openai`:** Chat Completions, async streaming and embeddings.
- **Python `requests`, `httpx` and `httpx2`:** direct JSON requests, streaming,
  timeouts and error handling.
- **curl:** model listing, chat, generation and both streaming wire formats.
- **JavaScript:** the `ollama` and `openai` libraries, plus dependency-free `fetch`.

## Configure the daemon

```sh
tllamad --config examples/server.toml --check-config
tllamad --config examples/server.toml --print-config
tllamad --config examples/server.toml --models-dir ./models --device Vulkan0
```

Global precedence is **CLI > environment > server TOML > defaults**.
Model definitions and supported request overrides retain their own stronger
precedence. A global device setting is a default for models.

File selection, all flags, environment names, source reporting and reload
behavior are described in [daemon.md](docs/daemon.md). The server configuration
is separate from per-model TOML. See [systemd.md](docs/systemd.md) for installing
the supplied [tllamad.service](systemd/tllamad.service) manually. Debian packages
provide the application and the separate modular native runtime; installation
and planned Linux/Windows/macOS distribution are described in
[distribution.md](docs/distribution.md).

## Client commands and REPL

```sh
tllama list
tllama show MODEL
tllama ps
tllama ps --detailed --json
tllama pull namespace/repository/model-file
tllama cp SOURCE DESTINATION
tllama create MODEL -f model.toml
tllama run MODEL --verbose
tllama stop MODEL
tllama rm MODEL
tllama --host http://server:54800 list --json
```

Interactive `run` uses prompt_toolkit, preloads the model with a spinner,
separates reasoning from the final answer and supports session slash commands.
One-shot prompts and stdin are also supported. Output/JSON stays on stdout;
progress, reasoning and optional timings go to stderr. See
[cli.md](docs/cli.md) for exact dispatch, options, commands and cancellation.
REPL inputs persist privately per server/model by default (`--no-history`
disables persistence). `/save NAME` and `/load NAME` explicitly preserve and
restore conversations and settings. Terminal bells are opt-in through
`[notifications]` or `--bell`/`--no-bell`.

To inspect devices locally without starting the daemon or loading a model:

```sh
tllamad --list-devices
```

Client preferences live in a separate `client.toml`, for example:

```toml
# ~/.config/tllama/client.toml
[client]
host = "http://inference-server:54800"
# timeout = 120

[display]
decimal_units = false
hide_thinking = false
verbose = true
```

Built-in defaults are overlaid by `/etc/tllama/client.toml`, then the user file
under `$XDG_CONFIG_HOME/tllama` (normally `~/.config/tllama`). Only explicitly
set values override earlier values. `--config PATH` instead selects one file
over built-in defaults. CLI flags override the file and supported environment
values; for the server address, `--host` overrides `TLLAMA_HOST`.

```sh
tllama --host http://other-server:54800 --timeout 120 ps --json
tllama --config examples/client.toml ps --no-decimal-units
tllama run MODEL --option num_ctx=32768 --option temperature=0.7 --no-verbose
```

For an interactive session, `tllama run MODEL --option num_ctx=32768` sends the
context setting in the initial preload, before the REPL opens. This avoids
loading with the default context and then reloading solely because the user
changes it in the REPL. The option is retained for subsequent requests without
rewriting the model definition. `--option KEY=VALUE` is repeatable; each VALUE
uses JSON syntax, so numeric values can be written directly as above.

This gives the HTTP client direct controls as well as persistent preferences;
it does not change server defaults or model definitions. See the
[client configuration example](examples/client.toml) and [CLI guide](docs/cli.md).

## Models and imports

Only registered TOML definitions appear as models. Public names identify the
repository area:

| Public name | Definition location under the model store |
| --- | --- |
| `MODEL` | `Local/MODEL.toml` |
| `collection/MODEL` | `TLlama/collection/MODEL.toml` |
| `namespace/repository/MODEL` | `HuggingFace/namespace/repository/MODEL.toml` |

To upload a local GGUF from the client:

```toml
[llm]
from = "./model.gguf"
```

```sh
tllama create my-model -f model.toml
```

Relative `from` paths use the client's **CWD**, including when the TOML is
elsewhere. Model and optional projector files are uploaded into the server's
`Local` storage. Complete split-GGUF groups are supported. Without `from`, use
`model = "Local/existing.gguf"` to reference an existing repository file.
Creation writes the definition after validation; it does not load or convert
the model. Copying definitions shares weights. Default conflicts require
explicit `--force`. Removing a definition retains weight files.

For manually copied weights, `tllamad rebuildrepo --dryrun` previews missing
definitions. See [model-repositories.md](docs/model-repositories.md) and
[configuration.md](docs/configuration.md).

## APIs and model modes

| Workflow | Public endpoints / documentation |
| --- | --- |
| Generation and chat | `/api/generate`, `/api/chat`, `/v1/chat/completions` |
| Model listing/details/lifecycle | [API compatibility](docs/api-compatibility.md), [CLI](docs/cli.md) |
| Function tools | [Native tools](docs/native-tools.md); caller executes functions |
| Image input | [Native vision](docs/native-vision.md), [public API](docs/native-public-api.md) |
| Embeddings | `/api/embed`, `/api/embeddings`, `/v1/embeddings`; [embedding API](docs/embedding-api.md) |
| Decision models | `/v1/systemone`; [Decision API](docs/decision-api.md) |
| Fill-in-the-middle | `/api/generate` with nonempty `suffix`; [FIM](docs/native-fim.md) |
| Embedded MTP | Opt-in model runtime settings; [MTP](docs/native-mtp.md) |

Embedding and Decision models have their own admission/capability contracts;
a generation endpoint does not automatically work on every GGUF architecture.
Image requests require the configured projector and live runtime support.
Audio APIs, `/v1/responses`, `/v1/completions`, full Ollama parity and REPL conversation
persistence remain separate work. Native TOML create is not an Ollama
Modelfile parser or a full implementation of Ollama's structured create body.

Third-party clients can use:

```sh
OLLAMA_HOST=127.0.0.1:54800 ollama list
```

OpenAI-compatible clients use `http://127.0.0.1:54800/v1`. A placeholder API
key may satisfy SDK requirements; TLlama does not turn that placeholder into
public API authentication. See [usage.md](docs/usage.md) and
[request compatibility](docs/request-compatibility.md).

## Runtime, memory and caching

Each admitted worker generation owns its process and immutable load settings.
Changes drain old generations while existing leases finish. Backend selection
and device selectors follow [backend-priority.md](docs/backend-priority.md).
The available backends depend on the installed native runtime.

Model information uses offline GGUF headers and persistent fingerprinted caches,
without loading inference weights. Content digest caches remain separate from
header schema changes. A cold index may take time; subsequent valid cache hits
reuse the stored metadata and digests.

Memory reporting preserves native provenance and unknown values; logical tensor
sizes are not automatically physical RAM/VRAM measurements. See
[native-memory.md](docs/native-memory.md) and [metadata-cache.md](docs/metadata-cache.md).

## Development

```sh
pip install -e '.[test,huggingface]'
pytest -q
```

See [development.md](docs/development.md) for focused checks and hardware smoke
scripts, and [native-config.md](docs/native-config.md) for the supported runtime
vocabulary. The [documentation review](docs/release-review-100.md) records the pre-1.0.0
checks and remaining release decisions. TLlama-owned settings are explicit contracts; new upstream options
are not automatically public API features.

## License

[MIT](https://opensource.org/licenses/MIT)
