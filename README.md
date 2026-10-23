# TLlama

TLlama is a Python server and client for local GGUF inference. It exposes
Ollama-compatible and OpenAI-compatible APIs and runs inference in private
`tllama-server` subprocesses based on llama.cpp. It uses FastAPI and Pydantic;
it does not require a Python inference binding.

The project is under active development. Supported API contracts, model modes
and explicit limitations are documented below.

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
the supplied [tllamad.service](systemd/tllamad.service) manually. Debian
application packaging follows separately.

## Client commands and REPL

```sh
tllama list
tllama show MODEL
tllama ps
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
| Embedded MTP | Opt-in model runtime settings; [MTP](docs/native-mtp.md) |

Embedding and Decision models have their own admission/capability contracts;
a generation endpoint does not automatically work on every GGUF architecture.
Image requests require the configured projector and live runtime support.
Audio APIs, full Ollama parity, Responses API state and REPL conversation
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
vocabulary. TLlama-owned settings are explicit contracts; new upstream options
are not automatically public API features.

## License

[MIT](https://opensource.org/licenses/MIT)
