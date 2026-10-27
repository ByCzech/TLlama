# Development

TLlama's Python application manages private native `tllama-server` workers.
The native runtime is developed and packaged separately; a Python
`llama-cpp-python` build is not required.

## Checkout and dependencies

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,huggingface]'
python -m compileall -q src
pytest -q
```

The `huggingface` extra is optional outside pull tests/workflows. The client
REPL uses prompt_toolkit 3.0.51 or newer. Use a pytest-asyncio version compatible
with the installed pytest/Python, especially when treating resource warnings
as errors. Test-only native worker fakes let most tests run without GPU
hardware or real weights.

## Local daemon

```sh
python -m tllama.daemon --no-config --models-dir ./models \
  --runtime-path /absolute/path/to/tllama-server --log-level debug
python -m tllama list
python -m tllama run MODEL --verbose
```

Or select a server file with `--config PATH`; inspect with `--check-config`
and `--print-config` before starting. Relative paths use CWD. Details are in
[daemon.md](daemon.md). Development `--reload` replays the configuration in
replacement processes; it is not live model/config hot-reloading.

## Focused verification

For daemon configuration, startup and logging:

```sh
pytest -W error -q tests/test_daemon_config.py tests/test_daemon_cli.py \
  tests/test_cli_entry_points.py tests/test_config_from_env.py \
  tests/test_server_logging.py tests/test_cors.py
```

The daemon subprocess tests exercise normal and Uvicorn reload-process startup
with a separate model store, source precedence, CORS, and shutdown. Inspection
tests block imports of application/inference dependencies and assert no storage
creation. Run the full suite after changes affecting shared configuration or
stream ownership.

For imports and model management:

```sh
pytest -W error -q tests/test_model_create.py tests/test_model_copy.py \
  tests/test_model_uploads.py tests/test_client_mutations.py
```

Use the supplied [systemd instructions](systemd.md) to verify the unit on a real
host. `systemd-analyze verify` checks syntax/dependencies when the referenced
executable is installed; it does not exercise service permissions or GPU access.

## Models and caches

A copied GGUF needs a TOML definition before it can be listed/loaded.
Prefer the client import workflow:

```toml
[llm]
from = "./model.gguf"
```

```sh
tllama create my-model -f model.toml
tllama show my-model
tllama run my-model "Say hello."
```

For an existing host-side repository, preview `tllamad rebuildrepo --dryrun`
and then run without `--dryrun` to add missing definitions. Rebuild force/backup
semantics are described in [model-repositories.md](model-repositories.md).

To evaluate metadata caching, compare the first and repeated `tllama list`,
then repeat after a daemon restart. Use an isolated development model store
for destructive cache experiments. Replacing/touching a GGUF changes its
fingerprint and invalidates cached metadata/digests; a definition-only change
does not require rehashing shared weight bytes.

## Native and hardware smoke tests

The scripts in `scripts/` complement unit tests. Read each script's help and
adjacent Markdown before running; they need explicit runtime/model paths and
some consume substantial device memory. Use independent evidence directories.

| Area | Entry points |
| --- | --- |
| Worker configuration | `worker-config.py`, `worker-smoke.py` |
| Lifecycle and memory | `manager-smoke.py`, `memory-smoke.py` |
| Public API and tools | `api-smoke.py`, `tools-api-smoke.py` |
| Vision and embeddings | `vision-smoke.py`, `embeddings-api-smoke.py` |
| Decision heads | `decision-api-smoke.py` |
| MTP | `mtp-smoke.py` |
| Devices | `device-inventory.py` |

CPU/Vulkan tests and evidence do not certify CUDA/ROCm/Metal on untested hosts.
When comparing behavior with upstream native inference, retain the actual
runtime fingerprint, model definition, request, output and hardware details.
Keep upstream behavior such as `num_predict=0` unchanged pending the agreed
upstream report.

## Code and patch workflow

Use small, independently reviewable Conventional Commit changes (`fix:`,
`feat:`, `test:`, `docs:`). Keep code comments and commit messages in English.
Keep configuration, request validation, native serialization and response
ownership contracts explicit. Log through module loggers; let entry points
configure handlers. Do not execute client/model tool calls.

Daemon configuration and systemd integration are implemented. Before release,
apply the [documentation review](release-review-100.md) corrections, resolve
the recorded release decisions, then merge/tag the verified refactor. Debian application
packaging is planned afterward in a separate branch. Personal planning/handoff
Markdown lives outside the Git repository.
