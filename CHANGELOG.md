# Changelog

User-visible release changes summarized from the Git history. Release tags
identify the source; the complete history retains tests, refactoring and
intermediate development steps. TLlama application versions
are independent of the separately packaged `tllama-server` native runtime.

## 1.2.0 — Unreleased

### Added

- REPL input history persists by default, privately scoped by server/model
  under the XDG state directory. Configure `[history] enabled`, override with
  `run --history`/`--no-history`, or clear saved inputs with `--clear-history`.
- Explicit REPL `/save`, `/load`, `/sessions` and `/forget` commands preserve
  conversations and settings in private, server/model-scoped JSON saves.

## 1.1.0

### Added

- Client TOML configuration: built-in defaults are overlaid by
  `/etc/tllama/client.toml`, then `$XDG_CONFIG_HOME/tllama/client.toml`
  (normally `~/.config/tllama/client.toml`). Values merge by key; the last
  explicitly configured value wins. `--config PATH` selects one file over
  built-in defaults instead. CLI flags and supported environment values keep
  their higher precedence. The `[client]` section configures the server
  address and optional HTTP timeout. See the [CLI guide](docs/cli.md).
- Persistent display preferences in `[display]`: `decimal_units`, `verbose`
  and `hide_thinking`, with CLI overrides including `--no-verbose` and
  `--show-thinking`. Reasoning display affects terminal rendering, preserving
  messages and JSON output. See the [example](examples/client.toml) and
  [display settings](docs/cli.md).
- Native `[runtime] fit_target_mib` to configure the fitter's memory margin
  in MiB, and explicit `[runtime] fit` to enable or disable native fitting.
  Both accept global defaults and model overrides. `n_gpu_layers = -1`
  together with `fit = false` requests `--n-gpu-layers all --fit off`.
  Existing automatic/explicit-device defaults are preserved when `fit` is
  unset; CPU placement disables it. See the
  [configuration reference](docs/native-config.md).
- Debian client/server configuration examples and this upstream changelog.

### Changed

- Human-readable client sizes now default to binary KiB/MiB/GiB/TiB units.
  `[display] decimal_units = true` or `--decimal-units` restores decimal
  kB/MB/GB/TB units. API and JSON byte counts are unchanged. See the
  [size display settings](docs/cli.md).
- `run` help spells the repeatable request option as `--option KEY=VALUE`;
  VALUE still uses JSON syntax. Examples show `--option num_ctx=32768`
  setting the context in the initial preload and retaining it for later
  prompts. This documents existing preload behavior; parsing is unchanged.
  See the [run examples](docs/cli.md).
- Documented client/runtime configurability and main/Decision/embedding model
  placement across native and Vulkan backends, including two backend views
  of one physical GPU. Such workers share physical VRAM; this does not create
  extra device capacity or a coordinated memory budget. See the
  [placement examples](docs/features.md#main-model-and-auxiliary-models-on-different-backends).

### Fixed

- Metadata and digest caches use cache-file modification times to validate
  the model snapshot. Writers set the cache-file timestamp to the captured
  model timestamp before atomic publication; readers compare file timestamps
  along with relative path, size and schema. The JSON `model.mtime_ns` field
  and its validation code have been removed. See the
  [cache guide](docs/metadata-cache.md).

### Upgrade notes

- Cache format 2 deliberately rejects older metadata/digest cache files.
  The first access regenerates them, including SHA-256 where required, so
  an initial listing may take longer. Model weights and definitions are
  unchanged; there is no migration or old-format fallback.
- Use decimal display preferences if existing workflows expect GB rather
  than GiB. Machine-readable byte counts need no conversion changes.
- No client configuration file is required. Creating one only changes the
  explicitly configured preferences. CLI configuration loading does not
  change the Python HTTP client library's behavior.
- Fit controls change native launch policy; they do not repair native fitter
  bugs or guarantee that a model/context fits. Disabling fit can result in an
  allocation failure when the requested configuration exceeds device memory.

Source baseline: [v1.0.0](https://github.com/ByCzech/TLlama/tree/v1.0.0).
After publication, the complete release range is available as
[v1.0.0...v1.1.0](https://github.com/ByCzech/TLlama/compare/v1.0.0...v1.1.0).

## 1.0.0 — 2026-10-27

First stable application release, with a Python wheel/source archive and
Debian client/daemon package.

- Private native `tllama-server` workers replace the Python inference binding,
  with authenticated local IPC, owned worker lifetimes, cancellation and
  model-specific runtime settings.
- Ollama-compatible and OpenAI-compatible interfaces cover generation/chat,
  tools, vision, embeddings and model management; FIM, Decision and opt-in
  embedded MTP have their documented model/runtime qualification scopes.
- A standalone HTTP client provides inspection and lifecycle commands,
  streaming prompts, an interactive REPL, initial preload, reasoning display,
  localized tables and optional request statistics.
- TOML model definitions support shared weights, model-specific devices,
  separate K/V precision and context/runtime/sampling settings. Client-side
  imports upload and verify GGUF blobs; repository caches use relative model
  identities across repository mount points.
- Native memory reports distinguish buffer allocations, occupied weight
  payload and CPU/GPU placement, with explicit limits on unknown locality.
- Daemon CLI/environment/TOML configuration, the `tllamad` systemd service
  and Debian packaging provide managed Linux installation.

See the [release commit](https://github.com/ByCzech/TLlama/commit/2432a5dc6e2a71d8e738a51aef92fecf089252d5),
[release review](docs/release-review-100.md) and
[runtime qualification](docs/runtime-qualification-070.md) for historical
scope and evidence. Persistent REPL history and additional platform installers
remain future work.
