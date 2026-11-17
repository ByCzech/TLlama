# Changelog

User-visible release changes summarized from the Git history. Release tags
identify the source; the complete history retains tests, refactoring and
intermediate development steps. TLlama application versions
are independent of the separately packaged `tllama-server` native runtime.

## 1.6.0 — 2026-11-17

### Added

- External MTP helpers in the existing `[draft]` slot, including whole-group
  recognition, ordinary pull/create import, manual-role preservation and
  active draft placement/KV/batch tuning. A selected external MTP head takes
  precedence over the integrated one; load failures do not silently switch heads.
- Optional per-device residency limits through `[[lifecycle.device]]` and
  `TLLAMA_DEVICE_LIMITS`. Named devices include `CPU`; an unnamed entry sets the
  default. Zero means unlimited. A nonempty device list implies global zero
  when omitted; an explicit positive global cap conflicts and prevents startup.
- Offline `/api/show` renderer/parser/template provenance and optional qualified
  thinking values/default. Unknown capabilities are omitted and absent templates
  stay empty. Embedding, Decision and standalone helpers omit chat diagnostics.
- Local repository/startup profiling with separate duration and operation-count
  measurements, without starting native workers or forcing cache invalidation.

### Changed

- Unify configured repository roots and repository-relative internal references;
  avoid repeated TOML parsing, shard/group/header inspection and listing I/O.
  Cancellation waits for owned filesystem work before returning. Measured
  operation reductions do not promise a fixed speedup on every repository.
- Resolve meaningful `reasoning.effort` before `reasoning_effort`; absent,
  null and blank values fall through. Conflicting explicit choices warn.
  Exactly qualified template profiles use known aliases/defaults and a warning
  for unsupported levels; unknown/custom templates remain passthrough.
- Clarify create/pull compatibility and error messages: TLlama TOML/import
  and HF GGUF references are distinct from Ollama create/registry contracts.
  Existing status codes and successful request formats are preserved.
- Keep documentation, Debian packaging and test fixtures in Python source
  archives built without Git metadata; extend the smoke against the built wheel.

### Upgrade notes

- OpenAI-compatible chat replies now return assistant `reasoning`, including
  streamed `delta.reasoning`, matching Ollama. Consumers reading the previous
  `reasoning_content` response field must update. It remains an input-only alias;
  unequal non-null aliases fail with HTTP 400 before admission. These reasoning
  text fields are extensions to OpenAI Chat Completions. Ollama APIs retain
  `thinking`; private native wire fields are unchanged.
- Existing global-only residency configurations retain their behavior. When
  enabling per-device mode, remove the explicit positive global cap or set it
  to zero. Every device entry requires `max_loaded_models`; configured devices
  absent from complete discovery warn rather than blocking startup. See the
  configuration examples.
- No repository conversion, cache reset, reimport or model-identity change is
  required. Repository format 2, metadata 9, digest 2 and cache envelope 2 remain
  unchanged. Existing valid content digests are reused.
- The separately versioned native runtime remains `b12359-c0819bbbae` with
  Debian floor `1.0.0~dev+b12359.1-1`. Sampling, KV, mmap and default speculative
  settings are unchanged. Attaching a draft does not activate speculation.

See the [1.6.0 release review](docs/release-review-160.md) for the selected
scope, qualification boundaries and final integration checkpoints.

## 1.5.1 — 2026-11-16

### Changed

- Adopt the separately built native `tllama-server` 1.0.0 runtime,
  `b12359-c0819bbbae`, including the project's Memory API V1 changes.
  Retain exact generation/embedding, text Decision and discovery admission
  policies and previously supported identities. Diagnostic pins remain exact.
- Require `tllama-server >= 1.0.0~dev+b12359.1-1` in Debian so the application
  and reviewed native update are delivered together. Native core and backend
  modules still require matching versions of their own package set.

### Fixed

- Do not report a complete legacy load-log total when additional layer-input
  or NextN buffer allocations are observed. Native Memory API V1 remains
  authoritative; admitting a new worker does not certify its log-only coverage.

No repository conversion, cache reset, reimport or content-hash change is
required. Sampling, KV, mmap, placement, speculation and public HTTP contracts
are unchanged. External MTP management, repo/startup cleanup and per-device
limits remain later work. See [the release review](docs/release-review-151.md)
for the distinction between source/automated checks and actual hardware smoke.

## 1.5.0 — 2026-11-13

### Added

- Standalone projector and draft model definitions. Pull/create register a
  helper TOML when importing it independently; composite local imports keep
  direct helper references in the main definition without creating extra TOMLs.
  Helpers share content-addressed weights and optional original-name companions.
- `tllama list --all` and `--type main|mmproj|draft`, typed helper details in
  show, and `cp SOURCE DEST --mmproj NAME --draft NAME`. Plain list and OpenAI
  model listings retain runnable main models. Helpers cannot run independently.
- Explicit `rm --remove-unused-helpers` for attached standalone helpers left
  without other main users. Default deletion reports and retains their TOMLs;
  unreferenced component weights without their own definition are reclaimed
  automatically, subject to shared references and live worker reservations.
- External DFlash/DFlash2 drafts, one draft reference and one selected strategy
  per definition. `none`, embedded `draft-mtp` and external `draft-dflash` are
  explicit choices; attaching a draft does not activate it. DFlash2 uses the
  same native strategy with its model-specific format handled by the loader.
- Typed draft device, GPU-layer, K/V precision and physical-batch settings;
  per-request Ollama `options.spec_type` and REPL `/set parameter spec_type`.
  Measured drafted/accepted counts and acceptance are shown by `run --verbose`
  when provided by the native worker. Diagnostic scripts capture off/on and
  cache/cancellation evidence for review.
- Local offline `tllamad repo prune-uploads`, preview by default and fresh
  checks with `--apply`, to discard abandoned upload staging and its caches.
  It requires stopped daemons/writers and does not delete published weights
  or definitions. No HTTP maintenance endpoint or automatic upload TTL is added.

### Changed

- Interactive pull progress updates one terminal line with a spinner and,
  when the total is known and space permits, a progress bar. Redirected/dumb
  terminals retain readable lines; JSON preserves the original NDJSON stream.
- Create/blob reuse verifies already published complete groups. Successful
  creation removes redundant upload aliases instead of retaining a second
  permanent weight location. Existing local digest/header caches are reused
  without needless rewrites or per-shard repeated group scans.
- Inactive external drafts are omitted from worker load/fitting arguments;
  changing the effective strategy creates the appropriate worker generation.
  Explicit request `draft_num_predict=0` disables speculation. Positive counts
  adjust the selected method, retaining the legacy MTP fallback only when no
  explicit request strategy was supplied. No automatic strategy tuning is added.

### Fixed

- Repeated pull through a configured repository-root symlink reuses published
  weights without mixing configured and physical roots in relative paths.
- RAM metadata cache keys distinguish full file fingerprints, including
  different canonical weight paths with equal size/time and nanosecond changes.
  Metadata filesystem/GGUF work runs outside the event loop with awaited
  completion on cancellation. Removed an obsolete, nonfunctional cache helper.
- Optional draft header-read failures follow the existing target/native-loader
  admission boundary; known incompatible architectures and invalid shard sets
  still fail explicitly. Unverified draft allocation in fallback logs remains
  unknown instead of producing falsely precise memory totals.

### Upgrade notes

- No repository-format conversion, cache schema bump or deliberate cache
  invalidation is required from 1.4.1. Do not delete valid digest caches merely
  to upgrade. Older unconverted repositories still need the offline converter.
- The agreed helper-management API additions advertise `repository_helpers`
  through `server_features`; model details use `model_definition`. New client helper
  operations check daemon support. No new HTTP routes or OpenAPI UI are added.
  See [API compatibility](docs/api-compatibility.md).
- Existing independently downloaded projectors/drafts can gain a helper
  definition by explicit repeated pull. Bulk repair of old origins or wrongly
  classified definitions is not automatic; inspect old TOMLs before removal.
- Restart the upgraded daemon. This application release does not rebuild or
  bundle `tllama-server`; the captured DFlash contract uses native
  `b11850-0e9433bfd3`. GPU/model/backend qualification remains bounded by the
  evidence in [the release review](docs/release-review-150.md). DFlash is opt-in
  and can be slower than ordinary decoding; acceleration is not guaranteed.
- Repository-root unification and further warm-cache startup optimization are
  deferred to a later cleanup release; they are not part of this update.

## 1.4.1 — 2026-11-01

### Fixed

- Record original GGUF basenames in optional UTF-8 `.inf` companions, one
  literal name per line and one companion per shard. Create, pull and offline
  conversion retain additional names when reusing the same immutable weights.
  Content-addressed directories, generic GGUF names, model IDs and TOML paths
  remain unchanged; equal content continues to share a single stored group.
- Accept corresponding regular companions during publication, reference
  validation and local checks. Missing companions are normal, with no error or
  warning. Content hashing and weight sizes exclude them; physical reclamation
  and local prune remove them along with their unreferenced weight group.
- Identify the actual failing GGUF path in conversion errors, including orphan
  files and continuation shards. Progress is printed before inspecting orphans.

No repository-format conversion is needed when upgrading from 1.4.0. Existing
companions created by the standalone journal backfill tool are accepted. The
converter can resume old unfinished journals and recover names from their
recorded source paths. HTTP routes and request/response schemas are unchanged.

## 1.4.0 — 2026-11-01

### Added

- Managed format-2 model repositories: portable selectively percent-encoded
  definition names and immutable content-addressed `data/<sha256>/` groups.
  Complete GGUF shard sets stay together; projectors and LoRA adapters can be
  shared independently. Local additions use `create`, remote additions use
  `pull`, and copies share existing groups instead of duplicating weights.
- A resumable, one-time offline converter in `tools/`, outside the Python
  wheel. It previews by default, preserves TOML comments/settings and valid
  current caches, and supports repositories older than 1.3.1. Debian ships
  the converter in the optional `tllama-doc` package.
- Local `tllamad repo check`, `prune`, `inspect` and `stats` commands for
  integrity, explicit orphan cleanup, sharing and managed weight accounting.
  These commands do not start/contact HTTP or load native workers. Prune
  previews by default and requires `--apply` for a freshly checked deletion.

### Changed

- Removing or replacing a definition reclaims its unused managed groups,
  matching repository-owned upload hardlinks and path caches. Shared
  definitions, worker generations and publications are protected across
  cooperating daemons by OS-backed reservations and a repository lock.
  Deferred cleanup has a durable retry queue and checks current protection
  again before every attempt.
- The obsolete `rebuildrepo` command and automatic definition generation
  from arbitrary GGUF locations have been removed. Nonempty unmarked stores
  require explicit offline conversion; the daemon does not guess a format.
- Framework documentation routes `/docs`, `/redoc`, `/openapi.json` and
  `/docs/oauth2-redirect` are disabled.

### Fixed

- `tllama rm` accepts an empty successful response from an Ollama daemon,
  prints a compatibility warning on stderr and retains a successful status.
- Delete, show and pull validate request bodies and field types before
  accessing models or the network. Invalid bodies use the existing HTTP 400
  Ollama error envelope instead of unhandled type errors. Existing aliases,
  valid requests and response formats remain supported.
- Public subprocess smoke fixtures register current repository definitions
  and handle concurrent lifecycle events without premature cleanup.

### Upgrade notes

- Back up existing repositories and stop **all daemons and writers sharing
  them**, including on other hosts. Convert using the matching new source
  or installed TLlama: first run the read-only preview, then `--apply`.
  See [conversion instructions](tools/README.md). Keep writers stopped after
  an interruption and use the documented `--resume` procedure.
- Upgrade/restart every daemon sharing a format-2 store before physical
  reclamation is used. Old daemons and external readers bypassing the
  reservation protocol cannot be detected. The shared filesystem must honor
  OS locks across participating hosts. Review service-account ownership/ACLs
  after conversion; package upgrade does not migrate the repository for you.
- Empty stores initialize automatically. Old/unknown caches are regenerable;
  conversion can reuse independently valid current caches. Renaming paths
  alone does not change model identity. No native runtime upgrade is implied
  by this Python application release.
- Repository commands are local daemon commands with normal storage/config
  precedence; `--models-dir` is optional. Whole-store prune does not sweep
  independent cache entries or old staging/download jobs.

Source baseline: [v1.3.1](https://github.com/ByCzech/TLlama/tree/v1.3.1).
The complete release range is
[v1.3.1...v1.4.0](https://github.com/ByCzech/TLlama/compare/v1.3.1...v1.4.0).

## 1.3.1 — 2026-10-31

### Fixed

- Public text Decision cancellation keeps its stop intent active through
  AnyIO/HTTPX transport checkpoints so an absorbed single cancellation
  cannot retain a model lease. Decision smoke reports pending tasks and
  residency explicitly if cancellation fails to drain within its deadline.
- Model listing sizes include every main-model and projector GGUF shard and
  any configured LoRA file. Shared files count once within each definition;
  incomplete shard sets do not produce a misleading partial size.

### Changed

- Public model IDs now combine the main GGUF, any projector and only the
  model TOML parameters differing from TLlama/GGUF defaults. An unchanged
  single-file LLM retains its original content SHA-256. Names, paths,
  server/environment settings and request overrides do not enter the ID.
  Running generations retain the ID captured before loading.
- Metadata cache schema 9 includes GGUF RoPE defaults for ID comparison;
  digest cache schema 2 verifies that files stay unchanged while hashing.
  Existing caches regenerate automatically; the first listing or model
  admission can therefore take longer while file SHA-256 values are rebuilt.

### Upgrade notes

- Model IDs for definitions with projectors or non-default TOML parameters
  can change after upgrading. An unchanged single-file LLM keeps its SHA-256;
  aliases and the same definition on different servers keep the same ID.
- Metadata cache 8 and digest cache 1 are regenerated as schemas 9 and 2.
  The first listing or load can take longer while file hashes are rebuilt.
  No manual cache removal is required.
- Restart the upgraded daemon to use the corrected reporting and Decision
  cancellation behavior. The native runtime version is unchanged.

Source baseline: [v1.3.0](https://github.com/ByCzech/TLlama/tree/v1.3.0).
The complete release range is
[v1.3.0...v1.3.1](https://github.com/ByCzech/TLlama/compare/v1.3.0...v1.3.1).

## 1.3.0 — 2026-10-31

### Added

- `tllama run --image FILE` and REPL `/image`, `/images`, `/unimage` attach
  ordered client-local images to vision chat prompts. Completed image turns
  persist in explicit conversation saves; failures/cancellation retain queued
  attachments for retry. Existing save-size limits include encoded images.
- Saved conversations now default to a 256 MiB complete JSON limit,
  configurable with `[sessions] max_file_mib` in the client TOML. The same
  limit applies to save and load. Individual images default to 16 MiB before
  base64 encoding, configurable through `[images] max_file_mib`; image saves
  use the same per-image limit without restoring it from the saved session.

### Fixed

- The root URL now displays `TLlama compatible Ollama is running` as plain
  text, retaining the substring required by JetBrains AI Assistant's
  connection check. `HEAD /` remains a body-free HTTP 200 probe for Ollama
  clients.

### Upgrade notes

- Client image input requires a completion model advertising vision and its
  configured projector. Upgrade the client to use `run --image` or the REPL
  image commands; the existing daemon image API remains unchanged.
- Saved conversations include image bytes as base64 and can be restored
  without the original files. Older clients reject image-bearing saves and
  saves exceeding their 16 MiB complete-file limit. The new default is
  256 MiB for the complete JSON and 16 MiB per image before encoding.
  Configure `[sessions] max_file_mib` and `[images] max_file_mib` locally on
  each client that saves or loads larger conversations. Saved sessions do
  not change these limits; server and model limits still apply.
- Restart the upgraded daemon to expose the root status text required by
  JetBrains AI Assistant's Ollama connection check.

Source baseline: [v1.2.0](https://github.com/ByCzech/TLlama/tree/v1.2.0).
The complete release range is
[v1.2.0...v1.3.0](https://github.com/ByCzech/TLlama/compare/v1.2.0...v1.3.0).

## 1.2.0 — 2026-10-29

### Added

- REPL input history persists by default, privately scoped by server/model
  under the XDG state directory. Configure `[history] enabled`, override with
  `run --history`/`--no-history`, or clear saved inputs with `--clear-history`.
- Explicit REPL `/save`, `/load`, `/sessions` and `/forget` commands preserve
  conversations and settings in private, server/model-scoped JSON saves.
- `tllamad --list-devices` forwards the selected native runtime's device table
  without starting the application or loading a model.
- Opt-in terminal bell notifications for model readiness, completed responses,
  pulls/imports and errors. Configure `[notifications]` or override with
  `--bell`/`--no-bell`; the default is off and stdout/JSON stays unchanged.

### Changed

- Debian application packages install the commented client example as
  `/etc/tllama/client.toml`, with standard conffile upgrade handling.
- `/api/ps` now returns a compact summary by default. Request
  `?detailed=true` or use `tllama ps --detailed --json` for full memory and
  lifecycle diagnostics. Existing consumers of these extensions must opt in.
  Summary sizes are cached per native observation using the same accounting
  policy as detailed reports; polling does not copy full native inventories.

### Upgrade notes

- Upgrade and restart the daemon to use compact process reports and the
  `detailed` query parameter. Updating only the client leaves the server's
  response unchanged. Scripts reading memory or lifecycle diagnostics must
  request `/api/ps?detailed=true` explicitly.
- REPL input history is enabled by default. Use `run --no-history` or
  `[history] enabled = false` to disable persistence. Named conversation saves
  remain explicit; starting a new REPL does not automatically restore one.
- The installed client configuration contains only commented example values,
  so it preserves built-in defaults. User configuration, supported environment
  settings and CLI overrides retain their existing precedence.

Source baseline: [v1.1.0](https://github.com/ByCzech/TLlama/tree/v1.1.0).
The complete release range is
[v1.1.0...v1.2.0](https://github.com/ByCzech/TLlama/compare/v1.1.0...v1.2.0).

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
