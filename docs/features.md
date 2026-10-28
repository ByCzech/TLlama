# Features and practical advantages

TLlama provides a model service, client and configuration layer over a patched
llama.cpp runtime. Its main advantages are explicit per-model control, extended
native model/hardware support, configurable client/daemon interfaces and
inspectable runtime evidence.
See the [documentation index](../TOC.md) for the complete guide set.

## Independent K and V precision

Cache precision is a model setting rather than a server-wide compromise:

```toml
[llm]
model = "Local/model.gguf"

[runtime]
device = "Vulkan0"
n_ctx = 32768
flash_attn = true
type_k = "q8_0"
type_v = "q4_0"
```

This is a configuration example, not a quality recommendation for every model.
`type_kv` sets both sides; `type_k` and `type_v` override individual sides within
that layer. Model settings override global values independently. The native
parser supports f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0 and q5_1; the
actual architecture/backend must also support the chosen combination.

The baseline is f16/f16. Quantized V with flash attention off is rejected by
the pinned native runtime. Instella's tested recipe requires matching K/V types.
Smaller caches trade precision for memory; the observed FIM quality change
between q4_0 and f16 makes that trade-off concrete. See
[configuration](configuration.md) and [native settings](native-config.md).

llama.cpp already exposes separate K/V switches; TLlama makes them available
through daemon configuration and persistent model profiles. The reviewed Ollama
implementation passes one global type to both native switches.

## Model placement across devices and backends

Set `[runtime].device` in each model definition to the exact native selector.
For example, three definitions can select `ROCm0`, `Vulkan0` and `none`
respectively under one daemon, with different contexts and cache settings.
CUDA and SYCL use the same configuration path; there is no model/backend whitelist
in the selector validator. The installed runtime supplies the actual devices.

Three distinct capabilities matter:

| Scope | TLlama behavior |
| --- | --- |
| Different models, different backends | Each model resolves its own device and owns its own worker; capacity is controlled by `max_loaded_models` |
| One model, explicit device list | Comma-separated native selectors are passed unchanged; layer/row splitting and `tensor_split` remain native controls |
| Automatic placement | An ordered backend policy tries one backend family per attempt, then CPU; it does not create a mixed-backend group |

For a compatible ordinary generation model, a requested mixed-backend profile
can look like this:

```toml
[llm]
model = "Local/model.gguf"

[runtime]
device = "CUDA0,Vulkan0"
split_mode = 1
tensor_split = [1.0, 1.0]
```

Both selectors must exist and refer to the intended devices. One physical GPU
can appear through several backends: listing it twice does not create additional
VRAM. TLlama does not deduplicate physical devices or promise that every
cross-backend/model combination works. Mixed-backend execution remains a
hardware qualification item. Instella rejects native `--split-mode tensor`
(mode 3, not exposed by this TLlama configuration); this is distinct from the
`tensor_split` ratios used with layer/row modes. Its layer/row recipes remain
unqualified.

Explicit placement bypasses the automatic backend preference list, even if the
selected backend is absent from that list. A failed explicit placement is
reported; it is not silently moved to another device. `main_gpu` is a native
index within the selected setup, not a backend selector or persistent PCI ID.

With no explicit device, the default order is CUDA, ROCm, SYCL, Vulkan, then CPU.
Administrators can reorder it or select CPU only. Known selected-accelerator
startup failures can advance to the next candidate; arbitrary model, OOM and
protocol failures are not a reason to replay requests across backends.
See [backend priorities](backend-priority.md) and [device inventory](device-inventory.md).

## Native model and hardware extensions

- **Instella-MoE:** converter, GGUF architecture, loader and gated MLA/FarSkip
  graph are included experimentally in the native fork. The initial target is
  AMD Instella-MoE-16B-A3B-Think. The maintainer confirmed loading, inference and
  thinking with matching K/V types. This is not a claim about all Instella
  families, cache combinations or backends. The implementation credits Csaba
  Kecskemeti's community port; see [source evidence](feature-source-audit.md).
- **Legacy Vega on modern ROCm:** the development package set adds native
  gfx900/gfx906/gfx90c kernels and matching optional BLAS add-ons. gfx90c does
  not need to impersonate gfx900. The open legacy HIP mmap problem and its
  no-mmap workaround remain documented in [hardware](hardware.md).
- **Instinct MI455X/gfx1250:** the native fork distinguishes its instruction
  capabilities from RDNA4, avoiding unsupported dot2/WMMA paths. This is a source
  compatibility fix, without a local MI455X inference acceptance run.
- **Older and newer NVIDIA hardware:** versioned CUDA plugins retain a CUDA
  12.x route for Maxwell/Pascal/Volta while offering CUDA 13.3 for newer cards.
  P100/CUDA12.9 is a tested configuration, rather than an inferred compatibility
  claim. CUDA version selection and family mappings are in [hardware](hardware.md).
- **CPU and Intel GPU choices:** portable CPU variants and OpenBLAS are packaged;
  Intel SYCL includes oneMKL, oneDNN and Level Zero integration. Additional CPU
  libraries and architectures remain build-dependent.

These extensions belong partly to the native runtime and partly to packaging.
The Python service does not implement GPU kernels. Ordinary upstream model
support, including the accepted Nemotron 3.5 Bolt support, is credited to the
native upstream base rather than presented as a unique TLlama invention.

## Inspectable configuration and model profiles

TLlama brings persistent client preferences, inspectable service defaults and
per-model runtime profiles into one configuration workflow. The CLI exposes
both client behavior and supported inference controls directly:

| Scope | Persistent configuration | Direct controls |
| --- | --- | --- |
| Client connection and display | System then user `client.toml`; a user can replace individual settings | `--host`, `--timeout`, `--config`, `--json`, `--decimal-units` / `--no-decimal-units` |
| Run/REPL presentation and requests | `[display] verbose` and `hide_thinking`; request controls remain explicit | `--verbose` / `--no-verbose`, `--hide-thinking` / `--show-thinking`, `--option NAME=JSON`, `--system`, `--think`, `--format`, `--keep-alive` |
| Daemon | `server.toml`, environment and CLI overrides | Listener/storage/runtime selection, device/backend defaults, context/K/V settings, lifecycle and telemetry; `--check-config` and `--print-config` |
| Model runtime | Each model's `[runtime]`, sampling, template and system settings | Context, device placement, independent K/V precision, native fitting and safety margin, MTP and other typed load controls |

- Client files layer built-in defaults, `/etc/tllama/client.toml`, then
  `$XDG_CONFIG_HOME/tllama/client.toml` (normally `~/.config/tllama/client.toml`).
  A user file can change only display preferences while retaining the system
  server address, or explicitly replace that address. `--config PATH` reads only
  the selected file over built-in defaults. CLI flags override supported
  environment settings and file defaults. Binary sizes are the default; JSON
  continues to carry the original byte values.
- Daemon defaults resolve CLI > environment > server TOML > built-in defaults.
  Per-model definitions and supported request overrides provide stronger scoped
  settings. `--check-config` validates configuration before serving;
  `--print-config` reports resolved values and their sources.
- Load controls include context, batches, CPU threads, GPU layers, split mode,
  native `fit` and `fit_target_mib`, RoPE/YaRN, NUMA, mmap/mlock,
  operation/KV offload, LoRA and typed GGUF metadata
  overrides. Templates, system prompts, sampling and FIM ordering are also
  explicit model settings; no arbitrary native command-line passthrough is needed.
- Different public model names can share the same GGUF/projector files while
  using different settings. `cp` copies a definition, not another copy of the
  weights. This saves disk space; it does not promise shared in-memory workers
  across different public names.
- TOML rewrites preserve comments and formatting. Repository rebuilding has a
  dry-run mode and protects existing backup files when forced regeneration is
  requested.
- GGUF sampling recommendations are used where explicit settings are absent.
  Unsupported settings fail visibly instead of silently disappearing.

Compared with the reviewed Ollama CLI, TLlama adds explicit connection/timeout
flags, common full-response `--json` output, repeated `--option NAME=JSON` request
overrides and layered client TOML preferences. Ollama already has options for
timings, thinking display, format and keep-alive; those basic capabilities are
not exclusive to TLlama.

The native `llama-server` already has extensive runtime flags, environment
settings and router model presets. TLlama's advantage is the integrated client
configuration and named per-model service layer around those controls, alongside
its OpenAI-compatible HTTP API. It is not a claim to expose more total native
flags than llama.cpp, or every option through an OpenAI request body.

See [client configuration and CLI](cli.md), [example client TOML](../examples/client.toml),
[daemon configuration](daemon.md), [model configuration](configuration.md),
[native fit controls](native-config.md#native-fit-memory-margin)
and [model repositories](model-repositories.md).

## Memory evidence beyond a process-size estimate

The native `/tllama/memory` extension supplies structured evidence to TLlama:

| Evidence | Practical use |
| --- | --- |
| Deduplicated backend buffers | Keep model, KV, compute, projector and adapter costs visible |
| Occupied weight ranges | Exclude sparse mmap holes and padding from weight payload; count tied/shared roles once |
| Final weight assignment and layers | Explain actual CPU/GPU offload, independently of a buffer's provider |
| Allocation locality and properties | Inspect successful CUDA/HIP/SYCL/Vulkan allocation paths and their documented coverage |
| Optional bounded tensor diagnostics | Trace a suspicious tensor/layer without treating repeated tensor roles as extra storage |

The Ollama-compatible `SIZE`/`PROCESSOR` view is derived separately from those
details, with explicit provenance and full-offload display policy. Polling `ps`
uses cached admitted-generation evidence; it does not load models, refresh TTL
or rescan multi-gigabyte weights. Invalid optional evidence can become unknown
without discarding unrelated valid sections. Load-log fallback remains labeled
and preserves incomplete coverage.

This is richer structured accounting, not a claim to measure physical residency,
driver overhead or free VRAM. It does not prove the original sparse mmap case
resolved. See [memory reporting](native-memory.md) and [runtime audit](runtime-audit-070.md).

## APIs and workflows in one service

| Workflow | Implemented surface and benefit |
| --- | --- |
| Text and chat | Ollama generate/chat and OpenAI chat-completions, streaming/nonstreaming, reasoning separation and usage/timings |
| Structured output | JSON/schema requests through the supported adapters and native sampler |
| Function tools | OpenAI call IDs/deltas and Ollama argument objects; validated tool histories and results; functions remain caller-executed |
| Image understanding | Inline images and a matching projector, with native capability checks and model-specific image-token controls |
| Semantic embeddings | Modern/legacy Ollama and OpenAI routes; text/token inputs, full output dimensions, float/base64 encoding and inline-image extensions for capable models |
| Text Decision models | `/v1/systemone` with choice, score and noul questions; Laya, Nimble and Clef have integrated typed contracts |
| Embedded MTP | Opt-in `draft-mtp` and draft limit per model, accepted/drafted statistics and request `draft_num_predict` translation |
| FIM | Literal prefix/suffix with model token metadata and configurable order; protocol support is separate from completion quality |
| Local and remote CLI | Interactive and one-shot use, stdin, model management, JSON output, reasoning display and locale-aware human statistics |

These are useful shared capabilities, not all exclusive to TLlama. The current
service has one native inference slot per worker; it does not advertise parallel
batch serving within that worker. `/v1/responses`, `/v1/completions`, audio APIs
and full Ollama compatibility remain separate work. The [API guide](api-compatibility.md)
and [runtime audit](runtime-audit-070.md) specify the actual surface and tested scope.

## Model storage and process ownership

Local GGUF imports can upload complete header-validated split-file groups and
projectors from a remote client, using digest-verified staging before publishing
the model definition. Direct Hugging Face pulls use an optional dependency and
retain the file's recognizable repository path; pull is a separate file-based
workflow, not a promise to fetch all split shards automatically.

Metadata comes from GGUF headers without starting inference. Persistent JSON
header and SHA-256 digest caches have independent schemas, and repository-relative
identity lets valid entries survive a mount-point change. A cold digest calculation
can still read the full file. These caches complement shared weight storage;
they do not replace filesystem permissions or the model definition registry.

Private native subprocesses isolate worker load settings. A definition or load
configuration change creates a new generation while admitted leases protect
the old one. Shared loads, expiry after last use, unload, worker failure and
shutdown use the same ownership model for generation, embeddings and Decision.
Cleanup is awaited and failed cleanup keeps capacity occupied. Private transport
credentials are separate from the public API, which has no built-in authentication.

See [imports](cli.md#create-and-import-models), [metadata cache](metadata-cache.md),
[worker transport](native-worker.md) and [residency](native-residency.md).

## Comparison with the reviewed projects

External comparison baseline reviewed on 27 October 2026; TLlama client
configuration and fit entries updated on 28 October for the develop/1.x branch.
Upstream development can change these details; the
[source audit](feature-source-audit.md) pins the inspected revisions and records
the configuration update.

| Area | TLlama | Reviewed Ollama | Official llama.cpp |
| --- | --- | --- | --- |
| K/V control | Independent K/V, global and per model | One global `OLLAMA_KV_CACHE_TYPE`, sent to both sides | Independent native K/V flags already exist |
| Device placement | Per-model native selectors, explicit lists across installed backends | Scheduler selects a backend group; numeric `main_gpu` and visibility controls also exist | Native device lists/split controls already exist |
| Instella-MoE | Experimental native-fork converter/loader/graph | No Instella registration found in inspected source | No Instella registration found in inspected source; community PR credited |
| Legacy ROCm deployment | Development kernels plus optional matching BLAS add-ons | Hardware/runtime policy differs; no blanket exclusivity claim | Generic HIP/GCN support already exists; the complete deployment is a separate concern |
| Memory inspection | Fork memory endpoint plus validated public provenance, payload/assignment/locality | No equivalent TLlama endpoint is claimed | Inspected source has no `/tllama/memory` extension |
| Client configuration and CLI | Layered system/user TOML; explicit address, timeout, JSON, request-option and display switches | Reviewed CLI uses `OLLAMA_HOST` for the address; no equivalent layered client preferences or common connection/JSON/option switches found | Rich native server flags, environment, router presets and web UI settings; a different scope from TLlama's remote CLI client |
| Service configuration | Persistent per-model TOML, inspectable global CLI/env/TOML layers and managed workers | Modelfiles, options and server environment have their own scope | Native flags/configuration and server capabilities; not the TLlama model-definition service |
| Native fitting policy | Global/model `fit` and `fit_target_mib`; automatic fitting or explicit all/off | Scheduler and runner policies have their own controls | `--fit`, `--fit-target` and explicit GPU-layer settings already exist |

The intended distinction is practical control and integrated extensions. GPU
backends, separate native K/V switches, embeddings and many model architectures
come from llama.cpp and its contributors. No universal speed, quality or
hardware-compatibility advantage is implied.
