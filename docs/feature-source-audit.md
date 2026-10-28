# Source evidence for features and hardware — 0129

Reviewed 27 October 2026 after the documentation-only 0128 patch. This record
supports the [feature overview](features.md), [hardware tables](hardware.md)
and [distribution guide](distribution.md); [TOC](../TOC.md) indexes all guides.
It records implementation and comparison provenance, not a new hardware test
or a security/performance audit.

## Source boundaries

| Source | Inspected identity | Use |
| --- | --- | --- |
| TLlama | Post-0128 tree, preparation commit `0c6b82654e279b260411d2ff63aacd01870a391d` | Complete application module inventory, public configuration/API surfaces and existing qualification records |
| Patched native runtime | `ByCzech/llama.cpp`, branch `tllama/native`, commit `0e9433bfd33d8cdb194285b70c0d421d39135df2` | Instella implementation, gfx1250 paths, memory extension, backend/CPU build inventory |
| Public packaging snapshot | `ByCzech/tllama-server`, commit `672130649135bbf85a613e6d289e52e4998393c0` | Package names, CUDA architecture lists, SYCL and CPU build flags, private layout and alternatives |
| Applied packaging development series | Package `0.7.0~dev+b11850.1-1`, series through 0031, documented native handoff and accepted captures | New native pin and the three added legacy HIP targets/optional BLAS add-ons; not falsely attributed to the older public main tree |
| Official llama.cpp | `e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf` | Current comparison of model registration, backend flags, gfx1250 classification and absent TLlama memory extension |
| Ollama | `8ec9ecdf692e417798169ff8d5972159078d3754` | K/V setting, native launch arguments, model registration, API options and scheduler grouping |

The packaging public main tree is older than the development package series
used in accepted captures. Its README still says legacy kernels are absent;
that statement cannot describe the applied 0.7.0 series. The current packaging
development tree was not available in that public main snapshot, so this review
does not claim to have diffed its complete unpublished changes. Legacy target
additions are grounded in the native handoff/development package record.

The application inventory covers all **78 Python modules** under `src/tllama`,
including configuration, helpers, schemas, routers, CLI, storage, lifecycle,
transport and telemetry. Public claims were checked against their implementing
modules, relevant tests and current guides. Native review covered model/GGUF
registration/conversion, the Instella graph, memory contracts and exporters,
backend dispatch/build definitions and package recipes. It is not a claim of
line-by-line formal verification of all upstream kernels or bundled libraries.

## Implementation map

Paths without a project qualifier are relative to the TLlama repository.
Native and packaging paths use the pinned external repositories above.

| Claim | Source and regression evidence | Scope |
| --- | --- | --- |
| Independent global/model K/V | `daemon_config.py`, `config.py`, `native_config.py`, `native_params.py`, `worker_config.py`; configuration/worker tests | Per-layer shorthand expansion, supported native types and explicit side precedence |
| Per-model backend/device list | `native_params.device_problem`, `resolve_worker_config`, `WorkerLaunchSpec.command`; native configuration/worker tests | Native selector strings/CSV retained; no physical identity or mixed-hardware qualification inferred |
| Ordered automatic backend fallback | `backend_policy.py`, `automatic_worker.py`, device inventory; backend priority tests | One family per attempt; CPU final candidate; explicit placement bypasses policy |
| Configuration source inspection | `daemon_config.py`, `daemon_options.py`, `daemon.py`; daemon config tests | CLI/env/TOML/defaults, check/print, supported schema rather than arbitrary environment dump |
| Typed advanced load/sampling options | `native_params.py`, `native_cli.py`, `native_config.py`, helpers/runtime_params, helpers/sampling_params | RoPE/YaRN, NUMA, offload, mmap/mlock, LoRA and metadata override vocabulary; no arbitrary CLI pass-through |
| Shared weights and editable profiles | `model_definitions.py`, `helpers/model_toml.py`, `maintenance.py`; model management tests | Copy definition, preserve TOML comments, backup protection; not shared live workers across names |
| Verified import and shards | `client_import.py`, `model_uploads.py`, `helpers/gguf_import.py`; client/import/upload tests | Client paths, headers/digests/group validation, staged publication and optional projector |
| Direct Hugging Face pull | `backend.py`, Ollama router and optional `huggingface` dependency | Requested file workflow; permissions and split-group behavior remain explicit |
| Header/digest persistence | `helpers/gguf_metadata.py`, `helpers/metadata_cache.py`; metadata and relative-cache tests | Offline header reads, separate schema/fingerprints, relocation reuse; cold digest can scan entire file |
| Lifecycle and isolated load state | `residency.py`, `worker.py`, `worker_config.py`, `worker_ipc.py` | Owned load tasks, immutable generations/leases, TTL, cancellation/reap, per-child environment cleanup |
| One-slot request transport | `worker.py`, `native_routes.py`, embedding/Decision transports | Requests serialized within a worker; multiple model capacity is a separate setting |
| Native memory evidence | Native `tools/server/tllama-memory-v1.md`, development allocation/weight contracts; TLlama `native_memory_api.py` | Immutable inventory, assignment, occupied ranges, locality and bounded tensor diagnostics |
| Public memory/unknown handling | `native_telemetry.py`, `native_reporting.py`, `native_memory.py`; native runtime fixtures and memory tests | Separate provenance/coverage, payload display policy, cached `ps`, no physical-residency claim |
| Generation/reasoning/schema | Routers, `native_routes.py`, `native_requests.py`, `native_api.py`, helpers/reasoning_split | Exact supported public surface; no implied Responses/audio API |
| Tools | `native_tool_input.py`, `native_tools.py`, helpers/tool_capability | Strict declarations/history, native call assembly, offline conservative advertisement; caller executes tools |
| Vision | `native_requests.py`, helpers/vision, projector launch settings | Inline media and live capability admission; output quality remains model/runtime dependent |
| Embeddings | `embedding_contract.py`, `embedding_media.py`, `native_embeddings.py`, schemas/embeddings | Dimensions/pooling/normalization/token accounting, float/base64 and explicit inline-image extension |
| Text Decision | `decision_contract.py`, `native_decision.py`, schemas/decision | Laya/Nimble/Clef metadata/head/template and typed choice/score/noul response validation |
| FIM | helpers/fim, `native_fim.py`, schemas/generation, model `[fim]` | Token metadata/order, literal prefix/suffix; quality selection remains separate |
| Embedded MTP | `native_params.py`, `native_config.py`, `ollama_options.py`, `worker.py` | Only none/draft-mtp strategy exposed; statistics and reviewed off/on captures |
| Client ergonomics | All `cli*.py`, `client.py`, `__main__.py`, `version.py` | Local/remote, REPL/stdin/JSON, human locale/timings/reasoning and clean help/version import boundary |
| Public schema and operational bounds | schemas, middleware, helpers/cors, errors and routers | Typed inputs/route-specific errors, configured origins and advertised compatibility; no built-in public authentication |
| Shared fingerprint policy | `runtime_fingerprint.py`, worker startup and qualification scripts | warn/strict and exact diagnostic pins; live protocol checks remain independent |
| Debian runtime modularity | Packaging `debian/control`, `build-component.mk`, maintainer scripts and CMake | Exact core/backend version pairing, SDK alternatives, independently buildable components and private library tree |

Implementation files in this table live under `src/tllama/` unless prefixed
with native/packaging or explicitly described as tests/scripts. The full
configuration/API guides remain authoritative for exact keys and rejection rules.

## Native extensions and attribution

### Instella-MoE

Native `src/models/instella-moe.cpp`, architecture registration, GGUF Python
constants and the HF converter implement the experimental family. Native
`tests/test-instella-moe.cpp`, `tests/test-instella-moe.py` and
`docs/development/tllama-instella-moe.md` describe the independent NumPy oracle,
cached decode/prefill and unsupported architecture variants. TLlama's metadata
reasoning handling was adjusted in 0113; maintainer inference/thinking acceptance
is recorded separately from synthetic graph checks.

The implementation adapts Csaba Kecskemeti's community
`instella-moe` commit `e90a9ad690643e49eec368e12f6b4f9f7f197801`, associated with
[upstream PR 26467](https://github.com/ggml-org/llama.cpp/pull/26467).
The official upstream and Ollama snapshots inspected here have no Instella
architecture/model registration. This is a dated source finding, not a claim
that no community fork or future upstream version can support it.
The implemented family is `InstellaMoEForCausalLM`, not every model using the
Instella name. Native `LLAMA_SPLIT_MODE_TENSOR` (mode 3) is explicitly rejected;
that is not a blanket rejection of the separate `tensor_split` ratio setting
used with other split modes. TLlama exposes only none/layer/row modes, and
Instella layer/row recipes remain unqualified. Matching K/V is the
maintainer-tested recipe. Broad SYCL/model/quantization acceptance is absent.

### AMD paths

Native `ggml/src/ggml-cuda/vendors/hip.h` defines `GFX1250` separately from
generic RDNA4; `common.cuh` retains distinct host capability checks and guards
the dot2 helper. Related MFMA/WMMA dispatch uses the resulting architecture
classification. The reviewed official upstream still uses generic `__GFX12__`
classification without this gfx1250 distinction.

The GCN APU host classification for gfx909/gfx90c is present in **both** the
fork and inspected current upstream. It must not be advertised as a new
exclusive fix. The project advantage is the maintained legacy kernel/library
deployment on the modern ROCm stack, with explicit coverage and its unresolved
mmap limitation. Hardware family names are checked against AMD/LLVM references
linked in [hardware](hardware.md).

### CPU and other inherited features

Native `ggml/src/CMakeLists.txt` enumerates CPU variants and backend options;
`ggml-cpu/CMakeLists.txt` and `ggml-blas/CMakeLists.txt` define optimizations and
BLAS providers. BLIS/AOCL support and GCN APU classification also occur in
inspected official upstream. CPU kernels, OpenBLAS, MKL/Accelerate integrations,
SYCL and the other general GPU backends are credited as inherited capabilities.
The Debian CPU default is OpenBLAS; no new BLAS/ZenDNN policy selector is claimed.

Nemotron 3.5 Bolt support is already in the pinned native upstream base and was
verified by the maintainer. Its presence is useful but is not presented as a
TLlama-only model implementation.

## Comparison checks

### Configuration and CLI update — 28 October 2026

The develop/1.x tree through `8473dc2` adds client configuration (0134–0136)
and native fit configuration (0137–0138) after the original 0129 review.
`client_config.py`, `cli.py`, `cli_run.py` and `cli_output.py` implement system/user
client TOML layering, isolated explicit file selection, address/timeouts, JSON,
typed request options and display preferences. `native_params.py`, `native_cli.py`,
`native_config.py`, `worker_config.py` and `automatic_worker.py` implement typed
fit margins and explicit fitting policy across device selection. Their client,
configuration and native-fit tests cover the respective contracts. These are
development-branch additions, not a claim that the 1.0.0 tag contained them.

For the Ollama comparison, the pinned `cmd/cmd.go` registers run controls for
verbose, format, think, hidethinking and keepalive, among others. These are not
advertised as TLlama-exclusive features. The inspected CLI reads the address via
`OLLAMA_HOST`/the environment-based API client; no equivalent system/user client
TOML or common `--host`, `--timeout`, `--json` and repeated `--option` switches
were found in that command surface. This does not claim that Ollama has no
configuration files: its FAQ also documents `~/.ollama/server.json` for disabling
cloud functionality. Server config, model Modelfiles and external launch-tool
configuration have different scopes from TLlama's client preferences.

For the context-before-preload example, TLlama's `RunSession.create` retains
the parsed options, `RunSession.preload` forwards them, and `submit` reuses the
same controls. In the pinned Ollama source, `RunHandler` calls
`loadOrUnloadModel` before entering the interactive loop; `cmd/interactive.go`
provides `/set parameter num_ctx`. No generic `run --option` registration was
found. The claim concerns avoiding a default-then-custom-context CLI sequence,
not the absence of Ollama Modelfile, environment or HTTP alternatives.

The official native `tools/server/README.md` and `common/arg.cpp` already expose
extensive server flags, including `--fit`/`--fit-target`, router INI presets and
web UI JSON settings. TLlama adds typed global/model configuration and a remote
CLI around a supported subset; it does not invent native fitting or claim a
larger total flag count than llama-server. OpenAI wire compatibility does not
imply that every load setting is an OpenAI request parameter.

Sources: [pinned Ollama CLI](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/cmd/cmd.go),
[API client](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/api/client.go),
[Ollama FAQ](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/docs/faq.mdx),
and [native server reference](https://github.com/ggml-org/llama.cpp/blob/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf/tools/server/README.md).

### Original native comparison findings

Ollama's inspected `envconfig/config.go` declares one `OLLAMA_KV_CACHE_TYPE`;
`llm/server.go` resolves it and `llm/llama_server.go` passes the same value as
both `--cache-type-k` and `--cache-type-v`. Official llama.cpp already has the
two independent native flags. TLlama adds persistent independently scoped
service/model configuration around those native controls.

Ollama's `api/types.go` **does include `main_gpu`**, and its scheduler has an
explicit-main-GPU path. Thus “Ollama cannot choose a GPU” is incorrect for the
inspected version. `server/sched.go` groups multi-device placement using
`ml.ByLibrary` and selects a library group. The supported TLlama difference is
direct per-model native selectors/lists and explicit heterogeneous requests,
with no claim that all mixed combinations were tested.

The inspected official native source does not include the fork's
`/tllama/memory` contract. General process metrics or memory fitting in another
project are not claimed absent; the specific distinction is the TLlama structured
payload/assignment/locality extension and its validated public projection.

### Main/auxiliary backend placement — 28 October 2026

The placement examples use the existing resolver, `WorkerLaunchSpec.command`
and `NativeResidency`: each public model keeps its selector, gets a private
worker and occupies one resident-generation slot. There is no physical-card
association or cross-backend exclusivity check in these paths. The device
inventory explicitly keeps native observations separate from physical PCI
evidence. The pinned native `common/arg.cpp::parse_device_list` resolves exact
device names; `src/llama.cpp::llama_prepare_model_devices` retains explicitly
selected devices instead of applying the automatic selection/deduplication
branch. Different workers can therefore request different backend views of one
card without application changes; this is not new hardware inference evidence.

Pinned Ollama `ml/device.go::DeviceID` contains both `ID` and `Library`.
`server/sched.go::selectLlamaServerPlacement` groups by library, and
`bestExplicitMainGPU` considers `group[mainGPU]` in each eligible group before
selecting a candidate. Thus numeric `main_gpu = 0` is not an explicit backend
selector, but saying Ollama cannot internally distinguish backend/device pairs
would be incorrect. TLlama's concrete difference is named per-model selectors
such as CUDA0/ROCm0/Vulkan0 for the main and auxiliary roles.

Sources: [Ollama scheduler](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/server/sched.go),
[device identity](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/ml/device.go),
[native device parsing](https://github.com/ggml-org/llama.cpp/blob/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf/common/arg.cpp),
and [native device selection](https://github.com/ggml-org/llama.cpp/blob/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf/src/llama.cpp).

### Pinned comparison links

- [Ollama K/V environment](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/envconfig/config.go),
  [native launch](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/llm/llama_server.go),
  [options](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/api/types.go)
  and [scheduler](https://github.com/ollama/ollama/blob/8ec9ecdf692e417798169ff8d5972159078d3754/server/sched.go).
- [Official native arguments](https://github.com/ggml-org/llama.cpp/blob/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf/common/arg.cpp)
  and [HIP architecture classification](https://github.com/ggml-org/llama.cpp/blob/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf/ggml/src/ggml-cuda/vendors/hip.h).
- [Patched native tree](https://github.com/ByCzech/llama.cpp/tree/0e9433bfd33d8cdb194285b70c0d421d39135df2)
  and [public packaging build recipe](https://github.com/ByCzech/tllama-server/blob/672130649135bbf85a613e6d289e52e4998393c0/debian/build-component.mk).
- [Ollama FAQ](https://docs.ollama.com/faq) corroborates the global K/V setting;
  links to vendor hardware documentation are collected in [hardware](hardware.md).

## Validation and remaining boundaries

0129 changes Markdown only. Validate local links/anchors, fenced example syntax,
complete TOML model examples using the real parser/resolver, selector/KV launch
serialization, TOC coverage, package/target inventories and clean application
after 0128. These checks do not load a model or require new captures.

The last complete application suite remains the maintainer's **4086 passed,
2 warnings** after 0127. Do not relabel that result as a new 0129 test run.
The accepted runtime audit is unchanged. Pull errno 13, original sparse mmap,
legacy HIP mmap and separate FIM quality/release decisions remain open.
Windows/macOS builds/installers are planned; neither native portability nor this
documentation change qualifies those application platforms or publishes 1.0.0.
