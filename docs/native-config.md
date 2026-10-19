# Native configuration checkpoint (block 03)

Runtime fingerprint: **b11552-ec2797836f**. The native configuration modules do
not import llama-cpp-python. Their vocabulary is TLlama-owned and pinned to the
parser/protocol; an upstream option does not automatically become a TLlama API.

Public inference uses managed native workers. Explicit selectors remain native
runtime controls; automatic backend priorities are documented in
[backend-priority.md](backend-priority.md). No Python inference binding is used.

## Resolution and defaults

`native_config.resolve_worker_config()` takes BackendConfig, a parsed TOML and
storage paths resolved by the caller. It returns a launch contract, resolved
runtime values and their sources, global/model sampling and the default system
prompt. It does not load weights, read GGUF metadata, import files or start a
process. Missing device selection defers backend selection until worker startup.

| Setting | Native migration policy |
|---|---|
| `device` | TOML `[runtime].device` > `TLLAMA_RUNTIME_DEVICE` > automatic backend priority |
| Context | Request `num_ctx` > TOML `n_ctx` > global context; global default 0 = trained maximum |
| GPU layers | Explicit placement retains default `all`; automatic placement fits unset GPU layer count natively; explicit counts retained on GPU attempts |
| CPU | `device=none` forces layers 0, operation/KV/projector offload disabled; explicit contradictory offload rejected |
| Fit / parallel / context shift | Explicit placement: off; automatic fitting: on / 1 / off unless runtime `context_shift` or request `num_keep` enables it |
| Flash attention | Off unless configured; supplied model profiles remain on |
| Batch / ubatch | 512 / 512 unless configured |
| Threads | Omitted when unset: native automatic policy, replacing the binding's thread defaults |
| Mmap / mlock | True / false, resulting in `mmap`; explicit combinations map to the four modes below |
| K/V types | Explicit f16 baseline; shorthand expanded independently in each precedence layer |
| Sampling | Request > TOML > global explicitly configured values > GGUF recommendations supplied by caller > TLlama baseline |
| Sampler chain | Native chain; no promise of identical floating-point/token results to the binding |

Native device selectors (`none`, `Vulkan0`, `ROCm0`, `CUDA0`, comma-separated
device names) are not persistent physical IDs. Backend visibility overrides
remain per-child environment values. Device inventory, stable IDs and automatic
placement are later work, not an implicit fallback in this resolver.

Within one KV layer, `type_k`/`type_v` override `type_kv`. A TOML shorthand overrides
both global sides; a TOML side overrides only that side. Allowed names/enums:
f32/0, f16/1, bf16/30, q8_0/8, q4_0/2, q4_1/3, iq4_nl/20, q5_0/6, q5_1/7.
These are the pinned parser's KV types, not its full weight quantization list.

## CLI mapping

Load options are typed and validated before CLI serialization. `WorkerLaunchSpec`
copies nested option containers and prevents additional options from replacing
supervisor-owned model/address/parallel/security fields. CLI is executed without
a shell; paths and serialized values stay individual argv elements.

| TLlama runtime setting | Native CLI | Scope |
|---|---|---|
| Model / projector paths | `--model`, `--mmproj` | Resolved storage paths |
| `device`, `n_gpu_layers`, `n_ctx` | `--device`, `--n-gpu-layers`, `--ctx-size` | Load |
| `type_k`, `type_v`, `flash_attn` | `--cache-type-k`, `--cache-type-v`, `--flash-attn on/off` | Load |
| `n_batch`, `n_ubatch` | `--batch-size`, `--ubatch-size` | Load |
| `n_threads`, `n_threads_batch` | `--threads`, `--threads-batch` | Load |
| `main_gpu`, `tensor_split` | `--main-gpu`, CSV `--tensor-split` | Load; main_gpu does not choose a backend |
| `split_mode` 0/1/2 | `--split-mode none/layer/row` | Load |
| `use_mmap`, `use_mlock` | `--load-mode none/mmap/mlock/mmap+mlock` | Load |
| `context_shift` | `--context-shift` / `--no-context-shift` | Load; request `num_keep` enables it |
| `offload_kqv`, `op_offload` | Positive/negative `--kv-offload`, `--op-offload` | Load |
| `swa_full`, `spm_infill` | Positive flags when true, omitted when false | Load; spm flag is not proof of public FIM parity |
| `rope_scaling_type` -1/0/1/2 | Omit / `--rope-scaling none/linear/yarn` | Load; longrope has no parser mapping |
| `rope_freq_base`, `rope_freq_scale` | `--rope-freq-base`, `--rope-freq-scale` | Load |
| `yarn_ext_factor/attn_factor/beta_fast/beta_slow/orig_ctx` | Corresponding `--yarn-*` flags | Load |
| `numa` false/0, true/1, 2, 3 | Omit, distribute, isolate, numactl | Load; mirror is unavailable in this parser |
| `seed` | `--seed`; explicit request seed wins | Worker sampler default |
| `last_n_tokens_size` | Native request `repeat_last_n`, default 64 | Request fallback originating in runtime config |
| `no_perf` | `--no-perf` / `--perf` | Internal libllama perf; server request wall-clock timings remain separate |
| `kv_overrides` | Typed CSV `--override-kv` entries | TOML-only; bool/int64/finite float/string |
| `lora_path`, `lora_scale` | `--lora` / `--lora-scaled` | Resolved inside model store; no new `[lora]` section |
| `verbose` | Only true accepted; supervisor log policy stays enabled | Diagnostics |
| `spec_type` | Omitted when unset; `--spec-type none/draft-mtp` when explicit | Load; embedded MTP is opt-in |
| `draft_n_max` | `--spec-draft-n-max` only with MTP; default 3 when enabled | Load; see [native-mtp.md](native-mtp.md) |
| `embedding` | `--embeddings` when true; speculation explicitly disabled | Load mode; see [native-embeddings.md](native-embeddings.md) |
| `pooling_type` -1/1/2/3 | Omit / `--pooling mean/cls/last` | Embedding load only; -1 retains the model default |
| `[template].jinja` | `--jinja --chat-template-file` | Private 0600 file owned and removed by worker |
| `[mmproj].image_min_tokens/max_tokens/batch_max_tokens` | `--image-min-tokens`, `--image-max-tokens`, `--mtmd-batch-max-tokens` | Projector load |
| `[mmproj].warmup` | `--warmup` / `--no-warmup` | **Whole worker**, replacing projector-only scope |

Decision mode is selected from audited GGUF Decision metadata, rather than a
new runtime flag. See [native-decision.md](native-decision.md) for its separate
head/template contract, conflicting generation settings and private transport.

LoRA's native loading behavior is used; the old binding's automatic mmap override
is not emulated. Scaled LoRA paths containing ':' cannot be represented by this
pinned parser and are refused. CSV quotes/commas in other paths/metadata strings
are escaped. KV override keys/strings must fit the native 127-byte UTF-8 fields.
TOML round-trip content is retained; resolving does not rewrite supplied files.

## Request mapping and migration errors

`build_native_sampling()` applies the existing TLlama precedence and translates
`mirostat_mode` to `mirostat`, `max_tokens` to `n_predict`, nonpositive top_k to 0,
and TOML token-bias keys to a JSON token-ID map. An unset seed is omitted so the
worker's configured seed applies; a request seed of 0 remains explicit.

An unset or negative token limit maps to native -1. Public 0 remains zero-token
prefill (requires native correction 0078); public -2 is translated to -1 with
context shifting disabled. See [ollama-options.md](ollama-options.md) for all
20 client parameters, request runner overrides and reload behavior.
Empty messages/prompts remain ModelManager preload/unload policy, not sampler work.
Format, thinking/template kwargs and public response mapping belong to their next
integration milestones. Merely storing system_prompt does not insert it twice.

| Removed/rejected setting | Reason or replacement |
|---|---|
| `tfs_z`, even 1.0 | No native tail-free equivalent; no silent emulation |
| Negative temperature | Native would clamp to greedy; refuse the semantic change |
| `chat_format`, `chat_handler` | Python handler registry is not native; use `[template]` / `[mmproj]` |
| `draft_model`, `tokenizer` | Python objects cannot be serialized; embedded MTP uses `spec_type`, external drafts remain unintegrated |
| `vocab_only` | Metadata reads do not need a worker |
| `logits_all` | Public logprobs require separate integration |
| `attention_type` | Pinned server CLI has no override; attention stays the model default |
| `lora_base` | No equivalent in pinned parser |
| Unknown names, unsupported KV/enum values, nonfinite values | Explicit configuration/request errors rather than a silent drop/clamp |

`model_path` still comes from `[llm]`. Dedicated context/K/V environment spellings
remain authoritative instead of duplicate `TLLAMA_RUNTIME_N_CTX/TYPE_*` variables.
Structured `stop`, `logit_bias` and `kv_overrides` remain TOML-only in the environment
interface. Public sampler input keeps its existing supported field scope.

## Preview and verification

From the repository root, with dependencies installed:

```bash
TLLAMA_RUNTIME_DEVICE=Vulkan0 PYTHONPATH=src python scripts/worker-config.py \
  --toml /path/to/Qwen3.6-35B-A3B-UD-IQ3_S.toml
```

Repeat for `Qwen3.6-35B-A3B-UD-mmproj-IQ3_S.toml`. The expected differences are:
text ctx 4096, K/V q4_0/q4_0; vision ctx 0, K/V q8_0/q4_0 and a projector path.
Both retain flash attention on. TOML device wins if present. `--models-dir` can
override the store for preview; `--ctx` simulates a request context override.

The JSON contains requested runtime settings, sources, argv and native sampling.
It is explicitly a dry run: key/template paths are illustrative, no process is
started, and GGUF recommended samplers are not read. Effective slot context,
device/kernel availability, template/vision parity and allocations are runtime
checks in the following integration milestones.

Sources: pinned `common/arg.cpp` (flags/CSV/KV types), `include/llama.h` and
`ggml/include/ggml.h` (enums), `tools/server/server-common.cpp` and
`tools/server/server-task.cpp` (request/response protocol). Compatibility stays
tied to fingerprint b11552-ec2797836f.

ModelManager now resolves these settings for owned worker generations; see
[native-residency.md](native-residency.md) for leases, capacity, TTL and the
managed hardware check. Public API translation remains the next milestone.
