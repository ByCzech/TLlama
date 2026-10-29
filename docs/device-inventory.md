# Device inventory and placement design — 0070

## One-shot daemon command (1.2)

```sh
tllamad --list-devices
tllamad --config /etc/tllama/server.toml --list-devices
tllamad --no-config --runtime-path /absolute/path/to/tllama-server --list-devices
```

This local command forwards the selected runtime's `--list-devices` stdout,
stderr and exit status. It uses normal daemon config/CLI/environment precedence
and worker environment sanitization. It does not start the HTTP application,
create the model store, load a model or require an existing daemon. The child
has a 30-second deadline and a combined 4 MiB output limit, with owned-process
cleanup on failure or interruption. It does not enforce a diagnostic fingerprint
allowlist or reinterpret the native table. Startup failures return 1; native
exit codes are preserved (signals use 128 + signal).

Run it as the intended service user with the intended visibility environment.
Separate backend selectors can describe the same physical GPU; CPU selection
is `none` and is not part of this native device table. Listing can initialize
SDKs/drivers and is not a guarantee of zero GPU overhead. The more detailed
capture procedure below remains available for qualification.

This checkpoint captures discovery evidence without loading a model or changing
production placement. Run it on the current test machine with the same installed
native binary and inherited environment as the accepted inference runs:

```bash
result_dir=/tmp/tllama-devices-$(date +%Y%m%d-%H%M%S)
PYTHONPATH=src python3 scripts/device-inventory.py --output "$result_dir"
tar -C "$(dirname "$result_dir")" -cJf "$result_dir.tar.xz" "$(basename "$result_dir")"
```

Return the whole tarball and pytest results. This first run needs no additional
GPU, model, download or native rebuild. It does not contact a running TLlama
service or alter registered TOMLs, alternatives, global environment or sysfs.
Backend discovery initializes installed SDK plugins and may create driver
contexts; enumeration alone does not prove usable model kernels or zero driver
overhead. Mixed-backend inference is a subsequent hardware gate.

## What is measured

The tool runs two short-lived owned processes sequentially: native `--version`,
then `--list-devices`. By default it requires the exact audited build
`b11850-0e9433bfd3` and checks
that executable identity did not change during capture. A separate temporary cwd
is removed at completion. No model argument or listener configuration is passed.
Both pipes share a 4 MiB cap; each command has a 30-second deadline, plus bounded
TERM/KILL cleanup. The owned process group is terminated as needed and the direct
child is reaped on failure/cancellation; an unreaped child is an explicit failure.
`--timeout` and `--output-limit` change these bounds. No shell is used.

`--expected-fingerprint` selects another exact identity. Both the current build
and the historical `b11552-3144c167ef` are independently audited for discovery;
an exact-match unreviewed candidate still sets `review_required=true`.
Malformed tables remain failures even for an audited identity. See
[runtime-audit-070.md](runtime-audit-070.md) for reviewed captures and scope.

`--executable` selects the installed binary; `--pci-root` defaults to Linux
`/sys/bus/pci/devices`. A repeated `--env NAME=VALUE` can override documented
visibility variables in the child only: CUDA_VISIBLE_DEVICES,
ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, GGML_VK_VISIBLE_DEVICES,
ONEAPI_DEVICE_SELECTOR, SYCL_DEVICE_FILTER, ZE_AFFINITY_MASK and
GGML_DISABLE_VULKAN. Empty values are retained literally. These controls belong
to their backend/SDK; acceptance here does not assign common index semantics.
The worker's existing environment policy removes inherited LLAMA_ARG_*, AIP_*,
LLAMA_API_KEY, MTMD_BACKEND_DEVICE and GGML_BACKEND_PATH. Installed library/plugin
selection remains unchanged. The full environment and credentials are not logged;
the report explicitly does not claim a complete environment fingerprint.

`inventory.json` equals stdout. `version.stdout/.stderr` and
`devices.stdout/.stderr` retain bounded raw evidence, including startup warnings.
An existing output path is refused. Exit 0 means the exactly selected native
table was captured successfully; inspect `review_required` for unaudited
candidates or incomplete PCI evidence.
Exit 1 reports capture/protocol failure; exit 2 is invalid CLI/output. A capture
can succeed while all physical associations remain unknown by design.

## Source audit and current evidence boundary

Pinned fork: `ByCzech/llama.cpp` at
`3144c167ef703e93cee0b58d7236b6a28c6e50ad`, inspected as source only:

- `common/arg.cpp::common_print_available_devices()` loads installed backends,
  omits CPU-type devices and prints native selector, description and total/free
  memory after integer division to MiB. The argument handler exits before model
  loading and HTTP listener setup in `tools/server/server.cpp`.
- `common/build-info.cpp.in` prints version plus explicit build/commit fields;
  the diagnostic derives the native fingerprint from those fields.
- `ggml/include/ggml-backend.h::ggml_backend_dev_props` already has a nullable
  `device_id`: PCI devices should use lower-case domain:bus:device.function.
  The human CLI table does not expose this property or the device's registry.
- `ggml/src/ggml-backend-reg.cpp` loads installed backend modules, including the
  conditional Vulkan registration path. Discovery depends on that binary,
  selected plugins and child environment.
- TLlama's current memory schema has per-snapshot device IDs, backend/name/type;
  these IDs do not provide a persistent registry or physical identity bridge.

The native table can include BLAS or other non-CPU accelerators. Rows are not
automatically labelled GPU. Backend/index hints for CUDA/ROCm/Vulkan/SYCL are
derived only from selector spelling and explicitly marked as hints. Unknown
selectors are retained. Duplicate selectors, malformed/truncated tables and
inconsistent memory values make the capture incomplete, rather than silently
choosing a device. Descriptions, including identical names, stay literal.

Reported memory stays in **truncated MiB**, not fabricated exact bytes. A zero
does not prove zero usable memory; native backends differ in their memory queries.
Free memory is transient, not a model-fit promise. Do not sum rows across APIs:
the same physical GPU may appear through ROCm/CUDA and Vulkan, and iGPU budgets
may describe overlapping shared memory.

Separately, sysfs supplies display/processing-accelerator PCI locations,
vendor/device/subsystem IDs, class and driver basename. BDF is a host topology
location, not a globally unique immutable card serial. Missing/malformed fields
remain unavailable and are reported; absent sysfs is not an empty proven machine.
The diagnostic never pairs PCI rows and native rows by order, description,
memory size or a backend index. All such associations remain unavailable.

## One card, multiple native selectors

`CUDA0`/`ROCm0` and `Vulkan0` are backend-scoped selectors. They may refer to the
same physical GPU or to different GPUs; matching index numbers or descriptions
do not establish identity. The explicit model selector is passed unchanged to
the native worker, and separate model workers may use separate backend views
of one card. This shares physical capacity, not allocations or an extra memory
pool. The inventory does not merge their rows or impose a physical-card lock.
See the [main/auxiliary placement example](features.md#main-model-and-auxiliary-models-on-different-backends)
and its [shared-card variant](features.md#one-physical-card-exposed-through-two-backends).

## Identity model for the next implementation

| Layer | Meaning | Stability / rule |
| --- | --- | --- |
| Capture/observation ID | One report and one native row | Ephemeral; never store as model configuration |
| Native selector, e.g. ROCm0 or Vulkan0 | Argument accepted by this enumeration context | May change after visibility filtering, SDK/plugin changes or reboot |
| Backend registry identity | Actual native registry/API owning a device | Must come from structured native evidence, not a guessed prefix |
| Physical key | Typed PCI BDF or backend-provided UUID with explicit scope/provenance | Absent means unknown; PCI location can change or be occupied by another card |
| Placement target | Selected backend interface plus explicit physical constraint | Resolve uniquely under the actual child environment before loading |
| Actual placement | Loaded tensor/compute/projector assignment evidence | Check after loading; a requested selector is not proof of placement |

Two interfaces can share a physical key while retaining different selectors and
capabilities. Multiple cards with the same name remain distinct. Do not combine
unavailable identities into one device, treat a Vulkan index as a ROCm index, or
reuse an index from a discovery process with different visibility restrictions.
`device=none` remains an explicit CPU launch control, not an invented native
enumeration row or an automatic fallback for an unavailable GPU.

The next native change should add a **versioned structured, model-free discovery
CLI** to the existing server executable. A proposed `--list-devices-json` is not
implemented or required by 0070. It should expose exact selector, backend registry,
device type, nullable physical identifier with kind/scope/provenance, memory
values with their backend meaning and capability flags. The existing C device
properties provide part of this evidence; missing UUID/backend data must stay
unknown. Use the same loader/environment as the eventual worker, without ctypes
in the application or a second persistent server. Review native compatibility
and any new build separately.

Then implement and test a TLlama resolver that rejects missing/ambiguous targets
before loading and validates the actual worker assignment. A placement config
should select a backend plus a physical constraint where available, retaining
legacy native selectors as explicitly context-scoped choices. New TOML fields,
registry API names and automatic scheduling are still design decisions; this
patch accepts none of them. Existing main_gpu keeps its native configured-list
meaning and is not promoted to a global physical GPU ID.

Finally verify two resident models concurrently: main generation on CUDA/ROCm
dGPU and embedding/Decision on Vulkan iGPU, with distinct per-model KV settings,
actual assignments, cancellation/recovery, memory and unload. Keep model
residency/capacity policy in ModelManager and native slot scheduling inside each
worker. Do not infer mixed-backend support for one model from separate workers.

## Regression checks

```bash
pytest -q tests/test_device_inventory.py tests/test_worker_config.py \
  tests/test_native_config.py tests/test_native_memory_api.py
pytest -q
```

Tests use synthetic PCI files and short-lived executable stubs. They verify
duplicate/same-name/index-zero cases without fabricated physical links, units,
format/version drift, visibility isolation, strict no-load argv, bounded pipes,
timeouts and process cleanup. They do not certify hardware kernels or placement.
