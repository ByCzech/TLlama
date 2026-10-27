# Hardware and native backends

TLlama selects devices exposed by its installed `tllama-server` runtime.
The Python application does not compile GPU kernels or infer support from a
product name. The current development baseline is native
`b11850-0e9433bfd3`, distributed in the reviewed Debian package set
`0.7.0~dev+b11850.1-1`.

This guide distinguishes **included targets/build support** from **accepted
inference configurations**. A compiler target, listed device or successful
library test alone is not a model inference qualification. See
[runtime audit](runtime-audit-070.md), [distribution](distribution.md) and the
[source audit](feature-source-audit.md) for evidence and version boundaries.

## Selecting a device

List the selectors supplied by the installed runtime:

```sh
/usr/lib/tllama/server/tllama-server --list-devices
```

Use the reported spelling in a model's `[runtime].device`, for example
`CUDA0`, `ROCm0`, `SYCL0` or `Vulkan0`. CPU is `none`. Indices depend on the
runtime and visibility settings; they are not persistent physical identifiers.
Explicit selection takes precedence over automatic backend priorities.
See [placement examples](features.md#model-placement-across-devices-and-backends).

| Backend | Deployment in the reviewed package family | Hardware route |
| --- | --- | --- |
| CPU / OpenBLAS | Core + CPU package | Host CPU; automatic supported instruction variant |
| Vulkan | Optional package | AMD, NVIDIA or Intel device with a usable Vulkan compute driver |
| ROCm/HIP | Versioned ROCm 10.1 package | AMD target plus matching HIP/BLAS libraries |
| CUDA | Versioned 12.4, 12.9, 13.3 packages | NVIDIA; select toolkit generation for the GPU family |
| SYCL | Intel oneAPI package | Supported Intel GPU and compute driver |
| Metal | Native build capability | Apple hardware with a suitable Metal build; packaged macOS delivery is planned |

Installed plugins can initialize or print messages even when another backend
is selected. Use the admitted worker's device/configuration and actual inference
evidence to identify the backend in use.

## AMD: legacy Vega extensions

The development package series adds these three HIP targets alongside the
existing 25 targets. Matching optional BLAS kernel add-ons complement the core
ROCm stack; install the add-on for the actual target.

| Native target | Family and representative hardware | Matching BLAS add-on |
| --- | --- | --- |
| `gfx900` | Vega 10 / GCN5: Radeon RX Vega 56/64, Vega Frontier Edition, Radeon Instinct MI25 | `amdrocm-blas10.1-gfx900` |
| `gfx906` | Vega 20 / GCN5.1: Instinct MI50/MI60, Radeon VII and Radeon Pro VII | `amdrocm-blas10.1-gfx906` |
| `gfx90c` | Later Vega APUs, including Renoir/Cezanne-class Ryzen APUs; the project example is Ryzen 7 5700G | `amdrocm-blas10.1-gfx90c` |

The target must be verified from the actual machine, for example using
`rocminfo`; a Ryzen series number or the marketing label Vega is insufficient.
Earlier Vega APUs and other GCN targets such as gfx902/gfx904/gfx909 are not
implicitly included by adding these three targets. gfx90c is built natively,
without an `HSA_OVERRIDE_GFX_VERSION` disguise as gfx900.

The fork's host classification also recognizes GCN APUs independently of CDNA;
a numerically higher gfx90c value must not select MI100-class MFMA assumptions.
Modern ROCm support here is a project-maintained deployment extension, not an
AMD official-support claim for legacy products.

### Legacy HIP mmap workaround

**For HIP on `gfx900`, `gfx906` and `gfx90c`, disabling mmap is usually
necessary.** The investigation points to a probable issue in the KFD driver
path within amdgpu; the driver root cause has not been conclusively confirmed
or fixed. Mapped model loading can be very slow or appear to stall. The
maintainer has verified disabling mmap as a working workaround on affected
hardware; this is not certification of every listed GPU or of the mmap path.

Set the following in the affected model's existing `[runtime]` table, retaining
its other settings:

```toml
[runtime]
use_mmap = false
```

Alternatively, **use Vulkan instead of HIP** if the installed Vulkan driver
exposes a usable compute device. Select the actual native device reported on
that host; for example:

```toml
[runtime]
device = "Vulkan0"
```

These are alternative profile edits, not two tables to paste into one TOML.
Reload the model after changing its profile. The setting is per model; TLlama
does not automatically disable mmap for every AMD device. gfx906/gfx90c library
checks do not qualify all GGML inference, and gfx900 has no completed project
hardware acceptance. Accepted MI50/Vulkan captures remain a separate result;
the unaffected MI100/gfx908 HIP captures do not settle this legacy driver issue.

This documented workaround is the agreed 1.0.0 disposition. Driver investigation
continues separately and does not block that release. It is unrelated to the
original P100 sparse mmap memory-accounting report.

The public packaging main snapshot inspected during this review predates the
legacy additions. Use the maintained development package series identified
above; an older build that enumerates `ROCm0` can still lack the required kernels.

## AMD: other included ROCm targets

The complete development target inventory is grouped below. It describes the
package kernel targets, not a claim that every retail SKU in each family has
been tested or that every matching BLAS library is installed by default.

| Family | Included target names | Representative hardware |
| --- | --- | --- |
| Legacy GCN/Vega | gfx900, gfx906, gfx90c | See legacy table above |
| CDNA | gfx908 | Instinct MI100 |
| CDNA2 | gfx90a | Instinct MI210, MI250/MI250X |
| CDNA3 | gfx942 | Instinct MI300A/MI300X, MI325X |
| CDNA4 | gfx950 | Instinct MI350X/MI355X |
| RDNA | gfx1010, gfx1011, gfx1012 | Navi 10/12/14 families |
| RDNA2 | gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036 | Navi 2x and selected RDNA2 APUs |
| RDNA3 | gfx1100, gfx1101, gfx1102, gfx1103 | Navi 3x and Phoenix-class APUs |
| RDNA3.5 | gfx1150, gfx1151, gfx1152, gfx1153 | Selected newer Ryzen integrated graphics |
| RDNA4 | gfx1200, gfx1201 | Radeon RX 9000 / Radeon AI PRO R9000 families with these targets |
| Dedicated gfx1250 path | gfx1250 | Instinct MI455X |

There are **28** targets. This is not every AMD ISA target: for example gfx803,
gfx902, gfx940 and gfx941 are absent from this package list. A different source
build may choose a different list and needs its own library/runtime validation.

### Instinct MI455X and the gfx1250 correction

The relevant new-device fix is **MI455X/gfx1250**. It is distinct from
**MI350X/MI355X/gfx950**, which use the CDNA4 path. AMD's ROCm release notes
identify MI455X with gfx1250; the native source uses that target identity.

The fork separates gfx1250 from the generic RDNA4 classification. Its kernels
retain the appropriate wave layout while avoiding unsupported RDNA4 WMMA/dot2
assumptions; the half-product helper retains FP32 products/accumulation for this
target. This prevents the incompatible instruction path that caused build
failures. It is an implementation/build fix; no local MI455X inference capture
has been accepted. See [source evidence](feature-source-audit.md).

## NVIDIA: choose the CUDA plugin by hardware generation

Multiple versioned CUDA packages can coexist. They provide alternatives for
the **one default CUDA plugin** loaded by a runtime process; installing all
three does not make that process use three CUDA versions simultaneously.

| Package | Included compilation targets | Intended family coverage |
| --- | --- | --- |
| `tllama-server-backend-cuda12.4` | 50-virtual, 60, 61-virtual, 70-virtual, 75-virtual, 80-virtual, 86-real, 89-real, 90-virtual | Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper |
| `tllama-server-backend-cuda12.9` | All the 12.4 targets above, plus 120a-real and 121a-real | Retains older families; adds architecture-specific Blackwell/GB10 images |
| `tllama-server-backend-cuda13.3` | 75-virtual, 80-virtual, 86-real, 89-real, 90-virtual, 120a-real, 121a-real | Turing and newer covered targets; excludes Maxwell/Pascal/Volta |

The target notation is CMake's `CMAKE_CUDA_ARCHITECTURES`: `-real` produces
machine code, `-virtual` PTX, and no suffix requests both. PTX JIT and binary
compatibility still depend on the installed NVIDIA driver. The package's
architecture-specific Blackwell images are sm_120a/sm_121a; this is not a claim
that dedicated sm_100/sm_103 images or every data-center Blackwell SKU were built
and qualified.

| NVIDIA family | Typical compute capability | Representative products | Plugin choice |
| --- | --- | --- | --- |
| Maxwell | 5.x | GTX 750/900, Tesla M40/M60 | CUDA 12.x |
| Pascal | 6.0/6.1 | Tesla P100/P40/P4, GTX 10 series | CUDA 12.x; P100/CUDA12.9 has accepted project captures |
| Volta | 7.0 | V100, Titan V | CUDA 12.x |
| Turing | 7.5 | T4, RTX 20, GTX 16 | 12.x or 13.3 |
| Ampere | 8.0/8.6 | A100, RTX 30, RTX A-series | 12.x or 13.3 |
| Ada Lovelace | 8.9 | RTX 40, L4/L40 | 12.x or 13.3 |
| Hopper | 9.0 | H100/H200 | 12.x or 13.3 |
| Blackwell / GB10 targets | 12.0/12.1 | RTX 50, RTX PRO Blackwell, DGX Spark/GB10 | 12.9 or 13.3 for the included specific images |

CUDA 13 removed offline compilation/library support for Maxwell, Pascal and
Volta. A recent driver cannot restore those toolkit targets: retain a CUDA
12.x plugin for such cards. Kepler/sm_3x is not targeted by these packages.
Product families with reused marketing names must be checked against NVIDIA's
actual compute-capability list. The reviewed Debian plugins are amd64 packages;
this table does not certify ARM Jetson builds.

Select a suitable installed plugin:

```sh
sudo update-alternatives --config tllama-server-cuda
```

Alternatives prefers the highest installed priority, so an older GPU may need
an explicit 12.x choice when 13.3 is also installed. CUDA13.3 package metadata
requires `libcuda1 >= 580`; compatibility with a particular GPU, toolkit/PTX
image and driver still needs to be checked. Core/backend package versions must
match. CUDA graphs are built in but can be disabled natively on older hardware;
that diagnostic alone does not mean inference is unsupported.

## Intel: SYCL, oneAPI and Level Zero

`tllama-server-backend-sycl` builds Intel-target SYCL with FP16, graph support,
host-memory fallback, Level Zero, oneMKL and oneDNN. Portable device compilation
avoids restricting the package to the builder's one GPU. A compatible Intel
compute driver and matching oneAPI runtime dependencies remain necessary.

The pinned native SYCL guide identifies these families and upstream-tested
examples; they are **not new TLlama hardware acceptance results**:

| Intel family | Examples in the native guide |
| --- | --- |
| Data Center GPU Max | Max 1100/1550 |
| Data Center GPU Flex | Flex 170 |
| Arc A-series / Alchemist | A770, A750, A730M |
| Arc B-series / Battlemage | B580 |
| Integrated Arc | Meteor Lake, Arrow Lake, Lunar Lake |
| Supported Core integrated GPUs | 11th generation and newer examples, subject to oneAPI's device list and FP16 capabilities |

An old Intel iGPU is not automatically SYCL compatible. OpenCL is a separate
source-build option, not part of this Intel SYCL package. The pinned backend
targets Intel; the generic SYCL standard does not imply this package supports
AMD or NVIDIA GPUs. Instella's historical SYCL Q8_0 key-expansion issue remains
unqualified. No local SYCL inference suite was accepted in the 0.7.0 audit.

## CPU execution and helper libraries

CPU-only execution selects `device = "none"`. TLlama forces zero GPU layers and
disables operation, K/V and projector offload. Native CPU kernels and installed
CPU helper libraries remain available. Selecting CPU does not require disabling
or uninstalling accelerator packages.

The Debian CPU package includes portable x86-64 variants plus OpenBLAS. Runtime
feature detection selects an appropriate available variant rather than assuming
the build host's instruction set:

| CPU capability group | Native variant examples |
| --- | --- |
| Baseline and SSE4.2 | x64, sse42 |
| AVX / F16C / FMA | sandybridge, ivybridge, piledriver |
| AVX2 / BMI2 | haswell |
| AVX-512 and extensions | skylakex, cannonlake, cascadelake, icelake, cooperlake, zen4 |
| AVX-VNNI | alderlake |
| AMX-enabled variant | sapphirerapids, subject to CPU/OS support |

These are feature profiles, not a rule that only CPUs with those codenames work.
Thread counts, batch threads, NUMA and mmap/mlock can be configured per model.

Source builds also contain ARM/NEON and optional dot-product/FP16/SVE/SVE2/SME
paths, Apple M-series variants/Accelerate, Power/VSX, s390x and RISC-V/RVV CPU
paths. They are inherited native build capabilities, not a promise of available
TLlama binary packages or tested service operation on every architecture.

The native BLAS build can integrate OpenBLAS, BLIS/AOCL, Intel MKL, Apple
Accelerate and other supported CMake BLAS providers. A separate native ZenDNN
backend is also present. The reviewed Debian default is **OpenBLAS**; MKL,
BLIS/AOCL, Accelerate and ZenDNN are not all bundled or selected by TLlama's
backend preference list. Helper libraries are build/deployment choices, not
device selectors or separate inference engines.

## Other native build options

The pinned native source additionally contains CANN, MUSA, OpenCL, OpenVINO,
WebGPU, Hexagon, zDNN, RPC, VirtGPU and ET backends, as well as Metal. TLlama's
selector policy recognizes their naming conventions where possible.
This is source integration, not a claim of shipped Debian plugins, configured
remote endpoints or completed inference on those devices. RPC/virtual devices
need their own runtime setup. New native flags do not automatically become
public TLlama settings.

## Evidence and references

The accepted 0.7.0 matrix includes MI50/Vulkan, MI100/HIP and P100/CUDA12.9 for
the documented text/vision/memory/lifecycle/MTP/tools recipes. Embeddings and
Decision have narrower MI50/Vulkan acceptance. See [runtime audit](runtime-audit-070.md)
for models, cache/context settings and remaining limitations.

Primary hardware references, checked 27 October 2026:

- [AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
  and [LLVM AMDGPU processor table](https://rocm.docs.amd.com/projects/llvm-project/en/docs-10.0.0/LLVM/llvm/html/AMDGPUUsage.html).
- [AMD ROCm 10.0 notes identifying gfx1250/MI455X](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html).
- [NVIDIA current compute capabilities](https://developer.nvidia.com/cuda/gpus),
  [legacy capabilities](https://developer.nvidia.com/cuda/gpus/legacy) and
  [CUDA 13 removed architectures](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).
- [Pinned native SYCL guide](https://github.com/ByCzech/llama.cpp/blob/0e9433bfd33d8cdb194285b70c0d421d39135df2/docs/backend/SYCL.md).
