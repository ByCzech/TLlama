# Packages, installation and platform roadmap

TLlama has two separately versioned parts: the Python daemon/client and the
native `tllama-server` runtime with its backend modules. A Python install does
not build GPU plugins or download model weights. See [README](../README.md) for
source installation and [hardware](hardware.md) for selecting a backend.

## Debian packages

Debian runtime packages have already been built and used for project hardware
qualification. The accepted development set is `0.7.0~dev+b11850.1-1`, using
native `b11850-0e9433bfd3`. The native project has modular Debian packaging:

| Package | Purpose |
| --- | --- |
| `tllama-server` | Metapackage installing core and CPU support |
| `tllama-server-core` | Private executable and common libraries |
| `tllama-server-backend-cpu` | Portable CPU variants and OpenBLAS |
| `tllama-server-backend-vulkan` | System-driver Vulkan compute |
| `tllama-server-backend-rocm10.1` | AMD HIP/ROCm 10.1 backend |
| `tllama-server-backend-cuda12.4` | CUDA route retaining older NVIDIA families |
| `tllama-server-backend-cuda12.9` | CUDA 12.x plus included Blackwell images |
| `tllama-server-backend-cuda13.3` | Newer NVIDIA targets, Turing onward |
| `tllama-server-backend-sycl` | Intel oneAPI/Level Zero backend |

Core and backend modules require exactly matching package versions; the
internal ggml module ABI is not assumed stable between snapshots. Versioned
accelerator modules and alternatives allow several SDK package generations to
coexist. CUDA's default must still suit the GPU: Pascal P100 needs CUDA12.x,
even if a higher-priority CUDA13 plugin is installed.

Legacy gfx900/gfx906/gfx90c deployment additionally uses the matching optional
`amdrocm-blas10.1-gfx900`, `amdrocm-blas10.1-gfx906` or
`amdrocm-blas10.1-gfx90c` add-on. These project-maintained libraries complement
the native kernels; they are not Python dependencies.

The default native executable is
`/usr/lib/tllama/server/tllama-server`. Common libraries/backend loader links
live in that private runtime tree, with versioned accelerator modules in its
backend subdirectories. Relative runtime paths keep native dependencies separate
from the Python application. TLlama can select another absolute runtime path.

For a configured APT source that supplies these exact package names:

```sh
sudo apt install tllama-server
# Add the backend appropriate for this machine, for example:
sudo apt install tllama-server-backend-vulkan
```

This example does not add an APT source or claim the development packages are
in Debian's official archive. Use the maintainer's configured package source
or locally built matching `.deb` files. Do not infer repository URLs, signing
keys or public 1.0.0 publication from package naming. Native source/build profiles
are maintained in [tllama-server](https://github.com/ByCzech/tllama-server).
Its public main snapshot may predate the applied development series.

The Python application can be installed from source with the instructions in
[README](../README.md); the supplied [systemd unit](../systemd/tllamad.service)
has a [manual setup guide](systemd.md). Completing the application package and
release installation flow remains separate from the already-built native
runtime packages. This documentation does not announce a released 1.0.0 package.

## Planned builds and installers

The distribution direction is ready-to-use builds and installers for **Linux,
Windows and macOS**, alongside Debian packages. These are planned deliverables,
without a promised release date or claimed published download:

| Platform | Present basis | Planned distribution work |
| --- | --- | --- |
| Linux | Source installation, supplied systemd unit and exercised Debian native packages | Finished application/native package integration and convenient builds/installers |
| Windows | Native upstream backend/toolchain capabilities and configurable runtime path | TLlama builds/installers, dependency bundling and platform lifecycle/transport qualification |
| macOS | Native CPU/Accelerate/Metal capabilities | TLlama builds/installers and platform/backend qualification |

Native portability is not equivalent to completed TLlama platform support.
The current worker defaults use a Linux runtime path and private POSIX/UDS
ownership mechanisms; Windows service/process/transport handling needs explicit
implementation and testing. Metal is not in the default automatic backend list,
so a macOS build also needs an appropriate explicit selector or policy.

Installer formats, CPU architectures, signing/notarization, update channels,
bundled SDK versions and release dates remain to be selected and validated.
No MSI, DMG, universal binary, self-updater or cross-platform feature parity is
promised by the present source tree. Track final release readiness in the
[review record](release-review-100.md).
