# Packages, installation and platform roadmap

TLlama has two separately versioned parts: the Python daemon/client and the
native `tllama-server` runtime with its backend modules. A Python install does
not build GPU plugins or download model weights. See [README](../README.md) for
source installation and [hardware](hardware.md) for selecting a backend.

## Debian packages

TLlama 1.3.1 builds as Python wheel/source archives and separate Debian
application/documentation packages (`1.3.1-1`). The maintainer's APT archive
supplies published Debian releases. See the [changelog](../CHANGELOG.md) for
changes and upgrade notes. Application and native runtime versions are
independent.

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
sudo apt install tllama
# Add the backend appropriate for this machine, for example:
sudo apt install tllama-server-backend-vulkan
```

The `tllama` application depends on the separate `tllama-server` runtime.
It installs `/etc/tllama/server.toml` and the `tllamad` systemd service, using
the `tllama` service account. See the [Debian package notes](../debian/README.Debian)
for service behavior, permissions and local-unit upgrades.

The optional `tllama-doc` package installs guides and client/server examples
under `/usr/share/doc/tllama/`, including `examples/`. From 1.2 the application
itself installs `/etc/tllama/client.toml` as a dpkg-managed conffile; example
values are commented out and built-in defaults apply. Local edits survive
upgrades through normal dpkg conffile handling. User preferences go in
`$XDG_CONFIG_HOME/tllama/client.toml` (normally `~/.config/tllama/client.toml`).
The client also works with no configuration file. See the
[client configuration rules](cli.md#http-client-basics).

This example does not add an APT source or claim the packages are in Debian's
official archive. Use the maintainer's configured package source or locally
built matching `.deb` files. Native source/build profiles
are maintained in [tllama-server](https://github.com/ByCzech/tllama-server).
Its public main snapshot may predate the applied development series.

The Python application can be installed from source with the instructions in
[README](../README.md); the supplied [systemd unit](../systemd/tllamad.service)
has a [manual setup guide](systemd.md). Building the Python wheel or source
archive does not build or bundle the native runtime.

## Planned builds and installers

The distribution direction is ready-to-use builds and installers for **Linux,
Windows and macOS**, alongside Debian packages. These are planned deliverables,
without a promised release date or claimed published download:

| Platform | Present basis | Planned distribution work |
| --- | --- | --- |
| Linux | Python wheel/source archive, Debian application/native packages and systemd service | Additional convenient builds/installers |
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
promised by the present source tree. Release changes are tracked in the
[changelog](../CHANGELOG.md); the [1.0.0 review record](release-review-100.md)
retains its historical review and qualification decisions.
