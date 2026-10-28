# TLlama documentation

Start with [README](README.md) for the overview and quick start. The guides
below describe current behavior; historical checkpoint sections retain their
original build identities and evidence.

## Overview, advantages and installation

- [Features and practical comparison](docs/features.md): per-model control,
  extensions, memory evidence, APIs, storage and process ownership.
- [Hardware and backends](docs/hardware.md): AMD targets and GPU families,
  CUDA versions, Intel SYCL, CPU variants and qualification boundaries.
- [Packages and platform roadmap](docs/distribution.md): Debian runtime packages,
  application installation and planned Linux/Windows/macOS builds/installers.
- [Usage and client integration](docs/usage.md).
- [Client examples: Python, curl and JavaScript](docs/client-examples.md): basic
  and advanced SDK/HTTP calls, streaming, async, history and errors.
- [Systemd installation and service permissions](docs/systemd.md).

## Configuration and model management

- [Daemon configuration](docs/daemon.md): server TOML, environment, CLI and inspection.
- [Model configuration](docs/configuration.md): model TOML, precedence and sampling.
- [Example server configuration](examples/server.toml).
- [Client configuration and flags](docs/cli.md#http-client-basics): layered
  system/user preferences, connection defaults and display controls.
- [Example client configuration](examples/client.toml).
- [Native configuration reference](docs/native-config.md).
- [Backend selection and fallback](docs/backend-priority.md).
- [Device inventory and selector evidence](docs/device-inventory.md).
- [Model repositories, pulling and maintenance](docs/model-repositories.md).
- [Persistent metadata and digest caches](docs/metadata-cache.md).
- [CLI, REPL and client-side imports](docs/cli.md).

## Public API and model capabilities

- [API compatibility and complete route table](docs/api-compatibility.md).
- [Request compatibility](docs/request-compatibility.md).
- [Ollama options and runtime overrides](docs/ollama-options.md).
- [Current native public API adapters](docs/native-public-api.md).
- [Tools and tool-result history](docs/native-tools.md).
- [Vision and projector integration](docs/native-vision.md).
- [FIM prefix/suffix completion](docs/native-fim.md).
- [Embedded MTP](docs/native-mtp.md).
- [Embedding API](docs/embedding-api.md).
- [Native embedding worker contract](docs/native-embeddings.md).
- [Decision API](docs/decision-api.md).
- [Native Decision worker contract](docs/native-decision.md).

## Architecture, diagnostics and development

- [Native worker transport and lifecycle](docs/native-worker.md).
- [Residency, leases and configuration generations](docs/native-residency.md).
- [Memory reporting, SIZE and PROCESSOR](docs/native-memory.md).
- [Native adapter checkpoint and ownership boundaries](docs/native-api-adapter.md).
- [Development workflow and smoke tools](docs/development.md).
- [Source evidence for feature and hardware claims](docs/feature-source-audit.md).

## Qualification and release records

- [Current native 0.7.0 runtime audit](docs/runtime-audit-070.md).
- [Runtime qualification procedures](docs/runtime-qualification-070.md).
- [Decision qualification procedures](docs/decision-qualification-070.md).
- [Pre-1.0.0 documentation review](docs/release-review-100.md).

The runtime audit distinguishes accepted hardware/model recipes from source
support and unresolved release items. A passing protocol comparison is not a
general model-quality guarantee.

## Diagnostic recipes

- [Public functional qualification script](scripts/functional-qualification-smoke.py).
- [Public API compatibility smoke script](scripts/api-smoke.py).
- [Tools API and SDK qualification](scripts/tools-api-smoke.md).
- [Native tools checkpoint](scripts/tools-smoke.md).
- [Embedding API and SDK qualification](scripts/embeddings-api-smoke.md).
- [Native embedding checkpoint](scripts/embeddings-smoke.md).
- [Decision API qualification](scripts/decision-api-smoke.md).
- [FIM protocol and content probes](scripts/fim-api-smoke.md).
- [Lifecycle qualification script](scripts/lifecycle-smoke.py).
- [Embedded MTP qualification script](scripts/mtp-qualification-smoke.py).
- [Native MTP checkpoint](scripts/mtp-smoke.md).
- [Native Decision checkpoint](scripts/decision-smoke.md).
- [Native FIM checkpoint](scripts/fim-smoke.md).
- [Multimodal embeddings checkpoint](scripts/embeddings-multimodal-smoke.md).
- [Historical memory checkpoint](scripts/memory-smoke-08c.md).
- [Memory diagnostic script](scripts/memory-smoke.py).

Use the actual script's `--help` and the qualification guide for dependencies,
fingerprints, parameters and output archives. Historical recipes are not a
request to repeat already accepted captures.
