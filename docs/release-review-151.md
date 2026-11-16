# TLlama 1.5.1 — native compatibility release

This release adopts the available `tllama-server` package set
`1.0.0~dev+b12359.1-1`, native `b12359-c0819bbbae`, based on llama.cpp v1.0.0
with the project's changes. It implements only item 1 of the post-1.5.0 work
plan. The application version is independent of the native version.

## Scope

- Update the ordinary worker, existing Laya/Nimble/Clef Decision and discovery
  reference identities without relaxing warn/strict policy or live checks.
  Keep previously accepted runtime identities for source installations.
- Keep exact Memory API V1 reporting. Additional layer-input/NextN lines must
  not leave old rounded load-log totals marked complete. The new native
  identity is not automatically added to the independent log audit set.
- Require the new runtime package version in Debian and prepare coordinated
  application/native publication. Install matching core/backend versions;
  P100 retains a CUDA 12.x backend. No GPU module is selected automatically
  merely because its package has a newer SDK number.
- Retain model definitions, weights, .inf companions, identities, repository
  format, cache versions and all existing inference defaults and public APIs.
  No migration, reset or reimport is required.

## Evidence and remaining checks

The [native audit](runtime-audit-100.md) records the exact source contract and
feature-specific identities. Python tests cover strict new/legacy admission,
unknown fingerprints, protocol drift, native V1 ownership/selection, CLI
construction and controlled transport scenarios. Stubs prove adapter behavior,
not numerical correctness of the actual native binary.

The previous native build work includes the complete package build, actual
SYCL SDK/ELF checks and native Memory API HTTP regression evidence. It does not
replace an application smoke using the installed candidate and the maintainer's
real GGUFs. Before final release, record a clean application test run, Debian
package smoke/build and representative actual model checks on available HW.
Do not claim unavailable hardware was tested or require a new catalogue of
large downloads solely to certify this compatibility update.

For a registered text model, an existing diagnostic performs native/public
comparisons, stream, cancellation/recovery and memory checks in a private
capture. Substitute the registered model and device names for the actual host:

```sh
PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile text --model 'YOUR_TEXT_MODEL' --models-dir /var/lib/tllama/models \
  --device Vulkan0 --expected-fingerprint b12359-c0819bbbae \
  --output /tmp/tllama-151-text
```

The output path must not exist. For an existing compatible embedded-MTP model,
the same text diagnostic can add `--require-mtp`. Existing tools/embedding
profiles, `decision-api-smoke.py`, `vision-smoke.py` and DFlash/state diagnostics
provide the other selected checks; use their documented model/context options
and this exact fingerprint where supported. Do not run competing large model
loads without first making room on the selected device.

## Deferred to 1.6.0 work

A5 configured-root/relative identity unification, startup/cache profiling,
measured optimization/A6, external MTP management and per-device residency
limits remain later work, subject to their existing detailed agreements.
MoE cache, FP8, tensor parallelism and new HTTP contracts still require their
own scope decisions. The compatibility release does not activate them.

The maintainer's selected checks, full pytest result, Debian build and final
tag/publication are release checkpoints, not assertions made by this document.
