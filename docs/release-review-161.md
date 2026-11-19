# TLlama 1.6.1 — native compatibility and documentation

Selected scope: blocks 01 and 02 of the next-release plan. Larger features are
reserved for 1.7.0. This source preparation does not publish a tag, upload
packages or certify every model/backend combination.

## Changes

- Default reviewed native identity `b12359-e148b6dbe6`, package set
  `1.0.0~dev+b12359.2-1`: ROCm 10.2 and selected post-v1.0 native fixes.
- Retain previously reviewed generation/embedding, text Decision and discovery
  identities in their separate audit sets. Unknown identities and malformed
  live protocols keep their existing warn/strict and validation behavior.
- Correct the API table: generate supports configured inline images.
- Document reviewed SDK limitations and raw HTTP access to extended statistics,
  show/ps and binary uploads. No own SDK or new HTTP field is introduced.

The [native audit](runtime-audit-100-backports.md) records the exact source
comparison, Memory API V1 boundary, retained builds and evidence limits. Native
kernel fixes already belong to the separately built server. This Python release
does not rebuild those kernels or promise a fixed MTP/backend speedup.

## Upgrade

No repository conversion, model reimport, deliberate rehash or cache reset.
Repository format 2, metadata 9, digest 2 and cache envelope 2 are unchanged.
The `[llm]`/`[mmproj]`/`[draft]` definition syntax, sampler/KV/load defaults,
public API, production UDS transport, systemd lifecycle and package split stay
unchanged. The Python version continues to come from distribution metadata.

The application retains native floor `1.0.0~dev+b12359.1-1`; the earlier build
is still supported. The recommended current set is `.2-1`. Native core and all
installed backend packages must have matching package versions. When upgrading
the native set, explicitly replace the old HIP package with ROCm 10.2; see
[distribution](distribution.md). Matching legacy 10.2 add-ons and the existing
load-mode workaround remain separate native deployment choices.

## Release checks

Run the focused identity/memory/discovery checks first:

```sh
pytest -q tests/test_runtime_fingerprint.py tests/test_native_memory.py \
  tests/test_native_memory_api.py tests/test_device_inventory.py
```

Then the ordinary integration and packaging checkpoint:

```sh
pytest -q
pytest -q debian/tests/test_smoke_paths.py
debuild -us -uc -b
```

On a checkout without the release tag, build the wheel/sdist candidate with
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TLLAMA=1.6.1`. The package smoke must import
the built wheel/code and matching metadata, not an earlier checkout/install.
Record environment exclusions separately: a TCP substitute or a synthetic
worker is not proof of production UDS or real GPU inference.

The maintainer has already reported installed native APU Vulkan/ROCm inference.
Further representative text/tools/vision/embeddings/Decision/speculation and
Memory API checks can use the existing qualification scripts with the new
exact fingerprint. Do not repeat all native builds or the whole model catalog
without a new failure or changed scope. Tag/publication follows the maintainer's
final application test and Debian-build checkpoint.
