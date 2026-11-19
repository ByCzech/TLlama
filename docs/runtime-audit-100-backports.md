# Native b12359.2 compatibility — TLlama 1.6.1

Reviewed 19 November 2026 against native commit
`e148b6dbe6e08911e59b60fefd8562fcf73e80a5`, tree
`f5315e1d1c46e03ae196d918bacc520c207a26cb`, fingerprint
**b12359-e148b6dbe6**, packaged as `1.0.0~dev+b12359.2-1`.
The base remains llama.cpp v1.0.0 / build 12359 plus the project's changes and
12 selected upstream backports. This is not a merge of the later upstream
master. The [original 1.0.0 audit](runtime-audit-100.md) and
[0.7.0 captures](runtime-audit-070.md) retain their original identities.

## Contract review

Compared with the previously accepted `c0819bbbae`, the following native files
are byte-identical: `common/arg.cpp`, `common/common.h`,
`common/speculative.cpp`, `common/chat.cpp`, `tools/server/server-context.cpp`,
`server-task.cpp`, `server-chat.cpp`, `server-common.cpp`,
`server-decision.cpp`, `server-memory.cpp` and `server.cpp`.
The mtmd loader/preprocessor and recurrent-memory implementation are also
unchanged. These comparisons cover the existing worker CLI, authenticated
health/props/admission, completion/chat/embedding/text Decision wire formats,
speculation controls and Memory API V1 used by TLlama.

The changed HTTP streaming provider now returns success after `sink.done()`;
this fixes connection reuse after a completed stream, without changing SSE
frames. The recurrent-state reset writes zero instead of multiplying by zero,
so a previous NaN does not survive reset. Its new internal tensor enumerator
feeds the existing deduplicated buffer enumerator. It does not change the
Memory API schema, owners or storage/locality collectors.

Other backports affect GCN MMQ/Stream-K, HIP MoE scheduling, AMD odd-column
MUL_MAT_ID, CDNA GDN, Vulkan matrix/attention kernels and SYCL attention masks.
GPU performance and numerics remain model/device dependent. Native provenance
is recorded in `docs/development/tllama-post100-backports.md` at the exact pin.
No additional native model/audio APIs, scheduler migration or DFlash refactor
are part of this set.

## Exact identity admission

| Contract | Default identity | Retained identities |
| --- | --- | --- |
| Generation and embeddings | b12359-e148b6dbe6 | b12359-c0819bbbae; b11850-0e9433bfd3; b11552-ec2797836f; b11552-d1570abd3d; b11552-3144c167ef |
| Text Decision: Laya, Nimble, Clef | b12359-e148b6dbe6 | b12359-c0819bbbae; b11850-0e9433bfd3; b11552-3144c167ef |
| Discovery version/table grammar | b12359-e148b6dbe6 | b12359-c0819bbbae; b11850-0e9433bfd3; b11552-3144c167ef |
| Initial load-log coverage | No new certification | Independent historical log audit set is unchanged |

`warn` still warns on an unreviewed build and `strict` refuses it. Explicit
diagnostic pins select exactly one identity. Accepted identities still require
valid health/alias/context/slots, live modalities and typed response parsing.
The broader generation set does not expand the narrower Decision set.
Memory reports must match their worker's observed identity and valid schema.
The new build is not added to the independent rounded-MiB log audit set;
`runtime_verified=false` there is not a failure of native Memory API admission.

## Evidence and limits

The maintainer reports the complete native package build, installed ELF and
identity checks, and successful TLlama inference on AMD APU RADV RENOIR with
both Vulkan and ROCm 10.2. The matching gfx900/gfx906/gfx90c BLAS add-ons are
reported built. This is not qualification of every GPU, model, long context,
projector, Decision head or speculative recipe. Native CPU/Memory/HTTP tests
and the original hybrid-state regression belong to the native build record;
they are not represented as a fresh native build performed by this app update.

Application regression tests use owned authenticated TCP stub processes for
new/retained strict identities, unknown identities, live protocol failures and
native Memory API selection. They exercise real Python IPC and cleanup, but
not native model inference. Production UDS remains the default. Tests cannot
establish an MTP speedup; compare the same weights/context/KV/FA/sampling with
speculation off and on on each target backend.

The existing device inventory and functional qualification tools accept the
new exact identity. For example, in a new capture directory:

```sh
PYTHONPATH=src python3 scripts/device-inventory.py \
  --executable /usr/lib/tllama/server/tllama-server \
  --expected-fingerprint b12359-e148b6dbe6 --output /tmp/tllama-161-devices
```

Legacy HIP qualification retains matching ROCm 10.2 add-ons and the existing
`--load-mode none` workaround. MI100 has no Vulkan compute in the maintainer's
current setup; P100 retains CUDA 12.x. APU HIP/Vulkan capacity reports are not
interchangeable physical-memory measurements; no capacity is added by this
update. See [distribution](distribution.md) for the native package transition.

## Release boundary

TLlama 1.6.1 changes the default audited build and documents existing APIs/SDK
limits. The Debian runtime floor stays `1.0.0~dev+b12359.1-1` because its
previously accepted contracts remain supported; `.2-1` is the current recommended
native set. All installed native core/backend packages must match each other.
No application HTTP route, request/response field, model role, configuration
key, default sampler/KV/load policy, repository format or cache schema is added.
Larger audio, SDK, model-definition and other feature work is planned for 1.7.0.
