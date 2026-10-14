# Private native worker (development)

The worker adapter is being developed on `refactor/tllama-server`. It does not
replace ModelManager or route public inference requests yet.

`tllama.worker_config.WorkerLaunchSpec` describes one resolved launch without
importing llama-cpp-python. The caller supplies absolute model/projector paths,
a native device selector, load parameters and optional child environment values.
CPU, CUDA, ROCm, Vulkan and other server device names use the same contract.
Mapping public TOML/environment/CLI configuration is a later milestone.

The initial contract uses one slot, explicit context size, fit disabled and
context shifting disabled. Its default 2048 context is for standalone testing,
not a change to TLlama's public context default. KV K/V types remain independent;
unset flash attention/cache types use the native defaults.

CPU placement is `device="none"`, which disables GPU layers, operation offload
and projector offload. A GPU selector with unset layers, or legacy `-1`, maps to
native `all`; native numeric `-1` would mean auto instead. Conflicting CPU/GPU
layer settings are refused.

The supervisor owns a private cwd and API key file. The loopback server normally
uses port 0; its announced bound address is discovered from the pinned native
log. A fixed loopback port can be supplied for controlled tests. Parent native
server settings (`LLAMA_ARG_*`, `LLAMA_API_KEY`, `MTMD_BACKEND_DEVICE`, Vertex
`AIP_*`) and `GGML_BACKEND_PATH` are removed. Per-worker GPU visibility variables
can override inherited values without changing global `os.environ`.

This does not isolate SDK plugins. Installed plugin loading and system
alternatives retain their current packaging behavior; do not switch alternatives
per request or interpret `GGML_BACKEND_PATH` as an SDK selection override.

The supported native fingerprint is `b11552-ec2797836f`; changing it is an
explicit launch decision and requires confirming protocol/CLI compatibility.
An executable alone is not proof of usable kernels on every discovered device.
