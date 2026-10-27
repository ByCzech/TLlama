# Native b11850-0e9433bfd3 regression excerpts

Captured on the user's MI50/Vulkan, MI100/HIP and P100/CUDA12.9 hosts during
0.7.0 qualification, reviewed 25–27 October 2026. `logs.json` and `devices.json`
identify the original archives, their SHA256 and members. Complete archives
remain outside Git. Excerpts preserve native lines without Python logging
prefixes; they do not include model paths, credentials or request content.

The nine log cases select allocation/offload/projector/load-complete lines,
in original order. Expected rounded totals/status/reason were checked against
the original capture summaries before inclusion. Vision remains partial;
projector model size is not an allocated buffer. Trailing lines remain in
the excerpts to exercise the parser's freeze after load completion.

The three device cases preserve complete stdout tables and version stderr.
Free memory is a captured observation, not a reusable capacity promise.
No native selector is assigned a physical PCI identity or a verified kernel.
