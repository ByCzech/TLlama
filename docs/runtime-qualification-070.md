# Native 0.7.0 qualification captures (0118)

Target: `b11850-0e9433bfd3`, native commit
`0e9433bfd33d8cdb194285b70c0d421d39135df2`, Debian package
`0.7.0~dev+b11850.1-1`. TLlama integration through 0117 passed the user's full
suite: **3842 passed, 2 warnings**. That result verifies the adapter tests;
hardware captures for the new memory accounting remain required.

This preparation adds capture options to existing tools. It does not change
`RUNTIME_FINGERPRINT`, `SUPPORTED_RUNTIME_FINGERPRINTS`, the log grammar
allowlist, `AUDITED_RUNTIME` or the separate Decision policy. Candidate discovery
can succeed with `protocol_ok=true` while `runtime_audited=false` and
`review_required=true`: the raw evidence still needs review. It proves neither
usable kernels nor PCI-to-native device identity.

## 1. First capture: identity and devices, no model load

Run from the TLlama checkout with the same Python environment as pytest, on
each target host (P100 and MI50 first):

```bash
qualification_dir=$(mktemp -d /tmp/tllama-070-XXXXXX)
PYTHONPATH=src python3 scripts/device-inventory.py \
  --expected-fingerprint b11850-0e9433bfd3 \
  --output "$qualification_dir/devices"
dpkg-query -W -f='${binary:Package}\t${Version}\t${db:Status-Abbrev}\n' \
  'tllama-server*' > "$qualification_dir/packages.tsv" 2> "$qualification_dir/packages.stderr"
tar -cJf "$qualification_dir.tar.xz" -C "$qualification_dir" .
printf '%s\n' "$qualification_dir.tar.xz"
```

The inventory is read-only: two bounded, private `--version`/`--list-devices`
processes, no listener and no weights. The output preserves stdout/stderr,
visibility variables, parser issues, PCI evidence and cleanup state. On a version
mismatch, version evidence is saved and the device probe is not launched.
The default guard still requires the existing audited build when the candidate
flag is omitted. Drift is retained and fails the probe even after explicit opt-in.
Do not change global constants merely to get a green result.

Send the archive even if the probe fails. Check installed core/backend versions
match. The P100 needs the CUDA 12.x plugin; CUDA 13.3 is not the Pascal test.
MI50 HIP inference needs the matching gfx906 BLAS addon. Installed libraries and
device discovery alone do not qualify inference. Existing unresolved ROCm
failures remain open until their actual error/configuration is captured.

## 2. Memory/inference capture after reviewing the device table

Choose the actual native selector from the captured table. Release any loaded
service model first: this tool launches an independent worker and uses physical
GGUF paths, not a registered alias or model TOML. Keep the original model,
context, K/V, FA and offload settings with the report. Use matching settings for
mmap/no-mmap comparison. Each output path must be new.

Example command shape; substitute the physical model and the original settings:

```bash
PYTHONPATH=src python3 scripts/memory-smoke.py \
  --expected-fingerprint b11850-0e9433bfd3 \
  --profile full --device CUDA0 \
  --model /absolute/path/to/model.gguf --ctx 4096 \
  --type-k q4_0 --type-v q4_0 --flash-attn on \
  --weight-storage --locality --memory-tensor-diagnostics 4096 \
  --output /tmp/tllama-070-p100-full
```

`full` requests all GPU layers with mmap and no projector. It still records
actual mixed placement; a CPU noninput tensor does not get hidden to make the
capture pass. At least some accelerator noninput placement must be observed.
It does not require an artificial `size == size_vram` result. `no-mmap` remains
the existing full-offload profile without mmap. Use `vision` with `--mmproj` for
a projector, or `partial --gpu-layers N` to request a known partial placement.
CPU and available Vulkan runs remain useful independent profiles. CPU inference
on the large original model can be slow; record any smaller-model substitution.

`--weight-storage` requires known deduplicated payload coverage and classified
CPU/accelerator assignment. Missing, partial, malformed or unclassified evidence
fails this qualification check, with all raw output retained. This opt-in
requirement does not change production handling of partial or older V1 data.
Optional tensor diagnostics may be valid and partial because their bounded
records/ranges are not the payload total. Locality coverage is independent too.

The tools deliberately bypass model TOML. If the original case uses an independent
draft, additional runtime options or another setup that this smoke cannot express,
retain that original configuration and obtain its worker capture before comparing
sizes; do not describe a simplified run as a reproduction of the original case.
For Instella, retain the tested equal K/V quantization. Its successful thinking
and inference tests, and the earlier Nemotron success, are already recorded.

Send `raw.jsonl`, `summary.jsonl` and `worker.log` together, including failed
runs. They preserve separate raw/selected snapshots and the actual launch argv.
Inspect P100 native layers 39/40 using the query in
[memory-smoke-08c.md](../scripts/memory-smoke-08c.md#bounded-tensor-capture-0117).
Use main intervals for totals and tensor-role records only for explanation.
Native IDs cannot be joined across raw and selected observations.

## 3. Acceptance after captures

Review actual worker health/props, stream/nonstream inference, tokenization,
template rendering, cancellation and cleanup. Qualify log fallback, device
discovery and Decision independently; only the evidence for the respective
contract can justify a policy change. Capture failed load/replacement, unload
and sleep/reload transitions without stale weight/locality/diagnostic evidence.
Use the existing feature smoke tools for MTP, embeddings, vision, FIM and tools
with appropriate real models. A captured device table does not establish those
feature results or all CUDA/HIP/SYCL combinations.

Source comparison of the old pinned `3144c167ef` and target `0e9433bfd3` retains
the `--list-devices` table formatting; the target still emits the model-buffer
and layer-offload log forms understood by the reader. This source check is
preliminary evidence, not a replacement for actual SDK/runtime output. No
runtime allowlist is expanded in this preparation patch. Record hardware and
model results before the final baseline change and release checks.
