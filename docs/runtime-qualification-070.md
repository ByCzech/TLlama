# Native 0.7.0 qualification captures (0118–0119)

Target: `b11850-0e9433bfd3`, native commit
`0e9433bfd33d8cdb194285b70c0d421d39135df2`, Debian package
`0.7.0~dev+b11850.1-1`. TLlama integration through 0118a passed the user's full
suite: **3866 passed, 2 warnings**. Memory full/partial/vision captures have been
reviewed on MI50/Vulkan, MI100/HIP and P100/CUDA12.9; lifecycle and remaining
feature qualification are separate, as described below.

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

Like `full`, `vision` accepts actual mixed CPU/GPU weight placement while
requiring at least some accelerator noninput weights and a projector. CPU
weights remain in the report and display; this check does not assert 100% GPU.
This also applies when a diagnostic caller enables automatic fitting through
`WorkerLaunchSpec`. The default CLI launch still uses fixed offload with fit off.

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

## 3. Worker lifecycle capture (0119)

After applying 0118a, the user's complete suite passed **3866 tests, 2 warnings**.
Full/partial/vision captures have been reviewed on MI50/Vulkan, MI100/HIP and
P100/CUDA12.9. P100 automatic fitting also produced valid mixed CPU/GPU weights.
The original sparse mmap-span subtraction has not been reproduced on hardware;
more context/memory pressure is not requested. These results do not qualify the
lifecycle or other features below.

`lifecycle-smoke.py` uses the production `NativeResidency` supervisor and
`NativeWorker` against the installed executable, with physical GGUF paths and
no registered model/TOML edits. It is a private diagnostic, not a daemon HTTP API
test. Its exact fingerprint is mandatory and checked in strict mode, independently
of the daemon's version policy. No production allowlists/defaults are changed.
It performs, sequentially:

1. Load; health/props startup checks; template rendering and tokenization;
   chat/completion, each nonstream and stream; cancel a long chat stream after
   actual content/reasoning; successfully infer again.
2. Unload during another active long stream. The held lease must keep that
   generation alive and draining. Cancel the stream, release the lease and
   require removal from reporting plus process/private-directory cleanup.
3. Reload as a new generation with fresh telemetry. While its lease is held,
   request a replacement using a tiny intentionally invalid GGUF created only
   inside the new capture directory. Require failed startup, a positive failure exit code,
   cleanup and absence of the failed generation from reporting. The old draining
   lease must still infer and retain its own evidence, then close on release.
4. Recover as a fresh generation. Only this diagnostic launch changes the native
   `--sleep-idle-seconds` from the production `-1` to `5`. Wait for `/props` to
   report sleeping; require known empty buffers, weight intervals, locality
   allocations/bindings and tensor records, with an advanced snapshot sequence.
   Repeated memory/props reads must leave it asleep. A chat request must reload
   it in the same process with a newer sequence and valid nonempty evidence.
5. Final unload/shutdown, requiring every owned process and private directory to
   be cleaned, including failed attempts. Raw errors and partial captures remain
   available on failure.

This means **three successful initial loads plus one native wake/reload** and
one tiny failed load. No two valid models are loaded concurrently. The invalid
GGUF failure is expected and is recorded as a successful negative check only
when the runtime actually rejects it and cleanup succeeds; a timeout, crash
or forced termination is not accepted as that rejection. Original weights
are never modified. TLlama's ordinary idle expiry closes the worker; native
sleep/wake here is a separate protocol check, enabled only for this diagnostic.
The script does not claim testing nonstream request cancellation, arbitrary
model-to-model replacement, process crash, full prompts filling the context,
projectors, MTP, embeddings, FIM, tools, Decision or physical residency.

Start on **MI50/Vulkan0** using the already tested Q4_K_XL GGUF, ctx=4096,
K/V=q4_0, FA=on. Release the service's loaded model first, as for memory smoke.
Use its actual physical GGUF path below. Each output directory must be new:

```bash
PYTHONPATH=src python3 scripts/lifecycle-smoke.py \
  --expected-fingerprint b11850-0e9433bfd3 \
  --model /absolute/path/to/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --device Vulkan0 --ctx 4096 \
  --type-k q4_0 --type-v q4_0 --flash-attn on \
  --output /tmp/tllama-070-mi50-lifecycle

tar -cJf /tmp/tllama-070-mi50-lifecycle.tar.xz \
  -C /tmp tllama-070-mi50-lifecycle
```

Send the archive **even on failure**, including `result.json`, `raw.jsonl`,
`summary.jsonl`, `worker.log`, all `worker-N.load.log` files and the intentionally
invalid GGUF. Do not replace a failed capture with a second run in the same
output directory. `--dry-run` prints the initial and sleep argv without creating
files/starting processes. `--transport tcp` explicitly selects authenticated
loopback instead of the default private UDS; record any transport change.

After reviewing the first MI50 result, use the same command on **MI100/ROCm0**
with the same Q4_K_XL file and a new `...-mi100-lifecycle` output. On **P100/CUDA0**
use its already tested **IQ3_XXS** model and a new `...-p100-lifecycle` output.
Keep ctx=4096, matching q4_0 caches and FA; no projector or active MTP in this
step. Do not repeat the high-context pressure/vision profiles. Native rebuild
and daemon restart are not needed for this script-only patch.

Each raw response and selected telemetry observation is validated independently;
IDs and sequences are not joined across observations or generations. Sequence
advancement is checked only across sleep/wake within the same worker lifetime.
Partial valid locality and bounded tensor diagnostics are retained; malformed
extensions, missing requested diagnostics, unknown weight payload/assignment,
wrong model state or HTTP errors fail the capture. HTTP 401/503 cannot pass by
falling back to loader logs or masquerading as an unsupported endpoint. Native
state and the compatibility SIZE/PROCESSOR policy are kept separate.

## 4. Acceptance after captures

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
