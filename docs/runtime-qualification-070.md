# Native 0.7.0 qualification captures (0118–0121)

Target: `b11850-0e9433bfd3`, native commit
`0e9433bfd33d8cdb194285b70c0d421d39135df2`, Debian package
`0.7.0~dev+b11850.1-1`. TLlama integration through 0120 passed the user's full
suite: **3933 passed, 2 warnings**. Memory full/partial/vision and the 0119
lifecycle have been reviewed on MI50/Vulkan, MI100/HIP and P100/CUDA12.9.
P100 wake completed with a 600-second request timeout. Active embedded MTP
passed on all three backends: 62 checks and 12 memory snapshots per capture,
off/on parity, measured draft acceptance and cancellation/recovery, all with
mmap. Remaining public feature qualification is separate, as described below.

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

## 4. Active embedded MTP capture (0120)

After 0119, the user's complete suite passed **3903 tests, 2 warnings**.
The lifecycle captures on all three backends passed, including P100 sleep/wake
after extending the HTTP request timeout to 600 seconds. Do not repeat them.

`mtp-qualification-smoke.py` uses physical GGUF paths, the installed executable
and private workers. It bypasses registered aliases/TOML and daemon runtime
environment overrides. The expected fingerprint is mandatory and strict. It
does not change production allowlists or qualify public APIs/Decision.

The same embedded-MTP GGUF is loaded twice, sequentially: explicit `none`, then
`draft-mtp` with a positive draft limit. The first worker is closed before the
second starts. Both use ctx=4096, K/V=q4_0, FA=on, batch/ubatch=512, all GPU
layers and fit off. The default load and request timeouts are both 600 seconds.
No projector, registered-model modification, download or native rebuild is needed.

Each profile checks nonempty inference for number-copy and Python-code prompts,
nonstream/stream parity of exact content/reasoning, finish and generated count,
then closes a long stream after a live content/reasoning delta and compares the
next control answer with its pre-cancel control in the same PID. The MTP profile
requires measured drafted **and accepted** tokens in both streaming modes;
argv, model filename or an allocated MTP context alone cannot pass this check.
The off profile must have no positive draft counters. Missing MTP measurements
remain unknown and fail qualification. Inspect the saved native log for actual
early task stop: transport closure and successful reuse do not prove it alone.

Each profile saves raw HTTP and separately selected memory observations after
load, inference and cancel: **12 memory snapshots** for a complete capture.
Every observation requires known, classified weight payload and actual device
assignment, valid extensions, locality and bounded diagnostics when requested.
The active profile additionally requires the shared main/MTP model owner and
nonempty `mtp_layers` payload. Role-filtered totals are checked against the union
of unique intervals; overlapping main/MTP subtotals are never added. Tensor
diagnostic records do not supply totals. Payloads must remain unchanged within
each worker, including after cancel; raw and selected IDs are never joined.
No equality of total memory between off/on is required: MTP allocates additional
weights/context/work buffers. Sizes are backend accounting, not physical VRAM/RSS.

Off/on output comparisons are saved separately. A difference yields
`protocol_ok=true`, `review_required=true`, `ok=false` and exit 1 if the other
checks passed. This preserves possible numerical/model differences for review;
it does not silently call a changed answer a successful qualification. A normal
capture has `ok=true`; generated code correctness and broad model quality still
need human review. Short timing samples are not a performance benchmark.

Start on **MI50/Vulkan0**, using the same physical Q4_K_XL embedded-MTP GGUF
as the previous captures. Release any service-loaded model first. Run the full
pytest suite after applying 0120, then:

```bash
PYTHONPATH=src python3 scripts/mtp-qualification-smoke.py \
  --expected-fingerprint b11850-0e9433bfd3 \
  --model /absolute/path/to/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --device Vulkan0 --ctx 4096 --draft 3 --predict 128 \
  --type-k q4_0 --type-v q4_0 --flash-attn on \
  --output /tmp/tllama-070-mi50-mtp

tar -cJf /tmp/tllama-070-mi50-mtp.tar.xz \
  -C /tmp tllama-070-mi50-mtp
```

Return the entire archive even on failure: `result.json`, `raw.jsonl`,
`summary.jsonl`, `worker.log`, `off-load.log` and `mtp-load.log` when reached.
Use a new output directory on every attempt; an existing directory is refused.
`--dry-run` prints both argv without creating output or starting a process.
`--transport tcp` explicitly selects authenticated loopback instead of UDS.

After reviewing MI50, repeat on **MI100/ROCm0** with its same Q4_K_XL GGUF,
then **P100/CUDA0** with its previously tested IQ3_XXS embedded-MTP GGUF and a
new `...-p100-mtp` directory. Lower-context MTP memory requirements can still
differ from the previous non-MTP run: preserve any OOM capture before changing
settings. Do not rerun the high-context pressure/vision matrix.

The parallel legacy HIP investigation reportedly points strongly at amdgpu/KFD
handling of mmap, with disabling mmap as a workaround. That is separate from
the original sparse mmap **accounting** case and from MI100/gfx908 qualification;
this patch neither diagnoses the kernel nor closes that issue. If this problem
blocks a HIP capture, add **`--no-mmap`** to both off/on diagnostic launches
using a new output directory. The actual native option is `--load-mode none`,
saved in argv; `result.json` also records `mmap=false`. Such a result qualifies
only that explicit no-mmap configuration. Do not silently change production
defaults or describe no-mmap success as validation of the mmap path.

Each raw response and selected telemetry observation is validated independently;
IDs and sequences are not joined across observations or generations. Sequence
advancement is checked only across sleep/wake within the same worker lifetime.
Partial valid locality and bounded tensor diagnostics are retained; malformed
extensions, missing requested diagnostics, unknown weight payload/assignment,
wrong model state or HTTP errors fail the capture. HTTP 401/503 cannot pass by
falling back to loader logs or masquerading as an unsupported endpoint. Native
state and the compatibility SIZE/PROCESSOR policy are kept separate.

## 5. Public functional qualification (0121)

`scripts/functional-qualification-smoke.py` runs the existing public probes
inside an isolated diagnostic manager, with one registered model per capture.
It pins the candidate fingerprint strictly before admission and checks the
observed fingerprint for exact equality. This also refuses drift between old
otherwise-supported fingerprints. It does not change production allowlists.
Decision models are refused and require a separate qualification.

| Profile | Existing reference | Evidence |
|---|---|---|
| `text` | `api-smoke.py` | Ollama chat/generate, raw generate, OpenAI chat, stream/nonstream, JSON schema, public ps, cancel/recovery and unload |
| `text --require-mtp` | Same | Additionally native measured drafting and all eight public result/stream counter projections |
| `tools` | `tools-api-smoke.py` | OpenAI/Ollama tool assembly, parallel/escaped arguments, followup histories, partial-argument cancel and recovery; no tools are executed |
| `fim` | `fim-api-smoke.py` | Supported suffix, independent full-prompt native reference, token counts, long input, stream/nonstream and recovery |
| `fim-unsupported` | Same | Explicit unsupported suffix response and ordinary completion recovery on a genuinely unsupported model |
| `embeddings` | `embeddings-api-smoke.py` | Three public text embedding endpoints against direct native vectors, float/base64, token arrays, usage, bounds/errors, transport close/reuse and unload |

The wrapper uses the ordinary metadata/configuration and public adapters. Its
diagnostic manager applies explicit request-level runtime overrides and adds
strict identity/transport/telemetry settings to the immutable resolved launch
before residency creates the worker. It does not mock API responses or replace
production tool/FIM/embedding consumers. All public requests use loopback TCP;
`--transport uds|tcp` selects the **private native** connection only.

Every worker has independently captured raw/selected memory after load and
before close, exact fingerprint, genuine device assignment, valid payload,
locality and bounded diagnostics, unchanged payload, and known SIZE/PROCESSOR.
Missing/invalid memory evidence fails qualification; log fallback cannot pass.
Native and public failures remain in `raw.jsonl`, `summary.jsonl`, `result.json`
and `worker.log`, with `qualification-N-load.log` for each owned worker.

Registered TOML and GGUF files are read only, checked by stat identity before
and after use (not a full-content hash of all shards). Metadata caches and the
embedding profile's temporary TOML stay under `--output`. Server runtime and
sampling overrides are not inherited; the model's template/sampling/system/FIM
recipe still applies to generation profiles. Choose a text-only definition
without a projector. Embeddings use the reference's text-only temporary
definition: no projector/template/system/FIM/sampling settings, spec=none,
F16 K/V, FA=off, ctx=256, batch/ubatch at least 512. Generation profiles default
to ctx=4096, q4_0 K/V, FA=on, batch/ubatch=512, MTP off unless explicitly
requested by `text --require-mtp`. Device, mmap and full offload are explicit.
Keep the default mmap on Vulkan and MI100/HIP; `--no-mmap` is optional for a
HIP problem and records a different tested configuration.

First apply 0121 after 0120 and run the whole pytest suite. Start on MI50/Vulkan0.
Use `tllama list` to identify the **registered reference**, not a GGUF path.
Use the existing Qwen3.6 embedded-MTP Q4_K_XL reference for text/tools.
Set `MODEL` below to that exact reference. With an external model store, pass
its actual path through `--models-dir`; otherwise the configured storage path
is used. No registered model is created or edited by the qualification.

```bash
tllama list
# Set this to the exact registered Qwen3.6 reference from that list:
MODEL='REPLACE_WITH_REGISTERED_QWEN_REFERENCE'

PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile text --model "$MODEL" --device Vulkan0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --require-mtp --dry-run \
  --output /tmp/tllama-070-mi50-public-text

PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile text --model "$MODEL" --device Vulkan0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --require-mtp \
  --output /tmp/tllama-070-mi50-public-text

tar -cJf /tmp/tllama-070-mi50-public-text.tar.xz \
  -C /tmp tllama-070-mi50-public-text

PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile tools --model "$MODEL" --device Vulkan0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --predict 256 \
  --output /tmp/tllama-070-mi50-public-tools

tar -cJf /tmp/tllama-070-mi50-public-tools.tar.xz \
  -C /tmp tllama-070-mi50-public-tools
```

Send both captures even on failure. `--dry-run` resolves the actual model and
prints the pinned command without starting a worker or creating the capture;
any temporary metadata cache is removed. It is not a hardware pass. Existing
output directories are refused by actual runs. Load/request timeouts default
to 600 seconds, including P100's slow reload; no rebuild/restart is needed.

Do not use the chat Qwen model as a positive FIM/semantic-embedding model.
After reviewing text/tools and the available references, select a supported
code GGUF (e.g. an already imported codegemma/starcoder2/qwen-coder with a
verified FIM recipe), a genuinely unsupported FIM control, and an already
imported pooled embedding model such as MiniLM or Qwen3-Embedding. Reuse the
same command with `--profile fim`, `fim-unsupported` or `embeddings`, the
appropriate model, and a fresh output. Embeddings default to ctx=256; omit
generation `--ctx 4096`/`--require-mtp` options when switching profiles.
Send `tllama list` if the appropriate registered references are unclear.
Do not download additional models merely to run every profile at once.

`embeddings --with-sdks` additionally runs the existing optional OpenAI/Ollama
Python SDK cycles and requires those packages in the same Python environment.
Without it, HTTP contracts are still exercised, but SDKs are explicitly marked
not exercised. The tools wrapper exercises HTTP consumers; optional tools SDK
cycles remain a separate existing smoke. Multimodal embeddings, Responses,
Decision, prefix-cache rollback and arbitrary model combinations are outside
this capture's claim.

### 5.1 Follow-up 0121a: external-store metadata and recovery deadline

The first captures exposed two diagnostic defects. Moving the ordinary disk
metadata cache outside the model repository made its relative paths invalid;
every public admission scanned the GGUF again. The wrapper now memoizes
successful header scans in its own process, guarded by the resolved file's
device/inode/size/mtime identity and a lock. Changed files and unsuccessful
scans are not reused. Each admission still resolves the registered TOML and
checks file identities. Production cache layout and schema are unchanged;
no metadata or digest is written into the registered model store.

The text recovery request after stream cancellation had a hard-coded 15-second
deadline. It now uses `--request-timeout` and names that stage on expiry.
Cleanup records reaping/removal separately from a started worker's exit status
and retains the original exercise exception if cleanup also fails.

Reviewed initial captures for `b11850-0e9433bfd3`:

| Backend | Text | Tools |
|---|---|---|
| MI50 / Vulkan0 | Passed, including all eight public MTP projections, cancel/recovery and unload | Protocol passed; four escaped-key semantic mismatches remain under review |
| MI100 | Both captures requested CUDA0; native rejected the unavailable device before loading | No HIP qualification from these runs |
| P100 / CUDA0 | All eight public MTP projections and schemas passed; recovery hit the diagnostic 15-second deadline | Protocol passed; four escaped-key semantic mismatches remain under review |

All 16 available raw/selected memory snapshots passed independent replay;
main/MTP intervals reconcile without double counting. GPU placement remains
full for noninput weights. The elapsed times here are not GPU benchmarks:
MI50 performed 64 header scans totalling about 763 seconds; P100 performed 63
totalling about 1675 seconds. No rerun of MI50 is needed for functional evidence.
Tools semantic mismatches remain unsuccessful captures (`ok=false`); existing
data do not independently locate them in model/native output versus adapters.
Do not repair escaped arguments or mark those cases accepted.

After applying 0121a after 0121 and running pytest, repeat only the two MI100
profiles with **ROCm0**, and P100 **text**. Keep mmap on. Use fresh outputs:

```bash
# MI100 / HIP
MODEL='byczech/qwen3.6-MTP-UD-Q4_K_XL:35b'
PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile text --model "$MODEL" --device ROCm0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --require-mtp \
  --output /tmp/tllama-070-mi100-public-text-retry
tar -cJf /tmp/tllama-070-mi100-public-text-retry.tar.xz \
  -C /tmp tllama-070-mi100-public-text-retry

PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile tools --model "$MODEL" --device ROCm0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --predict 256 \
  --output /tmp/tllama-070-mi100-public-tools-retry
tar -cJf /tmp/tllama-070-mi100-public-tools-retry.tar.xz \
  -C /tmp tllama-070-mi100-public-tools-retry

# P100 / CUDA
MODEL='qwen3.6-35b-A3B-MTP-UD-IQ3_XXS:35'
PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile text --model "$MODEL" --device CUDA0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --require-mtp --request-timeout 600 \
  --output /tmp/tllama-070-P100-public-text-retry
tar -cJf /tmp/tllama-070-P100-public-text-retry.tar.xz \
  -C /tmp tllama-070-P100-public-text-retry
```

The model references above are copied from the reviewed captures. Adjust only
if the registered names have since changed. No native rebuild or daemon restart
is needed. Send failed captures as well. MI50 and P100 tools need no immediate
repeat; the escaped-key review remains open separately.

Protocol success and semantic review are separate: a known model-specific
tool mismatch, a FIM semantic miss or embedding vector drift retains
`protocol_ok=true` when appropriate, but sets `review_required=true`,
`ok=false` and exit 1. Inspect generated text without executing it. Tools
historically had escaped-key semantic misses; do not repair arguments or
silently reinterpret those captures as a full pass. Cancellation after a
public frame and later reuse alone do not establish early native compute
stop; inspect actual native `cancel task`/release logs. Nonstream embedding
transport closure likewise does not prove early compute cancellation.

### 5.2 Follow-up 0122: raw escaped-key evidence

The three existing tools captures contain the same incorrect keys in native
`Parsed message` and public API results. That locates the difference before
the TLlama response adapters, but does not distinguish generation from the
native parser. Qwen3-Coder issue #531 describes similar symptoms; it does not
establish that the cause is the same. Ollama issue #15574 concerns a separate
XML parsing defect reported by this user.

The focused `tools-escape` profile sends four nonstream calls: the original
JSON-encoded key containing a real LF, and a control containing literal
backslash+n, each through TLlama's Ollama and OpenAI API surfaces. These are
compatibility endpoints of the isolated TLlama listener. A comparison against
the actual Ollama service is a subsequent experiment, not part of this profile.
There are no followup or cancellation calls in this focused capture; their
prior qualification is unchanged. MTP is off, ctx=4096, full GPU offload,
mmap on, K/V=q4_0, FA=on, and the original tool declarations and prompts are
reused. The control changes only the requested key. Other profiles are unchanged.

Only the diagnostic worker adds native `verbose: true`. In pinned native
commit `0e9433bfd3`, the nonstream `__verbose.content` comes from the raw
`slot.generated` text, while `choices[].message.tool_calls` contains the
parsed output. The same response also carries the actual prompt and generation
settings. Parsing and the tool grammar remain enabled; no second inference,
`parse_tool_calls` override, manually rendered completion, or raw endpoint is
used to manufacture the reference. No native rebuild is necessary.

`raw.jsonl` stores each complete `native-debug-response`, its actual native
request, the public request/response, and `escape-comparison` records. Missing
debug text/prompt/grammar or a native/public argument difference fails the
capture. A recognized single string parameter is inspected literally, without
unescaping or repairing it. Unrecognized XML and input text differences remain
review items. The diagnoses are observations about the recorded boundaries:

| Diagnosis | Evidence |
|---|---|
| `exact` | Raw XML value, native argument and public argument all match the requested key |
| `generated-value-mismatch` | Prompt preserves the user text; raw value is already wrong and survives parsing unchanged |
| `native-parser-difference` | Raw XML value differs from the parsed native argument |
| `input-needs-review` | The actual native prompt does not contain the submitted user text verbatim |
| `raw-format-needs-review` | The diagnostic cannot identify the exact single-parameter XML envelope safely |

The first two error diagnoses still need review of the captured prompt and
grammar before assigning responsibility to model weights versus template or
sampling. `exact` in this shorter fresh-worker sequence does not clear the
older failures. A mismatch retains `protocol_ok=true` when the protocol is
valid, but sets `review_required=true`, `ok=false`, and exit 1. Send the archive
in either case. The profile never executes a generated tool.

First capture: **MI100 / ROCm0**, with the same registered Q4_K_XL model that
showed the mismatch. Use a new output directory. After applying 0122 after
0121a and running the full pytest suite, run from the repository root:

```bash
MODEL='byczech/qwen3.6-MTP-UD-Q4_K_XL:35b'
PYTHONPATH=src python3 scripts/functional-qualification-smoke.py \
  --profile tools-escape --model "$MODEL" --device ROCm0 \
  --expected-fingerprint b11850-0e9433bfd3 \
  --ctx 4096 --predict 256 --request-timeout 600 \
  --output /tmp/tllama-070-mi100-tools-escape
tar -cJf /tmp/tllama-070-mi100-tools-escape.tar.xz \
  -C /tmp tllama-070-mi100-tools-escape
```

Only this one backend is needed initially. Do not repeat the full memory,
lifecycle, MTP or text matrix. Daemon restart and native rebuild are unnecessary.
The registered model TOML is not modified. A later actual-Ollama comparison
must use the same GGUF bytes and record its template and sampling; an Ollama
catalog alias alone does not establish model identity. Passing there would be
useful for locating a parser/template workaround, but two different generations
by themselves cannot prove which implementation is at fault.

## 6. Acceptance after captures

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
