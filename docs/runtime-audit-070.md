# Native 0.7.0 audit results — 0127

Reviewed on 27 October 2026: native **b11850-0e9433bfd3**, commit
`0e9433bfd33d8cdb194285b70c0d421d39135df2`, Debian packages
`0.7.0~dev+b11850.1-1`. The user's complete TLlama suite after the revised
0126 passed **4047 tests, 2 warnings**; after applying 0127 the user confirmed
**4086 passed, 2 warnings**. Warning details were not supplied. These results
do not by themselves authorize or complete the 1.0.0 release.

## Runtime policy

The reviewed build is the default identity for ordinary worker launches and
Decision references. Previously supported builds remain supported in their
respective contracts. These audit sets are maintained independently:

| Contract | Reviewed fingerprints |
|---|---|
| Ordinary generation/embedding worker | b11850-0e9433bfd3, b11552-ec2797836f, b11552-d1570abd3d, b11552-3144c167ef |
| Text Decision worker | b11850-0e9433bfd3, b11552-3144c167ef |
| Discovery table grammar | b11850-0e9433bfd3, b11552-3144c167ef |
| Initial load-log grammar | b11850-0e9433bfd3, b11552-ec2797836f, b11552-d1570abd3d, b11552-3144c167ef |

All worker modes use `evaluate_runtime_fingerprint`. The default `warn` policy
continues with a warning for an unreviewed/missing identity; `strict` refuses
it. An explicit internal `expected_runtime_fingerprint` is a singleton pin,
even when another observed build is in an audit set. Diagnostics retain their
strict exact-build checks. A legacy custom `runtime_fingerprint` launch value
also retains its exact-selection behavior for ordinary workers.

Discovery defaults to an exact pin to the reviewed 0.7.0 identity. To capture
an older build, explicitly select it with `--expected-fingerprint`. An older
reviewed build still reports `runtime_audited=true`; an unreviewed exact-match
candidate can be captured but requires review. `runtime_audited` describes
version provenance, not successful parsing: malformed tables still fail and
set `review_required=true`. The report also lists `audited_runtime_fingerprints`.

No identity bypasses live health, alias, slot, modality, typed response or memory
validation. The log audit set is explicit rather than derived from the ordinary
worker default, so a future default change cannot silently certify a log grammar.
There is no native upgrade, changed minimum dependency, new model loader or
change to sampling, K/V, flash attention, mmap or device selection in 0127.

## Reviewed hardware and feature scope

| Area | Evidence and accepted scope |
|---|---|
| Discovery | Actual MI50, MI100 and P100 version/device captures; selectors and table grammar retained. Kernel usability, PCI identity links and untested backends are not inferred. |
| Memory and vision | Full, partial and vision Qwen3.6 captures on MI50/Vulkan, MI100/HIP and P100/CUDA12.9; V1 buffers, weight payload/assignment, locality coverage, bounded tensor diagnostics and cleanup reviewed. MI50/MI100 used Q4_K_XL; P100 used IQ3_XXS. |
| Lifecycle | Unload, failed replacement, sleep/wake and cancellation/recovery reviewed on those three backends. The P100 wake capture needed a 600-second request deadline. |
| Embedded MTP | Off/on captures on those three backends: draft activity, acceptance, parity, cancellation/recovery and memory ownership reviewed. This does not certify every speculative method or model. |
| Public text/tools | Reviewed captures on those three backends, including compatible API/SDK paths and recovery. The escaped-key limitation below remains explicit. |
| Embeddings | MI50/Vulkan0, MiniLM Q8_0 (384) and Qwen3-Embedding-0.6B Q8_0 (1024): 36 vector comparisons per model, zero difference, both SDKs, error/recovery checks and 8 memory reports. K/V f16, FA off, context 256. Other backends/models were not qualified by these captures. |
| Text Decision | MI50/Vulkan0, Laya Q8_0, Nimble Q4_K_M and Clef Q4_K_M: 159 cross-path comparisons and 3 cold-start comparisons, all zero difference; typed requests, errors, cancellation/recovery, 12 memory reports and 6 clean worker exits. K/V f16, FA off; context/batch/ubatch 1024 for Laya, 2048 for Nimble/Clef. |
| FIM | MI50 CodeGemma public/native protocol and Llama negative control passed. Qwen public/native paths match after aligning prompt reuse. Semantic quality remains unresolved; no general FIM quality certification is claimed. |
| Load-log fallback | Actual full/partial/vision allocation lines for all three backends replayed without changing grammar. Rounded buffer totals agree with native within printed precision. Vision correctly remains partial because projector allocations are not fully reported. |

The general runtime audit is compatibility admission for a build. It is not a
backend/model quality certificate. Feature-specific behavior still depends on
the actual model, launch recipe and live native protocol. In particular, the
Decision result is protocol equivalence, not a classifier-accuracy benchmark.
Transport cancellation alone does not prove early native compute termination.
Locality and buffer capacities do not measure physical RAM/VRAM residency.

## Evidence and retained limits

The full capture archives are retained outside the Git repository. Bounded
device-table and allocation-log excerpts, source archive SHA256 and expected
results are in `tests/fixtures/native-runtime-070/`. These are captured bytes,
not regenerated example output. Existing historical fixtures keep their original
identity. No model paths, credentials or complete worker logs are added here.

- The Decision suite is `tllama-070-mi50-decision-suite.tar.xz`, SHA256
  `9cbb518ccc4601d13ac3813830a09925f38ec8eec77dbfabcc3745438e61b0c1`.
  Its 53 native/managed/public response triplets were independently compared
  for exact answers and integer usage. Nimble/Clef composition and question-order
  effects are permitted native behavior; identical requests still match exactly.
- The embeddings suite is `tllama-070-mi50-embeddings-suite.tar(1).xz`.
  The first attempt without the `openai` SDK did not exercise inference and is
  not counted as a failed embedding implementation.
- Qwen FIM with no prompt reuse had 12/12 identical public/native outputs.
  Content checks passed 0/12 with K/V q4_0 and 2/12 with K/V f16 (one of six
  unique tasks). Model weights remained Q4_K_M. K/V precision affected results;
  neither a model-only nor a TLlama-only root cause has been established.
  Reference-model selection and comprehensive independent FIM quality testing
  are scheduled after 1.0.0 by the maintainer's 27 October 2026 decision. They
  no longer block 1.0.0; a corrective release will follow if results require it.
  The evidence above is not promoted into a general FIM quality certification.
- The Qwen tools escaped-key case produced an extra JSON escape layer already
  in native generated XML; TLlama preserved it. Real Ollama reproduced the
  public values. This is retained as a model/prompt/runtime limitation, without
  speculative automatic unescaping in TLlama.
- The original P100 sparse mmap capacity inflation was not reproduced by the
  reviewed memory profiles. Their zero capacity reduction is not proof that
  the original case is resolved. The maintainer accepted further real-use
  testing of the new version after release; no additional pre-1.0.0 capture is
  required for this case and it is not declared fixed.
- Legacy HIP gfx900/gfx906/gfx90c mmap investigation is separate. The no-mmap
  workaround is not an upstream driver fix or certification of the mmap path.
  HIP on these targets usually needs mmap disabled, or Vulkan can be used
  instead when a usable device is available. A probable amdgpu/KFD issue is
  suspected, not conclusively confirmed. See the
  [hardware workaround](hardware.md#legacy-hip-mmap-workaround).
  MI50/Vulkan evidence does not qualify MI50/HIP; this documented limitation
  is the accepted 1.0.0 disposition rather than a pending driver-fix gate.

## Remaining work before 1.0.0

0127 has passed the user's complete suite (4086 tests, 2 warnings). Accepted
hardware captures need not be repeated solely for the documentation review. Keep new failures and any
changed native identity separate from the reviewed results above.

The complete source documentation review was performed in 0128 and supplemented
by 0129/0130 and the 0131 permissions guide. The
[review record](release-review-100.md) retains the checked scope and validation;
this audit does not require repeating completed reviews or accepted captures.

The remaining operational checks are the full user pytest result after 0131
and the actual service-account retest of Pull errno 13. Sparse mmap and FIM
follow-up are scheduled after release; legacy HIP has the documented workaround
above. Final merge/version/release-note/tag checks follow the operational result.
These dispositions do not by themselves perform or announce a release.
