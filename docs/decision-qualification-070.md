# Shared runtime fingerprint policy and Decision qualification — 0126 revised

This revision replaces the original, unapplied 0126. Apply it directly after
0124; 0125 is a capture runner, not a patch. Do not apply both versions of 0126.

## One admission policy

`tllama.runtime_fingerprint.evaluate_runtime_fingerprint` is the common
warn/strict evaluator. It receives the observed identity, expected identities,
policy and operation context. A match returns true without a warning. An
unknown or missing identity returns false with one warning under warn, or
raises under strict. NativeWorker maps the shared strict exception to the
existing WorkerProtocolError/startup failure and owns normal cleanup.

Generation, embeddings and Decision all use this same startup path and the
configured version policy. The default remains warn: an unaudited build alone
does not prevent inference. Strict refuses a build outside the applicable
expected set. Decision retains its separately audited build set; no candidate
is promoted into any production list by running diagnostics.

WorkerLaunchSpec selects its expected set separately from enforcement:

- An explicit expected_runtime_fingerprint is a singleton diagnostic pin.
- Otherwise Decision uses its existing Decision audit set.
- Other modes retain the existing custom runtime_fingerprint launch behavior,
  or the ordinary audited runtime set when using the default launch identity.

The explicit pin works for every worker mode, including a pin to the ordinary
default build. It is not inferred by comparing a requested value to a default
constant. Normal server/model configuration does not set this internal pin.
The existing runtime_fingerprint field remains compatible for older callers.

Decision API diagnostics now use the ordinary ProbeWorker and NativeWorker,
with an explicit singleton pin and strict policy. There is no extra admission
subclass, temporary warn policy, monkeypatch or after-start version bypass.
Both the independent and managed/public worker must satisfy the same exact
identity check before start returns and any inference lease can be delivered.
Without --expected-fingerprint, this diagnostic still pins the previously
audited Decision b11552-3144c167ef. Native-only Decision references use that
same strict shared startup gate. Lifecycle/MTP, functional and memory captures
also set explicit startup pins; memory capture without a requested identity
retains the configured warn/strict policy. Device inventory uses the same
strict evaluator for its explicitly selected capture build.

Independent diagnostic evidence checks may still record observed identities.
Log-grammar audit flags, discovery audit flags and equality between a running
worker's identity and its memory report are separate evidence checks, not
alternative warn/strict admission policies. In particular, a pin must not
claim an unaudited log parser is verified. Alias, slot, modality, JSON/schema,
probability and response validation are unchanged and still reject malformed
protocols independently of version policy.

## Capture on MI50

After applying the revised 0126 and running pytest, use the unchanged companion
0126-mi50-decision-captures.sh from the current TLlama checkout, or run:

```bash
PYTHONPATH=src python3 scripts/decision-api-smoke.py \
  --expected-fingerprint b11850-0e9433bfd3 --device Vulkan0 \
  --load-timeout 600 --request-timeout 600 \
  --output /tmp/tllama-070-mi50-decision-suite
```

The three already registered models are Laya Q8_0, Bespoke-Nimble-9B-v3 Q4_K_M
and Clef-Flash Q4_K_M, sequentially. Each uses two sequential independent and
managed/public workers. The native trained-head prompt/template, F16 K/V,
FA off and speculation none remain unchanged. Context/batch/ubatch is 1024
for Laya and 2048 for Nimble/Clef. CLI --transport tcp is available if UDS is
unavailable. The diagnostic records its strict policy and selected identity.

All 159 cross-path comparisons plus three public cold-start comparisons,
independent reference observations, literal keys, ordering, exact integer
usage, refusal probes, transport cancellation/recovery, unload and cleanup
remain required. Comparison tolerances remain atol=1e-4 and rtol=1e-3.
The synthetic subprocess tests use 5-second load/request deadlines within
their 35-second process deadline; the hardware capture keeps 600 seconds.

Use a new output directory and return the complete archive even on failure.
Exit zero alone is not acceptance: inspect review_required, observations and
raw evidence. Transport cancellation does not prove early native compute stop
or a latency bound. Backend memory is buffer capacity, not physical peak RAM
or VRAM. This is protocol equivalence, not model-accuracy certification.

The runner needs no new downloads or SDKs and does not edit registered model
definitions or a running service. No native rebuild or service restart is
needed. Keep the accepted embeddings captures and other completed GPU matrices.
