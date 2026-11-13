# TLlama 1.5.0 release scope

Prepared 13 November 2026 after application checkpoint `f068884` (0222).
The maintainer reports 614 passed, 2 warnings for the final audit-fix selection;
the last reported full suite before release preparation was 5531 passed,
1 skipped, 2 warnings after 0215. Release build/full-suite results are separate
checks, not implied by this documentation checkpoint.

## Included behavior

| Area | Scope |
| --- | --- |
| Helpers | Standalone mmproj/draft TOMLs; composite imports without implicit extra definitions; typed list/show and explicit cp bindings |
| Removal | Reference-based component reclamation; explicit removal of attached unused helper definitions; sharing and worker protection retained |
| Speculation | One explicit none/MTP/DFlash strategy, one external draft; DFlash2 handled by native metadata/loader; no automatic selection |
| Draft controls | Device/layers/K/V/ubatch configuration; request and REPL strategy changes; effective-load generation changes |
| Diagnostics | Native counters and verbose client output; physical off/on and cache-state capture scripts; unknown allocations remain unknown |
| Pull | Single-line TTY progress with redirected/JSON fallbacks; alias-root reuse correction |
| Upload lifecycle | Published-group reuse, redundant alias removal and explicit local offline prune-uploads; no TTL |
| Audit fixes | RAM metadata identity, off-loop owned I/O, dead-code removal and reduced repeated group/cache work |

The [changelog](../CHANGELOG.md) contains upgrade notes. Repository format 2,
metadata schema 9, digest schema 2 and cache envelope format 2 remain unchanged.
Valid existing caches, stored weight names and original-name `.inf` companions
are preserved. No upgrade-time migration or unconditional rehash is required.

## Public API boundary

Only the previously agreed helper contract and Ollama strategy selector are
included: helper filtering on the existing list route, `server_features` capability
advertisement, `model_definition` details, existing copy/delete/create helper
semantics and `options.spec_type` on chat/generate. There are no new public
HTTP routes. [API compatibility](api-compatibility.md) and the
[DFlash guide](native-dflash.md) describe exact behavior and limitations.
Framework documentation endpoints remain disabled; complete OpenAPI description
is deferred. Maintenance runs locally through `tllamad`, never through HTTP.

## Qualification limits

The captured native contract is `b11850-0e9433bfd3` (`0.7.0-dev`), supplied by
the maintainer. Application and native runtime versions are independent; this
release introduces no native build or new minimum Debian runtime dependency.

MI50/Vulkan captures for the supplied Ornith target/draft pair show real
drafting/acceptance and cancellation/recovery. The second capture also records
cache reuse, checkpoint restore and cached/cold recovery. MTP-off evidence on
the supplied Qwen3.6 GGUF shows skipped MTP tensors and lower model buffers.
These are concrete cases, not certification of all DFlash/DFlash2 exports,
GPUs, quantizations, placement/KV combinations or model templates.

Tests cover the TLlama tools/structured-output, vision, literal FIM and
cache/context adapters with owned synthetic native subprocesses. That validates
the integration contracts and cleanup; it does not prove native grammar quality,
every speculative rollback path or all hardware/model combinations. Reported
interactive MI50 runs can be slower with DFlash enabled; no performance benefit
is promised and activation remains explicit.

## Deferred work

- A5: one configured working prefix plus repository-relative references,
  including RAM/cache keys; no general physical-root rewriting. It belongs to
  a later cleanup release, potentially accompanied by a separately qualified
  native `tllama-server` update.
- Measure unchanged-cache `list` and `run` startup before choosing further
  cache/I/O optimizations. A cache version change is not assumed; preserve
  locally measured digest payloads where their meaning remains valid.
- Bulk discovery/repair of old standalone helpers or misclassified definitions;
  explicit repeated pull is available without automatic destructive repair.
- Broader GPU/model/backend qualification, additional MTP skip cases, MoE SLRU,
  optional upload expiry and complete OpenAPI documentation.

These are follow-ups, not claims of completed support or implicit additions to
the 1.5.0 release. No root refactor or cache redesign is bundled with release
metadata changes.
