# Draft metadata boundary (13 November 2026)

TLlama needs enough header evidence to identify the stored artifact, enumerate
complete shard groups and reject a known wrong role when activating DFlash.
It does not need a second implementation of the native model loader.

| Evidence | Existing TLlama path | Decision |
| --- | --- | --- |
| Architecture, projector flags and general type | Selected GGUF fields, normalized payload and disk cache | Retain; known non-dflash architecture rejects active DFlash |
| split.count / split.no | Header selection, shard enumeration and file fingerprints | Retain strict completeness checks and pin every shard |
| Context, blocks, embedding width and RoPE defaults | Architecture-specific selected keys | Already read, including the dflash prefix |
| Vocabulary size and compact FIM inventory | Compact derived metadata; large tokenizer arrays excluded from cache | Retain; counts are not proof of tokenizer equivalence |
| DFlash block size, target_layers and attention layout | Present in the supplied native load log, not used by TLlama | Do not add cache/API fields without an actual consumer |
| Full tokenizer/tensor/pair compatibility | Actual native startup, followed by functional qualification | Do not infer from filenames, vocabulary counts, widths or base-model labels |

The supplied b11850-0e9433bfd3 Ornith capture reports general.architecture=dflash,
block_count=6, context_length=262144, embedding_length=2048, block_size=16 and
target_layers=[2,7,12,17,23,28,33,38]. This is evidence for that export. It is not
a universal format specification or a certificate for every target/draft pair.
The exact pinned C++ source was unavailable for source-level inspection in this
audit; the native load/qualification logs do not establish an exhaustive list
of all loader checks. No new compatibility rule is derived from those samples.

## Narrow admission fix

The target already tolerates optional Python metadata-scan failure and lets the
native loader decide. Active external draft admission previously read the same
optional metadata again without that boundary, so ValueError/OSError prevented
native startup. The normalized architecture string `unknown` was also treated
as a known wrong architecture. These cases now reach the native loader with the
requested DFlash strategy and path unchanged. A native failure remains a failure;
TLlama does not retry with speculation disabled or another strategy.

This does not relax file/group requirements. Missing files, known incompatible
architecture, unreadable shard headers or missing evidence for a named shard
set are still rejected before startup. Unknown metadata cannot turn a multi-file
artifact into an apparently complete single-file model.

Stored `[draft]` references remain generic. `create`/`cp` may retain an inactive
draft of a different architecture; the DFlash-specific check belongs to active
strategy admission. Attaching a helper does not prove pair compatibility or
activate speculation. Existing projector-role checks remain separate.

No metadata keys, cache schema, model identity rules, public fields or repository
format change. The current metadata cache version 9 and independent content
digest cache remain valid; no cache deletion or full weight rehash is needed.
