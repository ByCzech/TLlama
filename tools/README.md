# Offline repository conversion

`migrate-repository-v2.py` converts a legacy TLlama model repository into
format 2. It is a one-time administration tool, outside the installed Python
package and console commands. The source distribution includes it so Debian
can ship it in the documentation package; the Python wheel does not.

The ordinary daemon now requires format 2. Empty stores initialize automatically;
existing unmarked stores must be converted explicitly while all writers are
stopped. The final marker is written only after successful publication,
definition changes and cleanup. Incomplete conversions block daemon startup.

## Requirements and preview

Use the updated source checkout (including the repository publication layer)
and its normal Python dependencies, or the matching installed TLlama when
running the documentation copy. The **input repository** may come from a
release older than 1.3.1; it does not have to be upgraded through 1.3.1 first.
This is a TLlama repository converter, not an Ollama manifest converter.

From the checkout:

```sh
python3 tools/migrate-repository-v2.py /var/lib/tllama/models
python3 tools/migrate-repository-v2.py /var/lib/tllama/models --json
```

Without `--apply`/`--resume` the tool writes nothing, including caches and lock
files. It checks definition names, portable-path collisions, TOML, referenced
files, complete shard groups and actual content hashes. It reports old/new
paths and the expected model IDs. It never prints system prompts or full TOML.
Current metadata and locally measured digest caches are reused when valid.
Missing, stale, corrupt or old/unknown caches are ignored and regenerated at
their new paths during application. A remote `hf` digest claim is measured
locally before it is used as a content address.

Only the existing fixed-depth `Local`, `TLlama` and `HuggingFace` TOML
definitions are models. Misplaced nested definitions inside those branches
are errors requiring correction before conversion. Bare GGUF files elsewhere
in the repository are moved too, without creating new model definitions.
Existing unimported `[llm].from`/`[mmproj].from` values stay pending; the tool
does not import arbitrary client-side files. LoRA references inside the
repository are included. Symlinked weight/control paths or directories are
rejected explicitly; a symlink to the whole repository root is supported.
Legacy `.TOML` extensions are canonicalized to `.toml`. Ambiguous duplicate
definition targets and noncanonical casing of the reserved branch directories
must be reconciled before conversion.

Original filenames/locations are treated as literal legacy names: a literal
`model%3A55b` is distinct from `model:55b`. The converted definition paths use
the shared percent codec. Weights have canonical names under their content
addresses; shards remain together in one directory. Definitions refer to the
first shard. Templates, prompts and model parameters retain their values;
comments and CRLF line endings are preserved.

The tool also checks that recomputing model identity with the changed paths
produces the same result. This compares locally measured content; if an older
cache contained an incorrect remote digest claim, that claim is not preserved.

## Apply and recover

Stop **every daemon using this repository**, including daemons on other hosts, and keep all other writers
stopped until conversion completes. The old runtime does not participate in
the new repository lock, so a lock cannot automatically stop an old daemon.
Run as the repository service account or another deliberately chosen account
with suitable access. When converting as an administrator for a different
service account, ensure that account can subsequently write the newly created
`data/`, definition directories, metadata cache and `.tllama/repository.lock`
through the intended owner/group/ACL policy. Private migration backups and the
journal may remain administrator-only. Running as the service account avoids
creating administrator-owned runtime files. No command below starts or stops
services for you.

```sh
python3 tools/migrate-repository-v2.py /var/lib/tllama/models --apply
```

The tool takes the shared repository lock, rechecks its preview, and records a
private `.tllama/migration-v2/journal.json` before publishing anything. Its
phases are:

1. Publish verified complete data groups. Repository-owned files on the same
   filesystem are hard-linked so conversion does not copy gigabytes. Source
   files on mounted subdirectories of another filesystem use verified copies.
   Publication staging and `data/` must be on the same filesystem for atomic
   directory rename; preflight rejects incompatible mounts.
2. Save every original definition in the private `backups/` directory using
   numeric filenames, then rewrite all paths and definition names. Backing up
   every source first also handles old names that coincide with another
   definition's newly encoded name. New definition inodes keep access bits
   and nanosecond timestamps; POSIX owner/group are restored where supported,
   including verified weight copies across mounts. Run as the original owner
   or an administrator when needed. Weight hard links retain their inode
   metadata. Identical groups with different owner/access bits are rejected
   for deliberate reconciliation rather than silently broadening access.
   Per-file extended ACLs are not replicated onto new definition/copy inodes;
   directory inheritance still applies. Review custom ACLs separately.
3. Verify final data and definitions, remove the old weight paths, invalidate
   obsolete path-keyed caches, remove stopped legacy upload staging and empty
   old directories, then write the format marker. Unrelated files are retained;
   their names/content are not converted and may need separate portability
   cleanup. Private backups and the journal remain for inspection/recovery.

If interrupted, keep daemons stopped and run:

```sh
python3 tools/migrate-repository-v2.py /var/lib/tllama/models --resume
```

Resume verifies the journal, published content, backups and definition hashes.
Use the matching converter/core revision for recovery; a different cache/layout
contract is rejected rather than labeling an older captured payload with a new
schema. This concerns an interrupted migration journal, not the old input cache:
old input cache versions remain supported as misses and need no intermediate
application upgrade.
Unexpected changed/new definitions, weights or uploads stop recovery rather
than being silently overwritten or deleted. Successfully completed conversion
is idempotent. `--resume` is the supported automated recovery direction; there
is no automatic rollback mode. The private journal retains the original
definition bytes, metadata and complete old/new weight mapping for deliberate
manual restoration if ever required. Do not edit it or remove backups while
recovery is pending. If a failure occurred before the initial journal was
published, original definitions/weights have not been changed; inspect any
unrecognized leftover control files before retrying.

After a successful conversion use the new runtime exclusively. An older
TLlama binary predating the guard may ignore the format marker; never start
it on a converted repository. The current runtime validates the explicit
format-2 marker and never falls back to legacy layout interpretation.
