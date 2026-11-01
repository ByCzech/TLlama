# Model repositories

TLlama separates model definitions from shared weight data. The root is selected
by `storage.models_dir`, `--models-dir`, or `TLLAMA_MODELS` (default
`/var/lib/tllama/models`). All local additions use `tllama create`; Hugging Face
files use `tllama pull`.

## Layout and public names

Definitions live at a fixed depth in three branches:

| Public name | Definition path |
| --- | --- |
| `my-model` | `Local/my-model.toml` |
| `ByCzech/my-model` | `TLlama/ByCzech/my-model.toml` |
| `namespace/repository/my-model` | `HuggingFace/namespace/repository/my-model.toml` |

The branches hold definitions. All managed GGUF files live in
`data/<group-sha256>/`, regardless of the branch or source. A single-file group
contains `weights.gguf`. A split group contains
`weights-00001-of-NNNNN.gguf` and every continuation shard in the same directory;
a definition references the first shard. Projectors and LoRA adapters have
independent groups, allowing sharing across models and quantizations.

A single-file directory uses the file's SHA-256. A split directory uses the
versioned hash of its ordered member digests. Original filenames, aliases and
TOML parameters do not enter this address. Public model IDs remain separate:
they include weights, projector/adapter and non-default parameters. Copying a
definition shares files; identical uploads/downloads reuse complete groups.
Published data is immutable and existing groups are verified before reuse.

### Portable definition names

Public names keep their spelling. Physical definition components selectively
escape `<>:"|?*%`, terminal dots/spaces and reserved Windows basenames using
uppercase `%HH`. Ordinary Unicode and internal spaces stay readable. For
example, `ByCzech/model:Q4%` lives in
`TLlama/ByCzech/model%3AQ4%25.toml`. The daemon encodes on lookup and decodes
physical names exactly once on listing. A literal `%3A` in a public name
therefore differs from `:`. TOML paths are literal physical paths, not URL
strings to decode.

Names differing only by case or Unicode normalization are rejected on every
OS, including Linux, to keep the repository portable. Component and full path
lengths still obey filesystem limits. A bare GGUF is never discovered as a
model; only a registered fixed-depth TOML defines a public name.

## Add a local model

Create a client-side TOML:

```toml
[llm]
from = "./model.gguf"

# Optional independent projector:
# [mmproj]
# from = "./projector.gguf"
```

Then:

```sh
tllama create my-model -f model.toml
tllama show my-model
tllama cp my-model ByCzech/another-name
```

Relative `from` paths use the client's current working directory, including
when the TOML is elsewhere. The client validates and uploads the complete GGUF
group. The server stages verified blobs under `.tllama/uploads/`, publishes
canonical groups into `data/`, rewrites `from` to the final physical `model`
path, and publishes the definition last. Comments and parameters survive. The
client's files stay untouched; no arbitrary server-side `from` file is read.

To reuse existing managed data without uploading, use its exact
`data/<group-sha256>/weights.gguf` (or first-shard) path in `model`. Refer to
`show --modelfile` for an existing definition. Default name conflicts require
`--force`; force replaces only the definition. See [CLI import](cli.md#create-and-import-models).

## Pull from Hugging Face

```sh
tllama pull namespace/repository/path/to/model.gguf
```

`POST /api/pull` accepts the same reference in `model` or `name`, with optional
`stream: false`; the default response is NDJSON progress. The `.gguf` suffix
may be omitted. This is a repository file reference, not an Ollama registry
tag or browser `blob/main`/`resolve/main` URL. The optional `huggingface` extra
must be installed on the daemon.

Downloads use a private temporary directory under `.tllama/downloads/`.
TLlama reads the first GGUF header, fetches all declared sibling shards,
validates and measures them, and atomically publishes the complete group.
Interrupted/failed transfers leave no partial data group or model definition.
Progress counts the whole group. Final path caches are warmed at publication.

An ordinary model gets the public name `namespace/repository/model`, using the
requested basename without `.gguf`. A conflicting existing definition is kept;
another available numbered name is used. Source records under
`.tllama/origins/` associate remote paths with chosen aliases and measured data.
Subsequent pulls of an unchanged source reuse data only after checking the
published digest of every remote member. If the source changes, TLlama updates
its associated definition's weight pointer while preserving comments and user
settings. Other aliases retain their original weights. A definition that the
user repointed elsewhere is preserved and a new alias is chosen. Tokens are
never stored in source records. Projector pulls publish data and return its
path without registering a standalone language model or guessing attachments.

## Upgrade an existing repository

An empty repository is initialized automatically with
`.tllama/repository.json`. A nonempty unmarked repository requires explicit
offline conversion; the daemon refuses to reinterpret it. Unknown formats and
incomplete conversions are refused too. The final format marker commits a
complete conversion; private recovery records need not be readable by the
service account afterward.

Stop every daemon/writer using the repository, preview and apply:

```sh
python3 tools/migrate-repository-v2.py /var/lib/tllama/models
python3 tools/migrate-repository-v2.py /var/lib/tllama/models --apply
```

Then start the upgraded daemon. See [conversion and recovery](../tools/README.md)
for permissions, old caches and `--resume`. The tool supports repositories from
versions older than 1.3.1. `rebuildrepo` and the old raw-file naming/discovery
code have been removed; copying weights into a branch is no longer an import.

## Removal and caches

`rm` removes the definition, retires local workers and reclaims only its
now-unused managed groups, including every shard, their disk caches and
matching repository-owned upload hardlinks. Other definitions keep shared
groups. Loading/live/draining workers and active publications keep their
reserved groups until release. `create --force`, `cp --force` and an updated
HF definition use the same targeted cleanup for the previous groups.
Unrelated orphan groups are left for separate whole-store maintenance.

Deletion responses include `cleanup` with removed/shared/deferred groups,
removed paths and warnings. A broken definition or reservation blocks physical
cleanup, while the requested definition removal still succeeds. Broken removed
definitions cannot supply reliable group candidates; no paths are guessed.
`kept_model_file` remains available and is null when the main weight is gone.
Create/copy add cleanup details when there is cleanup activity to report.
The CLI prints cleanup warnings and deferrals on stderr, including with
`--json`; a successful definition mutation keeps exit status zero.
See the [implementation contract](repository-v2.md#removal-and-maintenance).

Deferred candidates live in `.tllama/reclaim.json`. Reservation release/narrowing,
subsequent definition operations and daemon storage initialization retry
them under the repository lock after checking the current definitions/pins.
After a crashed owner disappears, retry occurs on one of these events;
there is no background timer. Successfully published upload blobs without an
active upload are reusable staging, not persistent model definitions: their
matching hardlinks can be removed when the last model using a group is deleted.
Clients can check/re-upload missing blobs before another create.

**Upgrade and restart all daemons sharing this repository before using
physical reclamation.** A daemon predating the reservation protocol, or an
external reader, cannot be detected by locks. The repository format marker is
unchanged; this update does not require another migration.
Uploads, download/publication jobs, worker snapshots and managed physical
diagnostic launches use the shared [reservation layer](repository-reservations.md).

Metadata and measured content digests live under `.tllama/metadata-cache/`.
Keys use repository-relative physical file paths, validated against size and
modification time. Moving the whole repository preserves valid caches. Missing
or obsolete caches regenerate on demand; removing this cache directory is safe
while daemons are stopped, but later hashing can read the full weights again.
Source records are bookkeeping distinct from these regenerable caches.
See [metadata-cache.md](metadata-cache.md).

## File permissions

Files are created by the **daemon account**, including requests sent from a CLI
running as another user. New model TOMLs (`create`, `copy`, pull registration),
uploaded GGUF blobs and their published hard links, and newly written metadata
and digest cache files use a creation mode of `0666`. The operating system
applies the daemon's **umask** and any parent directory default ACLs. Directories
use `0777` subject to the same filesystem rules. On an ordinary POSIX directory
without default ACLs:

| Daemon umask | New files | New directories |
|---|---|---|
| `0022` | `0644` | `0755` |
| `0002` | `0664` | `0775` |
| `0027` | `0640` | `0750` |
| `0077` | `0600` | `0700` |

Set the mask before starting the daemon, for example for group sharing:

```sh
(umask 0002; exec tllamad --config /etc/tllama/server.toml)
```

For a service, configure `UMask=0002` as shown in
[systemd.md](systemd.md#file-creation-mask). Changing the CLI shell's mask does
not change an already running daemon. There is no separate TOML permission
setting; the process mask also applies to libraries used by the daemon.

Replacing an existing TOML (for example `create --force`) preserves its access
bits instead of resetting them from the current mask. Atomic replacement creates
a new inode: ownership/group and inherited ACLs follow the destination directory;
custom extended ACLs and special mode bits from the old inode are not copied.
Uploaded weights are shared through hard links, so existing blobs retain their
permissions. Cache refreshes create new files under the current mask.

GGUF downloads and Hugging Face's internal cache/lock files remain managed by
`huggingface_hub`; TLlama does not recursively chmod them or normalize third-party
permissions. Windows access is governed by Windows ACLs; this POSIX
mode table is not an equivalent Windows security policy.

Older TLlama versions created TOMLs, uploads and cache entries with `0600`.
Changing the mask does not retroactively repair existing files or directories.
An administrator must adjust the intended model-store entries, group membership
and traversal permissions separately. For a shared store, directory setgid/default
ACLs can supply the intended group; umask alone does not choose one.

Private worker sockets, IPC credentials and runtime directories keep their
dedicated restrictive permissions; they do not use repository file modes.
