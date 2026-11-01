# Local repository pruning

`tllamad repo prune` inventories the local format-2 repository and previews
complete managed data groups that no definition or active reservation needs.
Only `--apply` deletes files. This is a one-shot filesystem operation on the
machine containing the repository, using the repository account's permissions.
It does not contact/start HTTP or load a native worker. There is no client
equivalent or maintenance HTTP endpoint.

```sh
tllamad repo prune --models-dir /var/lib/tllama/models
tllamad repo prune --models-dir /var/lib/tllama/models --apply --json
```

`--models-dir` is optional: CLI, `TLLAMA_MODELS`, server TOML and the default
`/var/lib/tllama/models` have the usual precedence. `--config`, `--no-config`
and `--models-dir` work before `repo` or after `prune`.

## Scope and protection

The preview lists each candidate group and the exact repository-relative files
to remove. A candidate consists of the entire `data/<group-sha256>/` directory:
all canonical GGUF shards, path caches for those weights, and repository-owned
upload hardlinks to the same inodes with their path caches. The empty group
directory is removed after its files. Independent upload copies and hardlinks
outside the managed upload area are not followed. No estimate of freed disk
space is given: remaining hardlinks and filesystem allocation affect it.

All model definitions participate, including projector and LoRA references.
An unattached projector/adapter group is an orphan just like unattached model
weights. Shared groups stay while any valid definition references them. Live
worker/publication reservations and retained protection records preserve their
groups; an upload-path reservation also defers a group with matching upload
hardlinks. Skipped reserved groups are not newly scheduled for later deletion.

Preview does not initialize missing storage, write caches, remove stale records
or drain the retry queue. `--apply` first performs a new structural inventory
of definitions, complete groups and GGUF headers, cache/control state and
reservations. Missing, unknown, unreadable, malformed or symlinked protection
and invalid data groups block every deletion. Regenerable unrelated cache
warnings and stale release records can be reported without blocking valid
candidates. Content hashing is available separately through `repo check
--hashes`; pruning does not require reading every weight byte.

One repository lock covers inspection, current reference/reservation checks,
scheduling and unlink. A writer cannot add a reference or reservation between
those checks and deletion. Preview output is a snapshot, never reusable
authorization. Apply may have different candidates if the store has changed.
All participating daemons must use the current reservation protocol and the
shared filesystem must honor its locks. Upgrade/restart older daemons first;
external readers that bypass the protocol cannot be detected.

## Failures and cancellation

Before the first unlink, all candidate groups' complete file lists are saved
durably in `.tllama/reclaim.json`. The existing targeted collector uses this
retry queue and repeats current reference/pin checks on every attempt. Removal
orders upload aliases before caches and weights, with directory fsync between
phases. Each completed group's queue entry is removed durably.

An I/O failure stops this invocation and preserves its partial-progress report
and unfinished queue entries. This is not an all-or-nothing transaction. The
ordinary collector can resume an interrupted group's deletion at subsequent
storage initialization, mutation or reservation release. Prune itself does not
bypass integrity errors to resume malformed/partially deleted groups: if its
structural inventory reports errors, it leaves the repository untouched.

Ctrl+C is cooperative. During inspection it stops before scheduling; during
deletion it finishes the current group before stopping. Other scheduled groups
remain queued and can be reclaimed by the ordinary collector later. Human and
JSON output retain the paths already removed and the cancellation status.

Exit status is 0 for a complete successful operation (including an empty
preview), 1 for inventory/deletion errors, 2 for invalid CLI/configuration,
and 130 for cancellation. JSON has report `version: 1`, an `apply` flag, issue
details, candidate/protected/deferred groups and actual removal paths/counts.
`ok` does not mean that every group was removed: protected groups are expected
to remain.

## Separate maintenance policies

This command does not sweep standalone cache entries, unrelated upload copies,
old staging/download directories or stale reservation records. Their ownership
and lifetime rules require a separate cleanup policy. It does not rewrite
definitions, repair corruption or convert older repositories. Use the
[offline converter](../tools/README.md) for legacy repositories.
