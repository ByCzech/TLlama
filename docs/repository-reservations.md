# Repository reservations

Reservations are the shared protection layer for future physical reclamation.
They describe resources in use; they do not delete weights or enable garbage
collection. The first integration protects streaming uploads and complete HF
download/publication jobs. Worker generation ownership is the next integration.
Ordinary `rm` still removes the definition and retires local workers.

## Ownership and coordination

Every reservation has a unique file in `.tllama/pins/`. Its JSON records a
schema version, purpose, protected group addresses and owned staging paths.
A temporary broad `all_data` reservation is available for resolution before
the exact definition snapshot is known. Records contain no tokens, prompts
or environment values.

Each owner holds a separate exclusive OS lock for its entire reservation.
Liveness does not depend on a PID, hostname, timestamp or expiry interval.
The record's inode stays fixed during updates. A leading whitespace byte
keeps the file valid JSON while providing a Windows lock byte; readers read
the payload after it. POSIX uses flock and Windows uses byte-range locking.
The shared filesystem must honor these locks across every participating host.

Creation, updates, snapshots and removal also take the existing repository
lock. A future collector must keep that transaction locked from its final
reservation/definition checks through unlink. It must never inspect a
snapshot, release the repository lock and later delete based on that stale
snapshot. Long downloads and uploads hold their reservation, not the global
repository lock; unrelated transactions can proceed.

## Completion and process failure

Successful or failed operations clean their temporary resources before
releasing the reservation. Async ownership joins acquisition and release even
when the request is cancelled, so late executor work cannot create an
abandoned live pin after cancellation returns.

An unlocked `release-*.json` record has no remaining owner. A snapshot can
report it without changing disk; an explicit `remove_stale=True` transaction
removes only these stale reservation records. It does not remove their data
or staging resources. A truncated release record left by process death is
still recoverable because the filename declares its orphan policy.

On POSIX the reservation exposes `inherited_fds` for explicit `pass_fds` to a
native child. The inherited open description can keep the flock alive after
the parent descriptor disappears. A future worker integration must pass it
to every actual backend attempt and keep normal ownership until the worker
has been reaped. This release adds and tests that primitive; it does not yet
attach it to daemon worker generations.

For an owner that cannot preserve the lock in a surviving child,
`retain_on_crash=True` creates a `retain-*.json` record. An unlocked leftover
remains protected for inspection rather than being automatically reclaimed.
Live or retained malformed/unknown records, symlinks and unexpected entries
abort the snapshot: an incomplete protection set cannot authorize deletion.
All cooperating writers must preserve this namespace and never replace or
remove another owner's active record.

## Current staging integration

`receive_blob` reserves `.tllama/uploads/` from before receiving the body
through digest verification, publication and cache recording. This protects
the temporary stream as well as its verified destination. Completed blobs
without a live operation are separate from active reservations; a future
staging cleanup policy must handle their reuse between HTTP requests.

HF pull creates its private download directory and reservation in one locked
transaction. The pin covers the complete job, including shard fetches and
hashing outside the repository lock. Once the group address is measured, it
adds that group before publication/definition registration. Temporary files
are removed before the pin is released, including on transfer failure.
Already verified repeat pulls perform their group/definition work under the
existing repository lock. Create/copy publication is likewise already locked.

Before physical reclamation can be enabled, worker generations and all other
cooperating readers must participate, and every daemon sharing the store must
be upgraded/restarted. Filesystem locks cannot detect an older daemon or an
external reader that does not use the reservation protocol. No first-stage
reservation patch changes that activation requirement or turns on deletion.
