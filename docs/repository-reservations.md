# Repository reservations

Reservations are the shared protection layer for targeted physical reclamation.
They describe resources in use; the collector checks them before deleting
retired groups. Integrations protect streaming uploads, complete HF
download/publication jobs and native worker generation snapshots.
Ordinary `rm` removes the definition, retires local workers and reclaims its
unused groups. Shared or reserved groups remain protected.

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
lock. The collector keeps that transaction locked from its final
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
the parent descriptor disappears. Worker generation ownership passes it
to every actual backend attempt and releases normal ownership only after
worker cleanup has reaped the process. Internal descriptor numbers are not
part of model/configuration identity and never appear as native CLI options.

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
Targeted reclamation removes only completed upload entries that are hardlinks
of a reclaimed group; a live upload path reservation defers that entire group.
Unrelated blobs, independent copies and temporary job leftovers are not scanned
by this targeted cleanup as whole-store orphans.

HF pull creates its private download directory and reservation in one locked
transaction. The pin covers the complete job, including shard fetches and
hashing outside the repository lock. Once the group address is measured, it
adds that group before publication/definition registration. Temporary files
are removed before the pin is released, including on transfer failure.
Already verified repeat pulls perform their group/definition work under the
existing repository lock. Create/copy publication is likewise already locked.

## Worker snapshots and physical diagnostics

Before model definition/header/digest I/O, admission takes an `all_data` pin.
The I/O runs outside the residency and repository locks. Once a stable launch
snapshot is resolved, the pin narrows to its actual model, projector and LoRA
group addresses under the repository lock, with no unprotected interval.
Resolution into a different managed repository is rejected rather than
attaching the original repository's unrelated protection descriptor.
Each group protects all its shards. Unmarked physical diagnostic inputs are
outside managed GC and do not acquire repository bookkeeping.

A newly admitted generation takes ownership of that pin. Reusing an existing
generation releases the unused resolution pin; validation failures, capacity
retries, cancelled admission and factory failures also release unused pins.
Cancellation joins resolution work before releasing its protection. Shutting
down the residency prevents late admission even if a definition reader is
still completing its owned I/O; its pin remains until that reader settles.

Loading, ready, retired/draining and closing generations keep their own
snapshot regardless of subsequent TOML edits/deletion. Pin release follows
successful worker cleanup. A cleanup failure keeps the pin and quarantined
capacity instead of claiming that resources are free. Automatic backend
selection forwards the same descriptor to every actual native attempt.

Direct NativeWorker launches used by physical diagnostics detect marked
managed groups, freeze symlink targets, and acquire their own pins before
spawning. They join a pending acquire on close/cancellation and cannot spawn
after closure. Ordinary physical files outside managed repositories are not
GC candidates. POSIX children inherit the protection descriptor; other
platforms conservatively retain records after owner death for inspection.
Temporary reservation records are legitimate runtime bookkeeping; model
definitions, weight bytes and source caches keep their source-preservation
checks. The model repository format and migration marker are unchanged.

Before using this version's physical reclamation, every daemon sharing the
store must be upgraded/restarted. Filesystem locks cannot detect an older
daemon or an external reader that does not use the reservation protocol.

The durable `.tllama/reclaim.json` queue records only groups retired by an
explicit definition transaction. Normal release retries this queue after
removing the owner's pin; narrowing a broad pin also retries it. Storage
initialization and later mutations retry
leftovers after a crash. Every attempt rechecks the full reference/pin set
while holding the repository lock through unlink; stale records never grant
permission by themselves. Malformed protection, unknown queue versions or
unexpected group contents are reported and preserved. Failed collection does
not turn successful worker reaping into a worker cleanup failure.
