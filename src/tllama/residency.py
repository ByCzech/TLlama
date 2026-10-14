"""Model policy over owned native worker generations, independent of routes."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import itertools
import logging
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Mapping, Callable, Awaitable

from tllama.native_config import ResolvedWorkerConfig, _plain
from tllama.helpers.common import never_expires_at, NEVER_EXPIRES_SECONDS
from tllama.native_params import freeze_values
from tllama.worker import NativeWorker, WorkerError, WorkerExited


logger = logging.getLogger(__name__)


class ResidencyClosed(WorkerError):
    pass


class ModelUnloaded(WorkerError):
    pass


class ModelCapacityError(WorkerError):
    pass


class ModelCleanupError(WorkerError):
    pass


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    resolved: ResolvedWorkerConfig
    fingerprint: Any = None
    file_info: Mapping = field(default_factory=dict)
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A model definition needs a public model name")
        object.__setattr__(self, "fingerprint", freeze_values(self.fingerprint))
        object.__setattr__(self, "file_info", freeze_values(self.file_info))
        object.__setattr__(self, "metadata", freeze_values(self.metadata))

    def same_version(self, other: ModelDefinition) -> bool:
        return (self.name == other.name and self.resolved == other.resolved
                and self.fingerprint == other.fingerprint)


@dataclass(frozen=True)
class ModelLease:
    """A definition snapshot and worker protected until its context exits."""

    generation: int
    definition: ModelDefinition
    worker: NativeWorker

    @property
    def resolved(self):
        return self.definition.resolved


@dataclass(eq=False)
class _Generation:
    number: int
    definition: ModelDefinition
    worker: NativeWorker
    refs: int = 0
    state: str = "loading"
    loaded: bool = False
    retired: bool = False
    reason: str | None = None
    error: BaseException | None = None
    keep_alive: float | None = 300
    expires_at: float | None = None
    load_task: asyncio.Task | None = None
    close_task: asyncio.Task | None = None
    watch_task: asyncio.Task | None = None
    exit_recorded: bool = False


@dataclass(eq=False)
class _Intent:
    name: str
    aborted: bool = False
    entry: _Generation | None = None
    load_failure: tuple[ModelDefinition, BaseException] | None = None


async def _settle_owned(task):
    """Let an owned cleanup finish even when its caller is cancelled."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class NativeResidency:
    """One current version per virtual name; retired versions still use capacity.

    References are reserved before awaiting a shared load and retained while
    the caller waits on the worker's single transport slot. There is no second
    inference queue here. Call expire() from the owner's janitor.
    """

    def __init__(self, resolver: Callable[[str, int | None], Awaitable[ModelDefinition]], *,
                 max_loaded_models: int = 1, worker_factory=NativeWorker, clock=time.monotonic,
                 generation_numbers=None):
        if type(max_loaded_models) is not int or max_loaded_models < 1:
            raise ValueError("max_loaded_models must be a positive integer")
        self._resolve = resolver
        self._factory = worker_factory
        self._clock = clock
        self.max_loaded_models = max_loaded_models
        self._condition = asyncio.Condition()
        self._numbers = generation_numbers if generation_numbers is not None else itertools.count(1)
        self._entries: dict[int, _Generation] = {}
        self._current: dict[str, _Generation] = {}
        self._intents: set[_Intent] = set()
        self._closing = False
        self._shutdown_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self.failures = deque(maxlen=32)

    @property
    def closed(self):
        return self._closing

    @property
    def occupied(self):
        return len(self._entries)

    def _owned(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Still re-raised for callers awaiting this task.

        task.add_done_callback(finished)
        return task

    def _ensure_open(self):
        if self._closing:
            raise ResidencyClosed("Native model manager is shutting down")

    def _forget_current(self, entry):
        if self._current.get(entry.definition.name) is entry:
            del self._current[entry.definition.name]

    def _record_failure(self, entry, phase, error):
        self.failures.append({"generation": entry.number, "model": entry.definition.name,
                              "phase": phase, "error": str(error)[:8192]})
        logger.error("Native model %s generation %s %s: %s", entry.definition.name, entry.number, phase, error)

    def _finish_closed(self, entry):
        entry.state = "closed"
        self._forget_current(entry)
        self._entries.pop(entry.number, None)
        self._condition.notify_all()

    def _schedule_close(self, entry):
        if entry.close_task is None:
            entry.state = "closing"
            entry.close_task = self._owned(self._close_entry(entry))

    def _retire(self, entry, reason, *, force=False):
        entry.retired = True
        entry.reason = entry.reason or reason
        entry.expires_at = None
        self._forget_current(entry)
        if entry.state not in ("closing", "cleanup_failed", "closed"):
            entry.state = "draining"
        if not entry.loaded:
            # Replacements let already-admitted callers finish loading their
            # snapshot. Explicit unload/shutdown cancels all loading waiters.
            if force or not entry.refs:
                entry.error = ModelUnloaded(f"Model {entry.definition.name!r} was retired during load ({reason})")
                if entry.load_task is not None and not entry.load_task.done():
                    entry.load_task.cancel()
                self._schedule_close(entry)  # Also handles cancellation before the load task starts.
        elif force or not entry.refs:
            self._schedule_close(entry)
        self._condition.notify_all()

    def _expire_locked(self):
        now = self._clock()
        for entry in list(self._entries.values()):
            if (not entry.retired and entry.loaded and not entry.refs
                    and entry.expires_at is not None and entry.expires_at <= now):
                self._retire(entry, "TTL expired")

    async def _load_entry(self, entry):
        try:
            await entry.worker.start()
        except BaseException as exc:
            failure = entry.error or exc
            try:
                await entry.worker.close()
            except BaseException as cleanup:
                async with self._condition:
                    self._forget_current(entry)
                    entry.retired, entry.state, entry.error = True, "cleanup_failed", cleanup
                    self._record_failure(entry, "cleanup", cleanup)
                    self._condition.notify_all()
                raise ModelCleanupError(f"Could not clean failed generation {entry.number}: {cleanup}") from cleanup
            async with self._condition:
                entry.error = failure
                self._finish_closed(entry)
                if not isinstance(failure, (asyncio.CancelledError, ModelUnloaded)):
                    self._record_failure(entry, "load", failure)
                    # Also notify callers whose definition I/O started before
                    # this load failed but has not reached admission yet.
                    for intent in self._intents:
                        if intent.name == entry.definition.name and intent.entry is None:
                            intent.load_failure = (entry.definition, failure)
            if isinstance(failure, asyncio.CancelledError):
                raise ModelUnloaded(f"Model {entry.definition.name!r} load was cancelled") from exc
            raise failure
        async with self._condition:
            entry.loaded = True
            entry.state = "draining" if entry.retired else "ready"
            entry.watch_task = self._owned(self._watch_exit(entry))
            if not entry.refs:
                if entry.retired or self._closing or entry.keep_alive == 0:
                    self._retire(entry, entry.reason or "zero keep_alive")
                else:
                    entry.expires_at = None if entry.keep_alive is None else self._clock() + entry.keep_alive
            self._condition.notify_all()
        return entry

    async def _watch_exit(self, entry):
        code = await entry.worker.wait_for_exit()
        async with self._condition:
            if entry.state not in ("closed", "closing", "cleanup_failed"):
                entry.error = WorkerExited(code, entry.worker.diagnostics)
                if not entry.exit_recorded:
                    self._record_failure(entry, "exit", entry.error)
                    entry.exit_recorded = True
                self._retire(entry, "worker exited", force=True)
            self._condition.notify_all()

    async def _close_entry(self, entry):
        try:
            if entry.load_task is not None and not entry.load_task.done():
                await asyncio.gather(entry.load_task, return_exceptions=True)
            await entry.worker.close()
        except BaseException as exc:
            async with self._condition:
                entry.state, entry.error = "cleanup_failed", exc
                self._record_failure(entry, "cleanup", exc)
                self._condition.notify_all()
            raise ModelCleanupError(f"Could not clean generation {entry.number}: {exc}") from exc
        async with self._condition:
            self._finish_closed(entry)

    async def _release(self, entry):
        async with self._condition:
            entry.refs -= 1
            assert entry.refs >= 0
            if not entry.refs and entry.number in self._entries:
                if entry.retired:
                    if entry.state not in ("closing", "cleanup_failed"):
                        self._retire(entry, entry.reason or "retired")
                elif entry.loaded:
                    if entry.keep_alive == 0:
                        self._retire(entry, "zero keep_alive")
                    else:
                        entry.expires_at = None if entry.keep_alive is None else self._clock() + entry.keep_alive
            self._condition.notify_all()

    async def _claim(self, name, num_ctx, keep_alive):
        intent = _Intent(name)
        claimed = None
        async with self._condition:
            self._ensure_open()
            self._intents.add(intent)
        try:
            claimed = await self._claim_intent(intent, num_ctx, keep_alive)
            return claimed
        finally:
            try:
                await _settle_owned(self._owned(self._forget_intent(intent)))
            except BaseException:
                # Cancellation after admission but before lease delivery must
                # release the reference which the caller never received.
                if claimed is not None:
                    await _settle_owned(self._owned(self._release(claimed)))
                raise

    async def _forget_intent(self, intent):
        async with self._condition:
            self._intents.discard(intent)

    async def _claim_intent(self, intent, num_ctx, keep_alive):
        name = intent.name
        while True:
            self._ensure_open()
            try:
                definition = await self._resolve(name, num_ctx)
            except Exception:
                await self.unload(name)
                raise
            if definition.name != name:
                raise ValueError("Resolver returned a different public model name")
            async with self._condition:
                self._ensure_open()
                if intent.aborted:
                    raise ModelUnloaded(f"Model {name!r} was unloaded before admission")
                if intent.load_failure is not None and intent.load_failure[0].same_version(definition):
                    raise intent.load_failure[1]
                self._expire_locked()
                entry = self._current.get(name)
                if entry is not None and not entry.definition.same_version(definition):
                    self._retire(entry, "definition changed")
                    entry = None
                if entry is not None and entry.loaded and entry.worker.state != "ready":
                    entry.error = self._worker_failure(entry)
                    if not entry.exit_recorded:
                        self._record_failure(entry, "exit", entry.error)
                        entry.exit_recorded = True
                    self._retire(entry, "worker exited", force=True)
                    entry = None
                if entry is None:
                    if len(self._entries) >= self.max_loaded_models:
                        if any(e.state == "cleanup_failed" for e in self._entries.values()):
                            raise ModelCleanupError("Native capacity is held by a generation whose cleanup failed")
                        if self.max_loaded_models == 1:
                            for other in list(self._entries.values()):
                                if not other.retired:
                                    self._retire(other, "single-model replacement")
                        if any(e.retired or not e.loaded for e in self._entries.values()):
                            await self._condition.wait()
                            continue
                        raise ModelCapacityError(
                            f"Loaded model limit reached ({self.max_loaded_models}). "
                            "Unload a model first or increase TLLAMA_MAX_LOADED_MODELS."
                        )
                    worker = self._factory(definition.resolved.launch)
                    entry = _Generation(next(self._numbers), definition, worker)
                    self._entries[entry.number] = entry
                    self._current[name] = entry
                    entry.load_task = self._owned(self._load_entry(entry))
                entry.refs += 1
                intent.entry = entry
                entry.keep_alive = keep_alive  # Latest admitted request sets policy.
                entry.expires_at = None
            try:
                try:
                    await asyncio.shield(entry.load_task)
                except asyncio.CancelledError:
                    # A task cancelled before its coroutine begins cannot
                    # translate operator cancellation in _load_entry itself.
                    if entry.load_task.cancelled() and entry.error is not None:
                        raise entry.error
                    raise
                async with self._condition:
                    self._ensure_open()
                    if entry.error is not None:
                        raise entry.error
                    if entry.worker.state != "ready":
                        raise self._worker_failure(entry)
                return entry
            except BaseException:
                await _settle_owned(self._owned(self._release(entry)))
                raise

    @asynccontextmanager
    async def lease(self, name: str, *, num_ctx: int | None = None, keep_alive_seconds: float | None = 300):
        if keep_alive_seconds is not None and (
            isinstance(keep_alive_seconds, bool) or not isinstance(keep_alive_seconds, (int, float))
            or not math.isfinite(keep_alive_seconds) or keep_alive_seconds < 0
        ):
            raise ValueError("keep_alive_seconds must be nonnegative and finite, or None for pinned")
        entry = await self._claim(name, num_ctx, keep_alive_seconds)
        try:
            yield ModelLease(entry.number, entry.definition, entry.worker)
        finally:
            await _settle_owned(self._owned(self._release(entry)))

    async def preload(self, name: str, *, num_ctx=None, keep_alive_seconds=300):
        if keep_alive_seconds == 0 and not isinstance(keep_alive_seconds, bool):
            await self.unload(name)
            return None
        async with self.lease(name, num_ctx=num_ctx, keep_alive_seconds=keep_alive_seconds) as lease:
            return lease.generation

    def _worker_failure(self, entry):
        code = entry.worker.process.returncode if entry.worker.process is not None else None
        if code is not None:
            return WorkerExited(code, entry.worker.diagnostics)
        return WorkerError(f"Native generation {entry.number} is unavailable ({entry.worker.state})")

    async def unload(self, name: str) -> bool:
        async with self._condition:
            entries = [e for e in self._entries.values() if e.definition.name == name]
            intents = [i for i in self._intents if i.name == name and i.entry is None]
            for intent in intents:
                intent.aborted = True
            for entry in entries:
                self._retire(entry, "explicit unload", force=not entry.loaded)
            self._condition.notify_all()
            return bool(entries or intents)

    async def expire(self):
        async with self._condition:
            if not self._closing:
                self._expire_locked()

    async def wait_closed(self, generation: int):
        async with self._condition:
            while generation in self._entries:
                entry = self._entries[generation]
                if entry.state == "cleanup_failed":
                    raise ModelCleanupError(f"Generation {generation} cleanup failed: {entry.error}")
                await self._condition.wait()

    def snapshot(self) -> list[dict]:
        now = self._clock()
        return [{"generation": e.number, "model": e.definition.name, "state": e.state,
                 "leases": e.refs, "pid": e.worker.pid,
                 "retirement_reason": e.reason, "keep_alive": e.keep_alive,
                 "ttl_remaining": None if e.expires_at is None else max(0, e.expires_at - now),
                 "requested_context": e.definition.resolved.launch.n_ctx,
                 "effective_context": (e.worker.props or {}).get("default_generation_settings", {}).get("n_ctx") if e.loaded else None,
                 "error": str(e.error) if e.error is not None else None} for e in self._entries.values()]

    def reporting_snapshot(self) -> list[dict]:
        """Copy owned generations without a lease, I/O, or changes to TTL.

        Metadata belongs to the admitted definition even when files change or
        disappear. Closing/draining generations remain visible until reaped.
        """
        now = self._clock()
        wall = datetime.now(timezone.utc)
        result = []
        for entry in self._entries.values():
            worker = entry.worker
            process = worker.process
            # A dead child has no buffers, even before the exit observer runs.
            if process is not None and process.returncode is not None:
                continue
            if entry.retired:
                expires = wall.isoformat()
            elif entry.expires_at is not None:
                expires = (wall + timedelta(seconds=min(NEVER_EXPIRES_SECONDS, max(0, entry.expires_at - now)))).isoformat()
            elif entry.keep_alive is None:
                expires = never_expires_at()
            else:
                expires = (wall + timedelta(seconds=min(NEVER_EXPIRES_SECONDS, entry.keep_alive))).isoformat()
            result.append({
                'generation':entry.number, 'model':entry.definition.name, 'state':entry.state,
                'pid':worker.pid, 'leases':entry.refs, 'expires_at':expires,
                'retirement_reason':entry.reason, 'file_info':_plain(entry.definition.file_info),
                'metadata':_plain(entry.definition.metadata),
                'context_length':(worker.props or {}).get('default_generation_settings', {}).get('n_ctx') if entry.loaded else None,
                'memory':worker.memory.snapshot(),
            })
        return result

    async def shutdown(self):
        self._closing = True
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await _settle_owned(self._shutdown_task)

    async def _shutdown(self):
        async with self._condition:
            entries = list(self._entries.values())
            for entry in entries:
                self._retire(entry, "shutdown", force=True)
            tasks = [task for e in entries for task in (e.load_task, e.close_task) if task is not None]
            self._condition.notify_all()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Worker close owns reaping; observers must not outlive shutdown even
        # when an OS cleanup failure kept a generation quarantined.
        for entry in entries:
            if entry.watch_task is not None and not entry.watch_task.done():
                entry.watch_task.cancel()
        await asyncio.gather(*(e.watch_task for e in entries if e.watch_task is not None), return_exceptions=True)
        async with self._condition:
            if self._entries:
                raise ModelCleanupError("Native shutdown could not clean every owned generation")
