"""Process-owned repository reservations, shared by cooperating daemons.

Publish/change/read/remove records under the repository lock. Each record's
separate OS lock proves liveness; PIDs and wall-clock expiry are not evidence.
GC consumes a snapshot under that same repository transaction.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import tempfile

from tllama.repository_store import sync_directory


_NAME = re.compile(r'(release|retain)-[0-9a-f]{32}\.json')
_GROUP = re.compile('[0-9a-f]{64}')
_STAGING = ('.tllama/uploads', '.tllama/downloads', '.tllama/staging')
_MAX_RECORD = 1024 * 1024


class ReservationError(RuntimeError):
    """Cannot prove the complete protection set; reclamation must stop."""


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or any(ord(char) < 32 for char in value)
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise ValueError('Reservation paths must be canonical repository-relative POSIX paths.')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError('Reservation paths must stay inside the repository.')
    return value


def _document(purpose, groups, paths, all_data, retain_on_crash):
    if not isinstance(purpose, str) or not purpose or len(purpose) > 80 or any(ord(c) < 32 for c in purpose):
        raise ValueError('Reservation purpose must be a short nonempty description.')
    if type(all_data) is not bool or type(retain_on_crash) is not bool:
        raise ValueError('Reservation flags must be booleans.')
    groups, paths = list(groups), list(paths)
    if len(groups) > 4096 or len(paths) > 4096:
        raise ValueError('Reservation has too many resources.')
    if any(not isinstance(group, str) or not _GROUP.fullmatch(group) for group in groups):
        raise ValueError('Reserved groups must be lowercase SHA-256 addresses.')
    for path in paths:
        _relative(path)
        if not any(path == root or path.startswith(root + '/') for root in _STAGING):
            raise ValueError('Reserved paths must belong to repository-owned staging.')
    return {'version': 1, 'purpose': purpose, 'groups': sorted(set(groups)),
            'paths': sorted(set(paths)), 'all_data': all_data, 'retain_on_crash': retain_on_crash}


def _try_lock(fd):
    try:
        if os.name == 'nt':
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise
        return False


def _unlock(fd):
    if os.name == 'nt':
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write(fd, document):
    # Leading whitespace keeps this valid JSON while reserving byte zero for
    # Windows locking. Other processes may read the payload past that byte.
    raw = (' ' + json.dumps(document, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
    if len(raw) > _MAX_RECORD:
        raise ValueError('Reservation document is too large.')
    os.lseek(fd, 0, os.SEEK_SET)
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if not written:
            raise OSError('Reservation write made no progress.')
        offset += written
    os.ftruncate(fd, len(raw))
    os.fsync(fd)


class Reservation:
    """Own a pin until close(), or use a sync/async context manager.

    The constructor is read-only. acquire()/update()/close() share the store
    lock with publication and future GC. On POSIX inherited_fds may be passed
    explicitly to a native child: its copy then keeps the flock alive after a
    supervisor crash. Close only after every protected user has finished.
    For a worker without inherited-lock support use retain_on_crash=True;
    an unlocked leftover then remains conservatively protected for inspection.
    """

    def __init__(self, store, purpose, *, groups=(), paths=(), all_data=False, retain_on_crash=False):
        self.store = store
        self.document = _document(purpose, groups, paths, all_data, retain_on_crash)
        self.path = None
        self._fd = None
        self._owner = os.getpid()
        self._finished = False

    def _check_owner(self):
        if os.getpid() != self._owner:
            raise ReservationError('Only the creating process may change/release a reservation.')

    @property
    def inherited_fds(self):
        if self._fd is None:
            raise ReservationError('Reservation is not active.')
        return (self._fd,) if os.name == 'posix' else ()

    def acquire(self):
        self._check_owner()
        with self.store.lock():
            if self._finished:
                raise ReservationError('A released reservation cannot be reacquired.')
            if self._fd is not None:
                return self
            directory = self.store._directory('.tllama/pins')
            policy = 'retain' if self.document['retain_on_crash'] else 'release'
            path = directory / (policy + '-' + secrets.token_hex(16) + '.json')
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
                         | getattr(os, 'O_BINARY', 0), 0o666)
            try:
                # Windows locks a real first byte. The repository lock hides
                # this prepublication placeholder from all snapshot readers.
                os.write(fd, b' ')
                if not _try_lock(fd):
                    raise ReservationError('Cannot lock a newly created reservation.')
                _write(fd, self.document)
                sync_directory(directory)
            except BaseException:
                os.close(fd)
                path.unlink(missing_ok=True)
                raise
            self.path, self._fd = path, fd
        return self

    def update(self, *, groups=None, paths=None, all_data=None):
        self._check_owner()
        with self.store.lock():
            if self._fd is None:
                raise ReservationError('Reservation is not active.')
            previous = self.document
            document = _document(previous['purpose'], previous['groups'] if groups is None else groups,
                previous['paths'] if paths is None else paths,
                previous['all_data'] if all_data is None else all_data, previous['retain_on_crash'])
            # Preserve the locked inode, also when a child holds an inherited
            # descriptor. Readers use the repository lock during this write.
            _write(self._fd, document)
            self.document = document
            # Narrowing a broad resolution pin can release unrelated retired
            # groups even while this generation remains loaded indefinitely.
            from tllama.repository_cleanup import collect_pending
            collect_pending(self.store)

    def close(self, *, retry_cleanup=True):
        """Release ownership; read-only auditors opt out of deferred GC."""
        if type(retry_cleanup) is not bool:
            raise ValueError('Reservation cleanup flag must be boolean.')
        self._check_owner()
        with self.store.lock():
            if self._fd is None:
                return
            if (self.path.is_symlink() or not self.path.exists()
                    or self.path.stat().st_ino != os.fstat(self._fd).st_ino):
                raise ReservationError('Reservation file was removed or replaced outside the protocol.')
            fd, self._fd = self._fd, None
            self._finished = True
            try:
                _unlock(fd)
            finally:
                os.close(fd)
            self.path.unlink()
            sync_directory(self.path.parent)
            # A definition may have disappeared while this owner was still
            # using its snapshot. Only durable targeted candidates are retried.
            if retry_cleanup:
                from tllama.repository_cleanup import collect_pending
                collect_pending(self.store)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.close()

    async def __aenter__(self):
        from tllama.residency import _settle_owned
        try:
            return await _settle_owned(asyncio.create_task(asyncio.to_thread(self.acquire)))
        except BaseException:
            await _settle_owned(asyncio.create_task(asyncio.to_thread(self.close)))
            raise

    async def __aexit__(self, *exc):
        from tllama.residency import _settle_owned
        await _settle_owned(asyncio.create_task(asyncio.to_thread(self.close)))


@dataclass(frozen=True)
class ReservationSnapshot:
    active: tuple
    retained: tuple
    stale: tuple

    def protects_group(self, group):
        if not isinstance(group, str) or not _GROUP.fullmatch(group):
            raise ValueError('Invalid group address.')
        return any(row['all_data'] or group in row['groups'] for row in self.active + self.retained)

    def protects_path(self, relative):
        relative = _relative(relative)
        return any(relative == path or relative.startswith(path + '/')
                   for row in self.active + self.retained for path in row['paths'])


def snapshot(store, *, remove_stale=False):
    """Read live/retained protection under the repository transaction.

    A caller planning deletion must keep store.lock() held through its final
    check and unlink. This function's lock is reentrant for such a caller.
    Live or retained unreadable/unknown records abort the entire snapshot.
    Unlocked release-* entries need no payload parsing, even after a crash
    halfway through writing; their filename declares the orphan policy.
    """
    active, retained, stale = [], [], []
    with store.lock():
        directory = store.root / '.tllama/pins'
        if directory.is_symlink():
            raise ReservationError('Reservation directory must not be a symlink.')
        if not directory.exists():
            return ReservationSnapshot((), (), ())
        for path in sorted(directory.iterdir()):
            match = _NAME.fullmatch(path.name)
            if not match or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise ReservationError(f'Unknown reservation entry: {path.name}')
            fd = os.open(path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
            acquired = False
            try:
                acquired = _try_lock(fd)
                if acquired and match[1] == 'release':
                    stale.append(path.name)
                    if remove_stale:
                        # Windows cannot remove a file while this fd is open.
                        _unlock(fd)
                        os.close(fd)
                        fd = None
                        path.unlink()
                        sync_directory(directory)
                    continue
                os.lseek(fd, 1, os.SEEK_SET)
                raw = os.read(fd, _MAX_RECORD + 1)
                value = json.loads(raw)
                if len(raw) > _MAX_RECORD or not isinstance(value, dict):
                    raise ValueError('Invalid reservation document.')
                expected = _document(value['purpose'], value['groups'], value['paths'],
                                     value['all_data'], match[1] == 'retain')
                if (value != expected or type(value.get('version')) is not int
                        or type(value.get('retain_on_crash')) is not bool):
                    raise ValueError('Unknown or inconsistent reservation format.')
                (retained if acquired else active).append(value)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ReservationError(f'Cannot inspect reservation {path.name}: {exc}') from exc
            finally:
                if fd is not None:
                    if acquired:
                        _unlock(fd)
                    os.close(fd)
    return ReservationSnapshot(tuple(active), tuple(retained), tuple(stale))


@contextmanager
def temporary_directory(store, *, parent, prefix, purpose):
    """Create/pin an owned staging job atomically, then clean up before unpin."""
    if parent not in _STAGING:
        raise ValueError('Temporary reservations require an owned staging root.')
    with store.lock():
        temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=store._directory(parent))
        directory = Path(temporary.name)
        try:
            pin = Reservation(store, purpose, paths=[directory.relative_to(store.root).as_posix()]).acquire()
        except BaseException:
            temporary.cleanup()
            raise
    try:
        yield directory, pin
    finally:
        try:
            temporary.cleanup()
        finally:
            pin.close()
