"""Transactional publication of immutable, content-addressed GGUF groups.

Shared by the format-2 runtime and the offline migration tool. Merely
constructing a store or preparing inputs does not create or change any file.
All cooperating writers must use the repository lock. Locks require a backing
filesystem that implements cross-process (and, when shared, cross-host) locks.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import errno
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import threading
import time

from tllama.helpers.gguf_import import validate_group
from tllama.helpers.gguf_metadata import build_model_metadata_payload
from tllama.helpers.metadata_cache import (build_model_file_fingerprint,
    load_digest_cache, load_metadata_cache, save_digest_cache, save_metadata_cache)
from tllama.repository_layout import group_paths


def file_signature(path):
    """Identify the regular file read, without depending on its mount point."""
    value = Path(path).stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'Not a regular weight file: {path}')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def hash_file(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def measured_digest(path, cache_dir=None):
    """Reuse only current, stable caches from actual local measurements.

    Old/unknown schemas are misses through the ordinary cache loader. A digest
    claimed by a remote service is not a measurement of these local bytes.
    This read-only operation never upgrades or writes an input cache.
    """
    before = file_signature(path)
    cached = load_digest_cache(cache_dir, path) if cache_dir is not None else None
    value = (cached or {}).get('content_sha256')
    if (not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
            or cached.get('source') != 'local'):
        value = hash_file(path)
    if before != file_signature(path):
        raise ValueError(f'Weight file changed while hashing: {path}')
    return value


@dataclass(frozen=True)
class PreparedGroup:
    sources: tuple[Path, ...]
    signatures: tuple[tuple, ...]
    digests: tuple[str, ...]
    metadata: tuple[dict, ...]

    @property
    def relative_paths(self):
        return group_paths(self.digests)


def prepare_group(paths, *, names=None, cache_dir=None):
    """Validate a complete ordered group and capture its measured identity.

    Explicit names allow verified upload blobs to have hash-only staging
    filenames. They describe the original shard names, never output paths.
    """
    sources = tuple(Path(path).absolute() for path in paths)
    if not 1 <= len(sources) <= 2048:
        raise ValueError('A group requires 1..2048 files.')
    if any(path.is_symlink() for path in sources):
        raise ValueError('Weight sources must not be symlinks.')
    names = list(names) if names is not None else [path.name for path in sources]
    if len(names) != len(sources) or any(not isinstance(name, str) or '/' in name
            or '\\' in name or not name.lower().endswith('.gguf') for name in names):
        raise ValueError('Group input names must be plain GGUF basenames.')
    signatures = tuple(file_signature(path) for path in sources)
    headers = validate_group(names, sources)
    digests = tuple(measured_digest(path, cache_dir) for path in sources)
    metadata = tuple((load_metadata_cache(cache_dir, path) if cache_dir is not None else None)
                     or build_model_metadata_payload(header) for path, header in zip(sources, headers))
    if signatures != tuple(file_signature(path) for path in sources):
        raise ValueError('Weight group changed during preparation.')
    return PreparedGroup(sources, signatures, digests, metadata)


def sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RepositoryStore:
    def __init__(self, root):
        # A symlink to the whole repository is supported, unlike links from
        # its control/data directories into some unrelated filesystem tree.
        self.root = Path(root).resolve()
        self.cache_dir = self.root / '.tllama' / 'metadata-cache'
        self._mutex = threading.RLock()
        self._depth = 0

    def _directory(self, relative):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root
        for part in Path(relative).parts:
            if part in {'', '.', '..'} or Path(part).is_absolute():
                raise ValueError('Invalid repository directory.')
            path = path / part
            path.mkdir(exist_ok=True)
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f'Repository directory must not be a symlink: {path}')
        return path

    @contextmanager
    def _thread_lock(self, timeout):
        if not self._mutex.acquire(timeout=timeout):
            raise TimeoutError(f'Repository is locked: {self.root}')
        try:
            yield
        finally:
            self._mutex.release()

    @contextmanager
    def lock(self, *, timeout=30):
        """Reentrant per instance/thread, exclusive across cooperating writers.

        OS locks are released on process exit. Never delete/replace the lock
        file: another process might still hold its original inode open.
        Run synchronous transactions in an owned I/O thread, not the event loop.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Repository lock timeout must be finite and nonnegative.')
        deadline = time.monotonic() + timeout
        with self._thread_lock(timeout):
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            directory = self._directory('.tllama')
            path = directory / 'repository.lock'
            if path.is_symlink():
                raise ValueError('Repository lock must not be a symlink.')
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
                         | getattr(os, 'O_BINARY', 0), 0o666)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError('Repository lock must be a regular file.')
                if os.name == 'nt':
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b'\0')
                    def acquire():
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    def release():
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    def acquire():
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    def release():
                        fcntl.flock(fd, fcntl.LOCK_UN)
                while True:
                    try:
                        acquire()
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'Repository is locked: {self.root}') from exc
                        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                self._depth = 1
                try:
                    yield
                finally:
                    self._depth = 0
                    release()
            finally:
                os.close(fd)

    def _check_sources(self, group):
        if (any(path.is_symlink() for path in group.sources)
                or group.signatures != tuple(file_signature(path) for path in group.sources)):
            raise ValueError('Prepared weight group changed before publication.')

    def _check_existing(self, group, paths):
        directory = paths[0].parent
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Existing group is not a regular repository directory.')
        if {path.name for path in directory.iterdir()} != {path.name for path in paths}:
            raise ValueError('Existing data group has an incomplete or unexpected file set.')
        if any(path.is_symlink() for path in paths):
            raise ValueError('Existing data group contains a symlink.')
        signatures = tuple(file_signature(path) for path in paths)
        validate_group([path.name for path in paths], paths)
        if tuple(measured_digest(path, self.cache_dir) for path in paths) != group.digests:
            raise ValueError('Existing data group does not match its content address.')
        if signatures != tuple(file_signature(path) for path in paths):
            raise ValueError('Existing data group changed during verification.')

    def publish(self, group, *, link_owned=False):
        """Publish a prepared group atomically; never overwrite an existing one.

        The default copies inputs into independent files. link_owned is only
        for immutable, repository-owned upload/migration sources; it preserves
        their inodes and avoids a second multi-gigabyte copy. It does not unlink
        the inputs: an offline migration journal owns their eventual removal.
        No model definition or repository-format marker is written here.
        """
        relative = group.relative_paths
        paths = tuple(self.root / path for path in relative)
        with self.lock():
            self._check_sources(group)
            data = self._directory('data')
            if paths[0].parent.exists() or paths[0].parent.is_symlink():
                self._check_existing(group, paths)
            else:
                staging_root = self._directory('.tllama/staging')
                temporary = staging_root / ('group-' + secrets.token_hex(16))
                temporary.mkdir()
                try:
                    for source, target, digest in zip(group.sources, paths, group.digests):
                        staged = temporary / target.name
                        if link_owned:
                            if not source.resolve().is_relative_to(self.root):
                                raise ValueError('Only repository-owned files may be hard-linked into data.')
                            self._check_sources(group)
                            os.link(source, staged, follow_symlinks=False)
                            if staged.is_symlink() or not os.path.samefile(source, staged):
                                raise ValueError('Source changed during hard-link publication.')
                        else:
                            measured = sha256()
                            with source.open('rb') as reader, staged.open('xb') as writer:
                                while chunk := reader.read(4 * 1024 * 1024):
                                    writer.write(chunk)
                                    measured.update(chunk)
                                writer.flush()
                                os.fsync(writer.fileno())
                            if measured.hexdigest() != digest:
                                raise ValueError('Copied weight data does not match its prepared digest.')
                            shutil.copystat(source, staged, follow_symlinks=False)
                            with staged.open('rb') as copied:
                                os.fsync(copied.fileno())
                    self._check_sources(group)
                    sync_directory(temporary)
                    os.rename(temporary, paths[0].parent)
                    sync_directory(data)
                    sync_directory(staging_root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            self._directory('.tllama/metadata-cache')
            for path, digest, metadata in zip(paths, group.digests, group.metadata):
                fingerprint = build_model_file_fingerprint(path)
                name = path.relative_to(self.root).as_posix()
                save_digest_cache(self.cache_dir, name, path,
                                  {'content_sha256': digest, 'source': 'local'}, fingerprint=fingerprint)
                save_metadata_cache(self.cache_dir, name, path, metadata, fingerprint=fingerprint)
        return paths
