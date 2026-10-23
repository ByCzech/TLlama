"""Private runtime storage and credentials for an owned local worker."""

from __future__ import annotations

import errno
import os
import secrets
import tempfile
from pathlib import Path


def runtime_directory() -> tempfile.TemporaryDirectory:
    """Prefer a service/session runtime directory with verified private access."""
    candidates = [*os.environ.get('RUNTIME_DIRECTORY', '').split(':'),
                  os.environ.get('XDG_RUNTIME_DIR', '')]
    for candidate in candidates:
        path = Path(candidate)
        if not candidate or not path.is_absolute():
            continue
        try:
            info = path.stat()
            if not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
                continue
            return tempfile.TemporaryDirectory(prefix='tllama-worker-', dir=path)
        except OSError:
            continue
    return tempfile.TemporaryDirectory(prefix='tllama-worker-')


class WorkerCredentials:
    """Pass a fresh bearer token without placing its value in argv or env."""

    def __init__(self, directory: Path):
        self.token = secrets.token_urlsafe(32)
        self.fd: int | None = None
        self.path = directory / 'api-key'
        self._file: Path | None = None
        # The existing native --api-key-file reader accepts /proc/self/fd/N.
        # Only this descriptor is inherited, and the parent closes its copy
        # after spawn. The memfd has no persistent filesystem name; seal it
        # where Python exposes the kernel sealing constants.
        if hasattr(os, 'memfd_create') and Path('/proc/self/fd').is_dir():
            try:
                import fcntl

                self.fd = os.memfd_create('tllama-worker-key', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                os.fchmod(self.fd, 0o600)
                with os.fdopen(os.dup(self.fd), 'w') as file:
                    file.write(self.token + '\n')
                os.lseek(self.fd, 0, os.SEEK_SET)
                if hasattr(fcntl, 'F_ADD_SEALS'):
                    fcntl.fcntl(self.fd, fcntl.F_ADD_SEALS,
                                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                self.path = Path(f'/proc/self/fd/{self.fd}')
                return
            except OSError as exc:
                self.close_fd()
                if exc.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EPERM, errno.EACCES, errno.EINVAL}:
                    raise
            except BaseException:
                self.close_fd()
                raise
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = self.path
        try:
            with os.fdopen(fd, 'w') as file:
                file.write(self.token + '\n')
        except BaseException:
            self.close()
            raise

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self.fd,) if self.fd is not None else ()

    def close_fd(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def close(self):
        self.close_fd()
        if self._file is not None:
            self._file.unlink(missing_ok=True)
            self._file = None
