"""Exclusive temporary repository files with ordinary file permissions."""

import os
from pathlib import Path
import secrets


def repository_tempfile(*, directory: Path, prefix: str, suffix: str = "",
                        mode: int = 0o666) -> tuple[int, Path]:
    """Return an open fd and unique path in the destination filesystem.

    Unlike tempfile.mkstemp's private 0600 files, repository data uses 0666
    at creation: the kernel applies the process umask and directory default
    ACLs. Never inspect the mask with os.umask(), which would briefly change
    it for every thread. O_EXCL prevents clobbering files or following a
    pre-existing symlink; callers own closing and unlinking the result.
    Replacement writers can supply existing access bits to avoid a temporary
    file that is more permissive than the document being replaced.
    Private worker sockets and credentials must not use this helper.
    """
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        path = Path(directory) / f"{prefix}{secrets.token_hex(16)}{suffix}"
        try:
            descriptor = os.open(path, flags, mode)
        except FileExistsError:
            continue
        return descriptor, path
    raise FileExistsError("Could not create a unique repository temporary file")
