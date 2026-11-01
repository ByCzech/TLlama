"""Optional UTF-8 original filenames, one line per name beside each GGUF.

Companions never contribute to content/model identity. Callers serialize
publication and merging with the ordinary repository lock.
"""

import errno
import os
from pathlib import Path
import stat

from tllama.helpers.repository_files import repository_tempfile


def info_path(weight):
    return Path(weight).with_suffix('.inf')


def validate_original_name(name):
    if (not isinstance(name, str) or name in {'', '.', '..'} or '/' in name
            or '\\' in name or any(c in name for c in '\r\n\0')):
        raise ValueError('Original weight names must be plain, single-line basenames.')
    name.encode('utf-8')


def regular_optional(path):
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f'Weight companion must be a regular file without symlinks: {path}')
    return metadata


def validate_group_files(directory, names):
    """Require every weight; permit only corresponding regular .inf files."""
    names = set(names)
    companions = {info_path(Path(name)).name for name in names}
    actual = {path.name for path in directory.iterdir()}
    if not names <= actual or not actual <= names | companions:
        raise ValueError('Incomplete or unexpected managed group file set.')
    for name in actual & companions:
        regular_optional(directory / name)


def append_original_names(weight, names):
    """Keep old bytes, add absent literal names, atomically replace if needed."""
    names = list(names)
    for name in names:
        validate_original_name(name)
    if not names:
        return
    path = info_path(weight)
    existing = regular_optional(path)
    raw = path.read_bytes() if existing is not None else b''
    present = {line.removesuffix('\r') for line in raw.decode('utf-8').split('\n')}
    additions = []
    for name in names:
        if name not in present:
            present.add(name)
            additions.append(name)
    if not additions:
        return
    raw += b'\n' if raw and not raw.endswith(b'\n') else b''
    raw += ('\n'.join(additions) + '\n').encode('utf-8')
    template = path if existing is not None else Path(weight)
    metadata = template.stat()
    acl = None
    if hasattr(os, 'getxattr'):
        try:
            acl = os.getxattr(template, 'system.posix_acl_access')
        except OSError as exc:
            if exc.errno not in {errno.ENODATA, errno.ENOTSUP, getattr(errno, 'ENOATTR', errno.ENODATA)}:
                raise
    descriptor, temporary = repository_tempfile(directory=path.parent, prefix='.weight-info-', suffix='.tmp', mode=0o600)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            if hasattr(os, 'fchown'):
                current = os.fstat(handle.fileno())
                if (current.st_uid, current.st_gid) != (metadata.st_uid, metadata.st_gid):
                    os.fchown(handle.fileno(), metadata.st_uid, metadata.st_gid)
            os.chmod(temporary, stat.S_IMODE(metadata.st_mode) & 0o777)
            if hasattr(os, 'setxattr'):
                if acl is not None:
                    os.setxattr(temporary, 'system.posix_acl_access', acl)
                else:
                    try:
                        os.removexattr(temporary, 'system.posix_acl_access')
                    except OSError as exc:
                        if exc.errno not in {errno.ENODATA, errno.ENOTSUP, getattr(errno, 'ENOATTR', errno.ENODATA)}:
                            raise
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        from tllama.repository_store import sync_directory
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
