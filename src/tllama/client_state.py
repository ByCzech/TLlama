"""Private client state paths, scoped by normalized server URL and model."""

import hashlib
import json
import os
from pathlib import Path
import stat

from tllama.client import server_url


def state_directory(server, model, kind):
    root = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local' / 'state')
    identity = json.dumps([server_url(server), model], ensure_ascii=False).encode('utf-8')
    return root / 'tllama' / kind / hashlib.sha256(identity).hexdigest()


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError('Client state directory must be a directory, not a symlink')
    path.chmod(0o700)


def private_file(path):
    """Create or check an application-owned private regular file."""
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NONBLOCK
                 | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError('Client state must be a regular file')
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
