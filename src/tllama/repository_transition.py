"""Reject converted/partially converted stores until the runtime format switch.

This guard intentionally does not initialize or migrate anything. It prevents
the current legacy writer from reopening a store converted by the offline tool.
The format-2 runtime will replace this guard with its ordinary format check.
"""

from pathlib import Path

from tllama.config import ConfigError


def require_legacy_repository(root):
    root = Path(root)
    # The private journal may belong to an administrator and be unreadable
    # to the daemon account. Detect its directory without traversing it.
    paths = (root / '.tllama' / 'repository.json', root / '.tllama' / 'migration-v2')
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ConfigError('Repository has a format-2 marker or migration control directory. '
                          'Keep daemons stopped, finish/resume the offline conversion, '
                          'and use a format-2-capable TLlama runtime before serving it.')
