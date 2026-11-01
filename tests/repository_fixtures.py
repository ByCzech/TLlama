"""Current-format inputs for subprocess diagnostics that start a real daemon."""

import json

from tllama.helpers.model_toml import write_model_toml
from tllama.repository_names import definition_path
from tllama.repository_runtime import ensure_format
from tllama.repository_store import RepositoryStore, prepare_group


def register_model(root, source, *, name='a', extra=''):
    """Initialize an empty store and publish real GGUF data before the probe."""
    store = RepositoryStore(root)
    ensure_format(store)
    paths = store.publish(prepare_group([source]))
    relative = paths[0].relative_to(store.root).as_posix()
    text = '[llm]\nmodel=' + json.dumps(relative) + '\n' + extra
    path = store.root / definition_path(name)
    write_model_toml(path, text)
    return path, text


def repository_snapshot(root):
    """Detect edits, replacement inodes and new paths, including disk caches.

    Reads may update atime. File contents, inode, mtime and access bits must
    remain unchanged; also retain directory names to detect empty leftovers.
    """
    result = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        value = path.stat()
        result[relative] = ((path.read_bytes(), value.st_ino, value.st_mtime_ns, value.st_mode)
                            if path.is_file() else None)
    return result
