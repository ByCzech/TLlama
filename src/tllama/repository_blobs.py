"""Resolve verified blob bytes from staging or complete published data groups.

No second persistent index: current locally measured digest caches are the
fast path. Missing caches can be rebuilt from bytes, never from filenames.
Callers keep the repository lock through use or acquire a group reservation.
"""

import re

from tllama.helpers.metadata_cache import (build_model_file_fingerprint,
    get_digest_cache_path, get_metadata_cache_path, load_digest_cache, save_digest_cache)
from tllama.repository_cleanup import _directory, _files, _regular
from tllama.repository_info import validate_group_files
from tllama.repository_pins import snapshot
from tllama.repository_store import measured_digest, prepare_group, sync_directory


_DIGEST = re.compile('[0-9a-f]{64}')


def verify_file(store, path, digest):
    if _regular(path) is None:
        raise FileNotFoundError(path)
    before = build_model_file_fingerprint(path)
    if measured_digest(path, store.cache_dir) != digest or before != build_model_file_fingerprint(path):
        raise ValueError('A blob changed or does not match its digest.')
    save_digest_cache(store.cache_dir, str(path), path,
        {'content_sha256': digest, 'source': 'local'}, fingerprint=before, repository=store)
    return path, before


def _groups(store):
    root = store.root / 'data'
    if not _directory(root):
        return
    for directory in sorted(root.iterdir()):
        if not _DIGEST.fullmatch(directory.name):
            continue
        if not _directory(directory):
            continue
        names = sorted(path.name for path in directory.iterdir() if path.name.endswith('.gguf'))
        try:
            if not names or names != _files(names[0]):
                continue
            validate_group_files(directory, names)
            for name in names:
                if _regular(directory / name) is None:
                    raise ValueError('Missing published shard.')
        except (OSError, ValueError):
            continue
        yield tuple(directory / name for name in names)


def find_published_blob(store, digest):
    """Find one file, including a continuation shard, in a complete group."""
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError('Invalid blob digest.')
    _directory(store.cache_dir)
    cold = []
    for paths in _groups(store):
        # For a single-file group the address narrows the candidate directly;
        # it still requires byte/cache and header verification below.
        if len(paths) == 1 and paths[0].parent.name != digest:
            continue
        cached = [load_digest_cache(store.cache_dir, path) for path in paths]
        values = [(item or {}).get('content_sha256') if (item or {}).get('source') == 'local'
                  else None for item in cached]
        if digest in values:
            found = _verify_group(store, paths, digest)
            if found is not None:
                return found
        elif any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in values):
            cold.append(paths)
    # Warm groups with other known digests never require a weight read. A cold
    # repository may need one full scan; save each actual measurement for reuse.
    for paths in cold:
        found = _verify_group(store, paths, digest)
        if found is not None:
            return found
    return None


def _verify_group(store, paths, digest):
    try:
        fingerprints = [build_model_file_fingerprint(path) for path in paths]
        group = prepare_group(paths, cache_dir=store.cache_dir)
        if tuple(store.root / relative for relative in group.relative_paths) != paths:
            return None
        if fingerprints != [build_model_file_fingerprint(path) for path in paths]:
            return None
        for path, value, fingerprint in zip(paths, group.digests, fingerprints):
            save_digest_cache(store.cache_dir, str(path), path,
                {'content_sha256': value, 'source': 'local'}, fingerprint=fingerprint, repository=store)
        return paths[group.digests.index(digest)] if digest in group.digests else None
    except (OSError, ValueError):
        return None


def remove_published_uploads(store):
    """Best-effort removal of redundant aliases; never delete the data source.

    Only verified equal bytes in a complete data group justify unlink. File
    identity alone is insufficient after a move across filesystems.
    """
    warnings = []
    with store.lock():
        try:
            root = store.root / '.tllama/uploads'
            if not _directory(root):
                return warnings
            _directory(store.cache_dir)
            pins = snapshot(store)
            for path in sorted(root.iterdir()):
                if not _DIGEST.fullmatch(path.name):
                    continue
                if pins.protects_path(path.relative_to(store.root).as_posix()):
                    continue
                try:
                    published = find_published_blob(store, path.name)
                    if published is None:
                        continue
                    verify_file(store, path, path.name)
                    caches = [get_digest_cache_path(store.cache_dir, path),
                              get_metadata_cache_path(store.cache_dir, path)]
                    for cache in caches:
                        _regular(cache)
                    for cache in caches:
                        cache.unlink(missing_ok=True)
                    sync_directory(store.cache_dir)
                    # Keep the alias as the retry key until its caches are
                    # durably gone. Interruption cannot strand an orphan cache.
                    path.unlink()
                    sync_directory(root)
                except (OSError, ValueError) as exc:
                    warnings.append(f'Cannot remove redundant upload {path.name}: {exc}')
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f'Upload cleanup skipped: {exc}')
    return warnings
