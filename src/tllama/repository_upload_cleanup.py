"""Explicit offline staging cleanup; no TTL and no changes to published data."""

import re

from tllama.helpers.metadata_cache import get_digest_cache_path, get_metadata_cache_path
from tllama.repository_check import CheckCancelled, _cancel
from tllama.repository_cleanup import _directory, _regular
from tllama.repository_pins import snapshot
from tllama.repository_runtime import read_format
from tllama.repository_store import file_signature, sync_directory


_BLOB = re.compile('[0-9a-f]{64}')
_PARTIAL = re.compile(r'\.upload-[0-9a-f]{32}')


def prune_uploads(store, *, apply=False, stop=None):
    if type(apply) is not bool:
        raise ValueError('apply must be boolean.')
    report = {'version': 1, 'operation': 'prune-uploads', 'apply': apply,
        'ok': False, 'complete': False, 'can_apply': False, 'cancelled': False,
        'candidates': [], 'removed_files': [], 'issues': []}
    try:
        with store.lock(create=False):
            _cancel(stop)
            if read_format(store) is None:
                raise ValueError('Upload cleanup requires a marked format-2 repository.')
            pins = snapshot(store)
            if pins.active or pins.retained:
                raise ValueError('Stop all TLlama daemons and repository operations before upload cleanup; '
                                 'active or retained reservations still exist.')
            root = store.root / '.tllama/uploads'
            cache_root = store.cache_dir
            _directory(cache_root)
            signatures = {}
            if _directory(root):
                for path in sorted(root.iterdir()):
                    _cancel(stop)
                    kind = 'blob' if _BLOB.fullmatch(path.name) else (
                        'partial' if _PARTIAL.fullmatch(path.name) else None)
                    if kind is None:
                        raise ValueError(f'Unknown upload entry left unchanged: {path.name}')
                    info = _regular(path)
                    if info is None:
                        raise ValueError(f'Upload disappeared during inspection: {path.name}')
                    paths = []
                    if kind == 'blob':
                        paths.extend(cache for cache in (get_metadata_cache_path(cache_root, path),
                            get_digest_cache_path(cache_root, path)) if _regular(cache) is not None)
                    # Cache-first unlink keeps the blob name available as the
                    # retry key until every associated cache is durably gone.
                    paths.append(path)
                    for member in paths:
                        signatures[member] = file_signature(member)
                    report['candidates'].append({'path': path.relative_to(store.root).as_posix(),
                        'kind': kind, 'size': info.st_size,
                        'remove_paths': [p.relative_to(store.root).as_posix() for p in paths]})
            report['complete'] = report['can_apply'] = True
            _cancel(stop)
            if apply:
                # The entire inventory is checked before the first unlink.
                # No preview is persisted or accepted as deletion authority.
                for path, signature in signatures.items():
                    if _regular(path) is None or file_signature(path) != signature:
                        raise ValueError(f'Upload cleanup source changed: {path}')
                for path, signature in signatures.items():
                    _cancel(stop)
                    if _regular(path) is None or file_signature(path) != signature:
                        raise ValueError(f'Upload cleanup source changed: {path}')
                    path.unlink()
                    report['removed_files'].append(path.relative_to(store.root).as_posix())
                    sync_directory(path.parent)
            report['ok'] = True
    except CheckCancelled:
        report['cancelled'] = True
        report['complete'] = report['can_apply'] = False
    except (OSError, ValueError, RuntimeError) as exc:
        report['can_apply'] = False
        report['issues'].append({'severity': 'error', 'code': 'upload_cleanup_failed',
                                'path': '.tllama/uploads', 'message': str(exc)})
    report['summary'] = {'blobs': sum(row['kind'] == 'blob' for row in report['candidates']),
        'partials': sum(row['kind'] == 'partial' for row in report['candidates']),
        'staging_bytes': sum(row['size'] for row in report['candidates']),
        'removed_paths': len(report['removed_files']), 'errors': len(report['issues'])}
    return report
