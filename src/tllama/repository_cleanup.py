"""Targeted reclamation of groups retired by a definition transaction.

The durable queue contains candidates, never permission to delete. Every
attempt rechecks all definitions and reservations under the repository lock,
and keeps that lock through unlink. Unrelated orphan groups are not scanned.
"""

import json
import logging
from pathlib import PurePosixPath
import re
import stat

from tllama.helpers.metadata_cache import get_digest_cache_path, get_metadata_cache_path
from tllama.helpers.model_toml import parse_model_toml
from tllama.repository_names import definition_name
from tllama.repository_pins import snapshot
from tllama.repository_store import sync_directory


logger = logging.getLogger(__name__)
QUEUE = '.tllama/reclaim.json'
_GROUP = re.compile('[0-9a-f]{64}')
_FIRST = re.compile(r'weights-00001-of-(\d{5})\.gguf')


def _files(first):
    if first == 'weights.gguf':
        return ['weights.gguf']
    match = _FIRST.fullmatch(first)
    if not match or not 2 <= int(match[1]) <= 2048:
        raise ValueError('Not the first file of a canonical managed group.')
    count = int(match[1])
    return [f'weights-{number:05d}-of-{count:05d}.gguf' for number in range(1, count + 1)]


def _reference(value):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Unmanaged or invalid definition reference; physical cleanup skipped.')
    path = PurePosixPath(value)
    if (path.as_posix() != value or len(path.parts) != 3 or path.parts[0] != 'data'
            or not _GROUP.fullmatch(path.parts[1])):
        raise ValueError('Unmanaged or invalid definition reference; physical cleanup skipped.')
    return path.parts[1], _files(path.parts[2])


def definition_groups(path):
    spec = parse_model_toml(path.read_text(encoding='utf-8'), source=str(path))
    if spec.llm_from is not None or spec.mmproj_from is not None:
        raise ValueError(f'Unpublished definition: {path}')
    result = {}
    for value in (spec.llm_model, spec.mmproj_model, spec.runtime.get('lora_path')):
        if value is not None:
            group, files = _reference(value)
            if group in result and result[group] != files:
                raise ValueError('Inconsistent managed group references.')
            result[group] = files
    return result


def _regular(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Cleanup file must be regular, without symlinks: {path}')
    return info


def _directory(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f'Cleanup directory must be regular, without symlinks: {path}')
    return True


def _read_queue(store):
    path = store.root / QUEUE
    if _regular(path) is None:
        return {}
    with path.open(encoding='utf-8') as handle:
        raw = handle.read(8 * 1024 * 1024 + 1)
    value = json.loads(raw)
    if (len(raw) > 8 * 1024 * 1024 or not isinstance(value, dict)
            or set(value) != {'version', 'groups'} or type(value['version']) is not int
            or value['version'] != 1 or not isinstance(value['groups'], dict)):
        raise ValueError('Unknown or invalid pending cleanup format.')
    for group, files in value['groups'].items():
        if (not _GROUP.fullmatch(group) or not isinstance(files, list) or not files
                or not isinstance(files[0], str) or files != _files(files[0])):
            raise ValueError('Invalid pending cleanup group.')
    return value['groups']


def _save_queue(store, groups):
    from tllama.repository_runtime import write_json
    if groups:
        value = {'version': 1, 'groups': groups}
        if len(json.dumps(value, sort_keys=True)) + 1 > 8 * 1024 * 1024:
            raise ValueError('Pending cleanup queue exceeds the supported size.')
        write_json(store, QUEUE, value)
    else:
        (store.root / QUEUE).unlink(missing_ok=True)
        sync_directory(store.root / '.tllama')


def remember_definition(store, path):
    """Before mutation, durably remember old groups. Failure does not block rm.

    Call with the repository lock held. A crash before the definition changes
    is harmless: the next attempt still sees the old definition as a reference.
    """
    try:
        if _regular(path) is None:
            return []
        candidates = definition_groups(path)
        if not candidates:
            return []
        pending = _read_queue(store)
        for group, files in candidates.items():
            if group in pending and pending[group] != files:
                raise ValueError('Conflicting pending cleanup group.')
            pending[group] = files
        _save_queue(store, pending)
        return []
    except Exception as exc:
        return [f'Cannot schedule physical cleanup: {exc}']


def _references(store):
    references = set()
    def walk(directory):
        # Path.glob/os.walk can silently suppress access errors or skip links.
        # Explicit traversal must either read every definition or fail closed.
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f'Symlink in definition tree: {path}')
            if stat.S_ISDIR(info.st_mode):
                walk(path)
            elif path.name.lower().endswith('.toml'):
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f'Definition is not a regular file: {path}')
                definition_name(path.relative_to(store.root).as_posix())
                references.update(definition_groups(path))
    for branch in ('Local', 'TLlama', 'HuggingFace'):
        directory = store.root / branch
        if _directory(directory):
            walk(directory)
    return references


def _plan_group(store, group, files, pins):
    directory = store.root / 'data' / group
    exists = _directory(directory)
    if exists and not {path.name for path in directory.iterdir()} <= set(files):
        raise ValueError(f'Unexpected files in data/{group}; cleanup skipped.')
    weights = [directory / name for name in files]
    identities = set()
    for path in weights:
        info = _regular(path)
        if info is not None:
            identities.add((info.st_dev, info.st_ino))
    uploads = []
    upload_root = store.root / '.tllama/uploads'
    if _directory(upload_root):
        for path in sorted(upload_root.iterdir()):
            if not _GROUP.fullmatch(path.name):
                continue
            info = _regular(path)
            if info is not None and (info.st_dev, info.st_ino) in identities:
                if pins.protects_path(path.relative_to(store.root).as_posix()):
                    return None
                uploads.append(path)
    caches = [cache for path in uploads + weights for cache in
              (get_metadata_cache_path(store.cache_dir, path), get_digest_cache_path(store.cache_dir, path))]
    for path in caches:
        _regular(path)
    # Remove staging aliases first: if interrupted after a weight unlink,
    # retry can finish from the durable complete file list without its inode.
    return (uploads, caches, weights), directory if exists else None


def collect_pending(store, *, warnings=()):
    """Best-effort targeted cleanup; report failures without undoing mutations.

    Also called after reservation release and storage initialization. No queue
    means no scan; each retry revalidates the entire current protection set.
    Missing expected files are allowed after an interrupted prior unlink.
    """
    report = {'removed_groups': [], 'shared_groups': [], 'deferred_groups': [],
              'removed_files': [], 'warnings': list(warnings)}
    pending = {}
    try:
        with store.lock():
            pending = _read_queue(store)
            if not pending:
                for warning in report['warnings']:
                    logger.warning('%s', warning)
                return report
            from tllama.repository_runtime import read_format
            if read_format(store) is None:
                raise ValueError('Physical cleanup requires a marked format-2 repository.')
            for relative in ('data', '.tllama/metadata-cache'):
                _directory(store.root / relative)
            references = _references(store)
            pins = snapshot(store)
            for group, files in list(pending.items()):
                if group in references:
                    report['shared_groups'].append(group)
                    del pending[group]
                    continue
                if pins.protects_group(group):
                    report['deferred_groups'].append(group)
                    continue
                try:
                    plan = _plan_group(store, group, files, pins)
                    if plan is None:
                        report['deferred_groups'].append(group)
                        continue
                    phases, directory = plan
                    for paths in phases:
                        changed = set()
                        for path in paths:
                            if _regular(path) is not None:
                                path.unlink()
                                report['removed_files'].append(path.relative_to(store.root).as_posix())
                                changed.add(path.parent)
                        # Make staging alias removal durable before losing the
                        # last group inode from which a retry finds those aliases.
                        for parent in changed:
                            sync_directory(parent)
                    if directory is not None:
                        directory.rmdir()
                        sync_directory(directory.parent)
                    report['removed_groups'].append(group)
                    del pending[group]
                except Exception as exc:
                    report['deferred_groups'].append(group)
                    report['warnings'].append(str(exc))
            _save_queue(store, pending)
    except Exception as exc:
        report['deferred_groups'] = sorted(set(report['deferred_groups']) | set(pending))
        report['warnings'].append(f'Physical cleanup skipped: {exc}')
    for warning in report['warnings']:
        logger.warning('%s', warning)
    return report
