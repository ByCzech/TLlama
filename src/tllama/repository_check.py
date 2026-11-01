"""Read-only repository inventory and optional independent content verification.

A report describes a snapshot, never authorization for a later deletion.
Full hashing holds a temporary reservation, not the global repository lock.
Neither cache refresh nor pending/stale cleanup is part of this operation.
"""

from hashlib import sha256
import json
import os
from pathlib import PurePosixPath
import re
import stat

from tllama.helpers.gguf_import import validate_group
from tllama.helpers.metadata_cache import SCHEMA_VERSION, DIGEST_SCHEMA_VERSION, CACHE_FORMAT_VERSION
from tllama.repository_cleanup import definition_groups, _files, _read_queue
from tllama.repository_layout import group_paths
from tllama.repository_names import definition_name, check_portable_collisions
from tllama.repository_pins import Reservation, snapshot
from tllama.repository_runtime import read_format
from tllama.repository_store import file_signature


_GROUP = re.compile('[0-9a-f]{64}')
_CACHE = re.compile(r'([0-9a-f]{64})(\.digest)?\.json')
_WEIGHT = re.compile(r'weights(?:-(\d{5})-of-(\d{5}))?\.gguf')


class CheckCancelled(Exception):
    """Cooperative interruption; the owning request still joins cleanup."""


def _cancel(stop):
    if stop is not None and stop.is_set():
        raise CheckCancelled('Repository check cancelled.')


def _signature(path):
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _hash_file(path, expected, stop):
    _cancel(stop)
    if file_signature(path) != expected:
        raise ValueError('Weight file changed before hashing.')
    digest = sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != expected:
            raise ValueError('Weight file changed while opening it.')
        while chunk := handle.read(4 * 1024 * 1024):
            _cancel(stop)
            digest.update(chunk)
    _cancel(stop)
    if file_signature(path) != expected:
        raise ValueError('Weight file changed while hashing.')
    return digest.hexdigest()


class _Inventory:
    def __init__(self, store, hashes, stop):
        self.store, self.stop = store, stop
        self.tracked, self.jobs, self.refs = {}, [], {}
        self.pins = None
        self.report = {'version': 1, 'ok': False, 'hashes': hashes,
            'complete': True, 'references_complete': True, 'reservations_complete': True,
            'definitions': [], 'groups': [], 'caches': [], 'staging': [],
            'reservations': {'active': [], 'retained': [], 'stale': []},
            'pending_cleanup': [], 'issues': []}

    def issue(self, code, path, error, *, severity='error'):
        self.report['issues'].append({'severity': severity, 'code': code,
            'path': path.relative_to(self.store.root).as_posix(), 'message': str(error)})

    def children(self, path, *, required=False, track=False):
        _cancel(self.stop)
        try:
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError('Expected a directory without a symlink.')
            if track:
                self.tracked[path] = _signature(path)
            return sorted(path.iterdir())
        except FileNotFoundError as exc:
            if not required:
                return []
            self.issue('directory_missing', path, exc)
        except OSError as exc:
            self.issue('directory_unreadable', path, exc)
        except ValueError as exc:
            self.issue('directory_invalid', path, exc)
        self.report['complete'] = False
        return None

    def definitions(self):
        names = []
        def walk(directory):
            children = self.children(directory, required=True, track=True)
            if children is None:
                self.report['references_complete'] = False
                return
            for path in children:
                _cancel(self.stop)
                try:
                    mode = path.lstat().st_mode
                    if stat.S_ISLNK(mode):
                        raise ValueError('Symlinks in definition trees prevent a complete reference inventory.')
                    if stat.S_ISDIR(mode):
                        walk(path)
                    elif path.name.lower().endswith('.toml'):
                        if not stat.S_ISREG(mode):
                            raise ValueError('Definition must be a regular file.')
                        relative = path.relative_to(self.store.root).as_posix()
                        names.append(relative)
                        name = definition_name(relative)
                        before = _signature(path)
                        groups = definition_groups(path)
                        if before != _signature(path):
                            raise ValueError('Definition changed during inspection.')
                        self.tracked[path] = before
                        self.report['definitions'].append({'name': name, 'path': relative,
                                                          'groups': sorted(groups)})
                        for group, files in groups.items():
                            current = self.refs.setdefault(group, {'files': files, 'names': []})
                            if current['files'] != files:
                                raise ValueError('Definitions disagree about the group file set.')
                            current['names'].append(name)
                    else:
                        self.issue('definition_entry_unexpected', path,
                                   'Non-definition entry left unchanged.', severity='warning')
                except (OSError, ValueError, UnicodeError) as exc:
                    self.issue('definition_invalid', path, exc)
                    self.report['references_complete'] = False
        for branch in ('Local', 'TLlama', 'HuggingFace'):
            walk(self.store.root / branch)
        try:
            check_portable_collisions(names)
        except ValueError as exc:
            self.issue('definition_collision', self.store.root, exc)
            self.report['references_complete'] = False

    def protections(self):
        try:
            if self.children(self.store.root / '.tllama/pins', required=True) is None:
                raise ValueError('Reservation directory is unavailable.')
            self.pins = snapshot(self.store)
            self.report['reservations'] = {'active': list(self.pins.active),
                'retained': list(self.pins.retained), 'stale': list(self.pins.stale)}
            for name in self.pins.stale:
                self.issue('stale_reservation', self.store.root / '.tllama/pins' / name,
                           'No live owner; record left unchanged.', severity='warning')
            if self.pins.retained:
                self.issue('retained_reservation', self.store.root / '.tllama/pins',
                           'Retained records still protect resources after owner exit.', severity='warning')
        except (OSError, ValueError, RuntimeError) as exc:
            self.report['reservations_complete'] = False
            self.issue('reservation_invalid', self.store.root / '.tllama/pins', exc)
        try:
            self.report['pending_cleanup'] = sorted(_read_queue(self.store))
        except (OSError, ValueError, TypeError) as exc:
            self.report['complete'] = False
            self.issue('cleanup_queue_invalid', self.store.root / '.tllama/reclaim.json', exc)

    def groups(self):
        children = self.children(self.store.root / 'data', required=True, track=True)
        seen = set()
        for path in children or []:
            _cancel(self.stop)
            group = path.name
            if not _GROUP.fullmatch(group):
                self.issue('data_entry_invalid', path, 'Expected a lowercase SHA-256 group directory.')
                continue
            seen.add(group)
            referenced = self.refs.get(group, {}).get('names', [])
            protected = self.pins.protects_group(group) if self.pins is not None else None
            usage = ('referenced' if referenced else 'reserved' if protected else 'orphan'
                     if self.report['references_complete'] and self.report['reservations_complete'] else 'unknown')
            row = {'id': group, 'path': 'data/' + group, 'files': [], 'references': sorted(referenced),
                   'reserved': protected, 'usage': usage, 'valid': False, 'hash_verified': False}
            self.report['groups'].append(row)
            try:
                members = self.children(path, required=True, track=True)
                if members is None:
                    continue
                names = [member.name for member in members]
                first = 'weights.gguf' if 'weights.gguf' in names else (names[0] if names else '')
                expected = _files(first)
                if names != expected:
                    raise ValueError('Incomplete or unexpected canonical group file set.')
                signatures = []
                for member in members:
                    if member.is_symlink():
                        raise ValueError('Weight files must not be symlinks.')
                    signatures.append(file_signature(member))
                    self.tracked[member] = _signature(member)
                row['files'] = [member.relative_to(self.store.root).as_posix() for member in members]
                validate_group(names, members)
                if signatures != [file_signature(member) for member in members]:
                    raise ValueError('Weight group changed during header inspection.')
                if group in self.refs and expected != self.refs[group]['files']:
                    raise ValueError('Definition reference disagrees with the published group.')
                row['valid'] = True
                if self.report['hashes']:
                    self.jobs.append((row, members, signatures))
                if usage == 'orphan':
                    self.issue('unreferenced_group', path, 'No definition or reservation references this group.',
                               severity='warning')
            except (OSError, ValueError, UnicodeError) as exc:
                self.issue('group_invalid', path, exc)
        for group in sorted(set(self.refs) - seen):
            self.issue('group_missing', self.store.root / 'data' / group,
                       'Referenced by: ' + ', '.join(self.refs[group]['names']))

    def cache_target(self, relative):
        if not isinstance(relative, str) or '\\' in relative:
            raise ValueError('Cache model path is not repository-relative.')
        path = PurePosixPath(relative)
        parts = path.parts
        if path.as_posix() != relative or len(parts) != 3:
            raise ValueError('Cache model path is not canonical.')
        if parts[:2] == ('.tllama', 'uploads') and _GROUP.fullmatch(parts[2]):
            pass
        elif parts[0] == 'data' and _GROUP.fullmatch(parts[1]):
            match = _WEIGHT.fullmatch(parts[2])
            if not match or (match[1] is not None and not 1 <= int(match[1]) <= int(match[2]) <= 2048):
                raise ValueError('Invalid cached group filename.')
        else:
            raise ValueError('Cache model path is outside managed resources.')
        current = self.store.root
        for part in parts:
            current /= part
            if current.is_symlink():
                raise ValueError('Cache model path contains a symlink.')
        return current

    def caches(self):
        directory = self.store.root / '.tllama/metadata-cache'
        for path in self.children(directory, required=True) or []:
            _cancel(self.stop)
            row = {'path': path.relative_to(self.store.root).as_posix(), 'valid': False}
            self.report['caches'].append(row)
            try:
                match = _CACHE.fullmatch(path.name)
                if not match or not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError('Unexpected cache entry or symlink.')
                with path.open(encoding='utf-8') as handle:
                    raw = handle.read(16 * 1024 * 1024 + 1)
                    mtime = os.fstat(handle.fileno()).st_mtime_ns
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError('Cache document exceeds the inspection limit.')
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError('Invalid cache document.')
                digest = bool(match[2])
                schema = DIGEST_SCHEMA_VERSION if digest else SCHEMA_VERSION
                if (type(value.get('schema_version')) is not int or value['schema_version'] != schema
                        or type(value.get('cache_format_version')) is not int
                        or value.get('cache_format_version') != CACHE_FORMAT_VERSION):
                    raise ValueError('Obsolete or unknown cache schema.')
                model = value.get('model')
                if not isinstance(model, dict):
                    raise ValueError('Missing cache model identity.')
                target = self.cache_target(model.get('path'))
                if sha256(model['path'].encode()).hexdigest() != match[1]:
                    raise ValueError('Cache filename does not match model.path.')
                info = target.lstat()
                if (not stat.S_ISREG(info.st_mode) or type(model.get('size')) is not int
                        or model['size'] != info.st_size or mtime != info.st_mtime_ns):
                    raise ValueError('Stale cache fingerprint.')
                payload = value.get('digest' if digest else 'metadata')
                if not isinstance(payload, dict):
                    raise ValueError('Cache payload is not an object.')
                if digest and (payload.get('source') != 'local'
                               or not isinstance(payload.get('content_sha256'), str)
                               or not _GROUP.fullmatch(payload['content_sha256'])):
                    raise ValueError('Digest cache is not a measured local SHA-256.')
                row['valid'], row['model_path'] = True, model['path']
            except (OSError, ValueError, UnicodeError) as exc:
                self.issue('cache_invalid', path, exc, severity='warning')

    def staging(self):
        for area in ('uploads', 'downloads', 'staging'):
            for path in self.children(self.store.root / '.tllama' / area) or []:
                _cancel(self.stop)
                try:
                    mode = path.lstat().st_mode
                    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                        raise ValueError('Unexpected staging entry or symlink.')
                    relative = path.relative_to(self.store.root).as_posix()
                    self.report['staging'].append({'path': relative,
                        'kind': 'directory' if stat.S_ISDIR(mode) else 'file',
                        'reserved': self.pins.protects_path(relative) if self.pins is not None else None})
                except (OSError, ValueError) as exc:
                    self.issue('staging_invalid', path, exc)

    def hashes(self):
        for row, paths, signatures in self.jobs:
            try:
                digests = [_hash_file(path, signature, self.stop) for path, signature in zip(paths, signatures)]
                expected = [str(path) for path in group_paths(digests)]
                if expected != row['files']:
                    raise ValueError('Measured bytes do not match the group content address.')
                row['hash_verified'] = True
            except (OSError, ValueError) as exc:
                row['valid'] = False
                self.issue('group_hash_invalid', paths[0].parent, exc)

    def finish(self):
        report = self.report
        report['complete'] &= report['references_complete'] and report['reservations_complete']
        report['ok'] = report['complete'] and not any(issue['severity'] == 'error' for issue in report['issues'])
        report['summary'] = {key: len(report[key]) for key in ('definitions', 'groups', 'caches', 'staging')}
        report['summary'].update({severity + 's': sum(issue['severity'] == severity for issue in report['issues'])
                                  for severity in ('error', 'warning')})
        return report


def check_repository(store, *, hashes=False, stop=None):
    if type(hashes) is not bool:
        raise ValueError('hashes must be boolean.')
    inventory = _Inventory(store, hashes, stop)
    pin = None
    try:
        with store.lock(create=False):
            _cancel(stop)
            if read_format(store) is None:
                raise ValueError('Repository has no format-2 marker; run the offline converter.')
            inventory.definitions()
            inventory.protections()
            inventory.groups()
            inventory.caches()
            inventory.staging()
            if hashes and inventory.jobs and inventory.report['reservations_complete']:
                pin = Reservation(store, 'repo-check', all_data=True).acquire()
            elif hashes and inventory.jobs:
                raise ValueError('Cannot protect weight files for full hashing.')
        if hashes:
            inventory.hashes()
            with store.lock(create=False):
                for path, before in inventory.tracked.items():
                    _cancel(stop)
                    try:
                        if before != _signature(path):
                            raise ValueError('Repository changed during hashing; rerun the check.')
                    except (OSError, ValueError) as exc:
                        inventory.issue('repository_changed', path, exc)
                        inventory.report['complete'] = False
                        for row in inventory.report['groups']:
                            if path.parent == store.root / row['path']:
                                row['valid'], row['hash_verified'] = False, False
    except CheckCancelled:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        inventory.issue('repository_unavailable', store.root, exc)
        inventory.report['complete'] = False
    finally:
        if pin is not None:
            # A read-only check must not trigger 0175's deferred reclamation.
            pin.close(retry_cleanup=False)
    return inventory.finish()
