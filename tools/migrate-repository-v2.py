#!/usr/bin/env python3
"""Offline, resumable migration of a legacy TLlama model repository.

This tool is intentionally outside the installed Python package. A checkout
uses its adjacent src/ tree; the Debian documentation copy uses installed
TLlama modules. Never run a daemon while applying or resuming a migration.
"""

import argparse
import base64
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys

_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if _SOURCE.is_dir():
    sys.path.insert(0, str(_SOURCE))

from tllama.helpers.gguf_import import shard_names
from tllama.helpers.gguf_metadata import read_gguf_metadata
from tllama.helpers.model_toml import parse_model_toml
from tllama.helpers.metadata_cache import SCHEMA_VERSION, DIGEST_SCHEMA_VERSION, CACHE_FORMAT_VERSION
from tllama.helpers.repository_files import repository_tempfile
from tllama.model_identity import file_group_digest, model_digest, parameter_changes
from tllama.repository_layout import REPOSITORY_FORMAT
from tllama.repository_names import definition_path, check_portable_collisions
from tllama.repository_info import validate_original_name
from tllama.repository_store import (PreparedGroup, RepositoryStore, file_signature,
    hash_file, prepare_group, sync_directory)

CONTROL = '.tllama/migration-v2'
JOURNAL = CONTROL + '/journal.json'
MARKER = '.tllama/repository.json'
AREAS = {'Local': 1, 'TLlama': 2, 'HuggingFace': 3}
CACHE_CONTRACT = {'metadata': SCHEMA_VERSION, 'digest': DIGEST_SCHEMA_VERSION, 'format': CACHE_FORMAT_VERSION}


def safe_path(root, relative):
    path = Path(relative)
    if not isinstance(relative, str) or path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'Unsafe repository-relative path: {relative!r}')
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Symlinks must be resolved before migration: {relative}')
    if not current.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes the repository: {relative}')
    return current


def ensure_directory(root, relative, *, mode=0o777):
    path = safe_path(root, relative)
    current = root
    for part in path.relative_to(root).parts:
        parent = current
        current /= part
        try:
            current.mkdir(mode=mode)
        except FileExistsError:
            if not current.is_dir() or current.is_symlink():
                raise ValueError(f'Not a regular repository directory: {current}')
        else:
            sync_directory(parent)
    return path


def atomic_write(root, relative, raw, *, mode=0o600, times=None, owner=None):
    path = safe_path(root, relative)
    ensure_directory(root, path.parent.relative_to(root).as_posix())
    fd, temporary = repository_tempfile(directory=path.parent, prefix='.migration-',
                                       mode=0o600 if mode is not None else 0o666)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            if owner is not None and hasattr(os, 'fchown'):
                current = os.fstat(handle.fileno())
                if (current.st_uid, current.st_gid) != tuple(owner):
                    os.fchown(handle.fileno(), *owner)
            if mode is not None:
                os.chmod(temporary, mode)
            if times is not None:
                os.utime(temporary, ns=tuple(times))
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def read_state(root):
    marker = safe_path(root, MARKER)
    journal = safe_path(root, JOURNAL)
    format_value = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
    state = json.loads(journal.read_text(encoding='utf-8')) if journal.exists() else None
    if format_value is not None and format_value != REPOSITORY_FORMAT:
        raise ValueError('Unknown repository format; migration refused.')
    if state is not None and (not isinstance(state, dict) or state.get('journal_version') != 1
            or state.get('phase') not in {'publish', 'definitions', 'cleanup', 'complete'}):
        raise ValueError('Unknown migration journal; migration refused.')
    return format_value, state


def inventory(root):
    """Find owned definitions/weights; never traverse internal/control links."""
    weights, definitions, uploads = [], [], []
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        relative = parent.relative_to(root)
        for name in list(directories):
            child = parent / name
            if child.is_symlink():
                raise ValueError(f'Symlinked repository directory: {child.relative_to(root)}')
            if child == root / '.tllama':
                directories.remove(name)
        for name in files:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if relative.parts[:2] == ('Local', '.uploads'):
                safe_path(root, rel)
                uploads.append(rel)
                continue
            if name.lower().endswith('.gguf'):
                safe_path(root, rel)
                weights.append(rel)
            elif (name.lower().endswith('.toml') and relative.parts
                  and relative.parts[0].casefold() in {area.casefold() for area in AREAS}):
                if relative.parts[0] not in AREAS:
                    raise ValueError(f'Noncanonical repository branch casing: {relative.parts[0]}')
                safe_path(root, rel)
                parts = path.relative_to(root).parts
                if len(parts) - 1 != AREAS[parts[0]]:
                    raise ValueError(f'Definition at unsupported depth: {rel}')
                definitions.append(rel)
    return {'weights': sorted(weights), 'definitions': sorted(definitions), 'uploads': sorted(uploads)}


def resolve_reference(root, value):
    path = root / value
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'External or symlinked weight reference: {value}')
    rel = path.resolve().relative_to(root).as_posix()
    if rel.startswith('.tllama/') or rel.startswith('Local/.uploads/'):
        raise ValueError(f'Definition points into internal staging/control files: {value}')
    path = safe_path(root, rel)
    if not path.is_file():
        raise ValueError(f'Missing referenced weight file: {value}')
    return rel


def legacy_name(relative):
    parts = Path(relative).parts
    return '/'.join((*parts[1:-1], parts[-1][:-5]))


def ownership(path):
    value = path.stat()
    return [value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode)]


def check_ownership_permissions(owner, relative):
    if hasattr(os, 'geteuid') and os.geteuid() != 0:
        if owner[0] != os.geteuid() or owner[1] not in {*os.getgroups(), os.getegid()}:
            raise ValueError(f'Run as the original owner or administrator to preserve ownership: {relative}')


def group_record(group, root):
    for path in group.sources:
        validate_original_name(path.name)
    return {'sources': [path.relative_to(root).as_posix() for path in group.sources],
            'signatures': [list(value) for value in group.signatures],
            'digests': list(group.digests), 'metadata': list(group.metadata),
            'targets': [path.as_posix() for path in group.relative_paths],
            'ownership': [ownership(path) for path in group.sources]}


def build_plan(root, *, progress=None):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Repository root must be an existing directory.')
    format_value, state = read_state(root)
    if state is not None and state['phase'] != 'complete':
        raise ValueError('An incomplete migration exists; use --resume with all daemons stopped.')
    if format_value is not None:
        raise ValueError('Repository is already in format 2.')
    if state is not None:
        raise ValueError('Completed journal without its format marker; inspect before proceeding.')
    control = safe_path(root, CONTROL)
    if control.exists() and any(control.iterdir()):
        raise ValueError('Migration control directory exists without a journal; inspect it first.')
    files = inventory(root)
    store = RepositoryStore(root)
    # Internal paths must not redirect cache reads or later writes elsewhere.
    safe_path(root, '.tllama/metadata-cache')
    # Directory rename requires data and publication staging on the same FS.
    def device_for(relative):
        path = safe_path(root, relative)
        while not path.exists():
            path = path.parent
        return path.stat().st_dev
    if device_for('data') != device_for('.tllama/staging'):
        raise ValueError('data/ and .tllama/staging must be on the same filesystem for atomic publication.')
    groups, mapping, source_group, group_permissions = [], {}, {}, {}

    def add_group(relative):
        if relative in source_group:
            return source_group[relative]
        path = safe_path(root, relative)
        if progress is not None:
            progress(relative)
        names = shard_names(path.name, read_gguf_metadata(str(path)))
        paths = [safe_path(root, (path.parent / name).relative_to(root).as_posix()) for name in names]
        group = prepare_group(paths, names=names if len(names) > 1 else ['weights.gguf'], cache_dir=store.cache_dir)
        record = group_record(group, root)
        address = record['targets'][0]
        previous = group_permissions.setdefault(address, record['ownership'])
        if previous != record['ownership']:
            raise ValueError(f'Identical data groups have different owners/access bits; reconcile them first: {relative}')
        for path, owner in zip(paths, record['ownership']):
            if path.stat().st_dev != root.stat().st_dev:
                check_ownership_permissions(owner, path.relative_to(root).as_posix())
        index = len(groups)
        groups.append(record)
        for source, target in zip(record['sources'], record['targets']):
            if source in mapping and mapping[source] != target:
                raise ValueError(f'Weight file belongs to conflicting groups: {source}')
            mapping[source] = target
            source_group[source] = index
        targets = tuple(safe_path(root, value) for value in record['targets'])
        record['preexisting'] = targets[0].parent.exists()
        if targets[0].parent.exists():
            store._check_existing(group, targets)
            if [ownership(path) for path in targets] != record['ownership']:
                raise ValueError(f'Existing identical group has different owners/access bits: {relative}')
        return index

    definitions, pending = [], []
    for relative in files['definitions']:
        path = safe_path(root, relative)
        stats = path.stat()
        check_ownership_permissions([stats.st_uid, stats.st_gid], relative)
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError(f'Definition exceeds 2 MiB: {relative}')
        spec = parse_model_toml(raw.decode('utf-8'), source=relative)
        refs = {}
        for key, value in (('llm', spec.llm_model), ('mmproj', spec.mmproj_model),
                           ('lora', spec.runtime.get('lora_path'))):
            if value is not None:
                ref = resolve_reference(root, value)
                refs[key] = ref
        name = legacy_name(relative)
        target = definition_path(name).as_posix()
        pending.append((spec, refs))
        definitions.append({'source': relative, 'target': target, 'name': name,
                            'original': base64.b64encode(raw).decode('ascii'),
                            'original_sha256': sha256(raw).hexdigest(),
                            'mode': stat.S_IMODE(stats.st_mode),
                            'owner': [stats.st_uid, stats.st_gid],
                            'times': [stats.st_atime_ns, stats.st_mtime_ns]})
    target_definitions = [entry['target'] for entry in definitions]
    if len(target_definitions) != len(set(target_definitions)):
        raise ValueError('Multiple definitions have the same canonical target path.')
    check_portable_collisions(target_definitions)
    for value in target_definitions:
        target = safe_path(root, value)
        if target.exists() and value not in files['definitions']:
            raise ValueError(f'Target definition would replace an unmanaged file: {value}')
        for parent in target.parents:
            if parent == root:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError(f'Target directory is occupied by a file: {parent}')

    for entry, (_, refs) in zip(definitions, pending):
        for ref in refs.values():
            index = add_group(ref)
            if groups[index]['sources'][0] != ref:
                raise ValueError(f'Definition must reference the first shard: {entry["source"]}: {ref}')
    # Include orphan weights, without inventing any model definitions.
    for relative in files['weights']:
        if relative not in source_group:
            header = read_gguf_metadata(str(safe_path(root, relative)))
            if header.get('split.no', 0) == 0:
                add_group(relative)
    missing = set(files['weights']) - set(mapping)
    if missing:
        raise ValueError(f'Continuation shards have no complete first-shard group: {sorted(missing)}')

    target_groups = {entry['targets'][0]: index for index, entry in enumerate(groups)}
    all_groups = {**source_group, **target_groups}

    def identity(spec, refs):
        if 'llm' not in refs:
            return None  # Pending client-side from= is preserved, not imported.
        main = groups[all_groups[refs['llm']]]
        projector = groups[all_groups[refs['mmproj']]] if 'mmproj' in refs else None
        adapter = groups[all_groups[refs['lora']]] if 'lora' in refs else None
        changes = parameter_changes(spec, main['metadata'][0],
                                    adapter_digest=file_group_digest(adapter['digests']) if adapter else None)
        return model_digest(file_group_digest(main['digests']),
                            file_group_digest(projector['digests']) if projector else '', changes)

    for entry, (spec, refs) in zip(definitions, pending):
        entry['id'] = identity(spec, refs)
        for key in ('llm', 'mmproj'):
            if key in refs:
                spec.document[key]['model'] = mapping[refs[key]]
        if 'lora' in refs:
            spec.document['runtime']['lora_path'] = mapping[refs['lora']]
        text = spec.document.as_string()
        # Preserve line endings while retaining tomlkit comments/formatting.
        original = base64.b64decode(entry['original'])
        if b'\r\n' in original:
            text = text.replace('\r\n', '\n').replace('\n', '\r\n')
        rewritten = text.encode('utf-8')
        changed = parse_model_toml(text, source=entry['target'])
        changed_refs = {key: value for key, value in
                        (('llm', changed.llm_model), ('mmproj', changed.mmproj_model),
                         ('lora', changed.runtime.get('lora_path'))) if value is not None}
        if identity(changed, changed_refs) != entry['id']:
            raise ValueError(f'Migration changes model identity: {entry["name"]}')
        entry['rewritten'] = base64.b64encode(rewritten).decode('ascii')
        entry['rewritten_sha256'] = sha256(rewritten).hexdigest()
    cache_files = []
    if store.cache_dir.exists():
        for path in store.cache_dir.iterdir():
            if re.fullmatch('[0-9a-f]{64}(?:\\.digest)?\\.json', path.name):
                relative = path.relative_to(root).as_posix()
                if not safe_path(root, relative).is_file():
                    raise ValueError(f'Nonregular cache entry: {relative}')
                cache_files.append(relative)
    plan = {'inventory': files, 'groups': groups, 'mapping': mapping,
            'definitions': definitions, 'old_cache_files': sorted(cache_files),
            'upload_signatures': {value: list(file_signature(safe_path(root, value))) for value in files['uploads']}}
    json_bytes(plan)  # Fail on non-serializable/nonfinite data before writing.
    return plan


def report(plan, status):
    groups = {entry['targets'][0]: entry for entry in plan['groups']}
    return {'status': status, 'data_groups': len(groups), 'weight_files': len(plan['mapping']),
            'unique_weight_bytes': sum(sum(value[2] for value in group['signatures']) for group in groups.values()),
            'definitions': [{key: entry[key] for key in ('name', 'source', 'target', 'id')}
                            for entry in plan['definitions']],
            'files': [{'source': source, 'target': target} for source, target in sorted(plan['mapping'].items())]}


def _validate_journal(root, journal):
    """Check all recorded paths and text before resuming any mutation."""
    if journal.get('cache_contract') != CACHE_CONTRACT or journal.get('repository_format') != REPOSITORY_FORMAT:
        raise ValueError('Migration journal uses a different converter/core contract; resume with the matching version.')
    plan = journal['plan']
    if sorted(entry['source'] for entry in plan['definitions']) != plan['inventory']['definitions']:
        raise ValueError('Journal definition inventory is inconsistent.')
    for entry in plan['definitions']:
        safe_path(root, entry['source'])
        safe_path(root, entry['target'])
        source_parts = Path(entry['source']).parts
        if (not source_parts or source_parts[0] not in AREAS
                or len(source_parts) - 1 != AREAS[source_parts[0]]
                or not source_parts[-1].lower().endswith('.toml')
                or legacy_name(entry['source']) != entry['name']
                or definition_path(entry['name']).as_posix() != entry['target']):
            raise ValueError('Journal definition path does not match its logical name.')
        for key in ('original', 'rewritten'):
            if sha256(base64.b64decode(entry[key], validate=True)).hexdigest() != entry[key + '_sha256']:
                raise ValueError('Journal definition checksum mismatch.')
    check_portable_collisions([entry['target'] for entry in plan['definitions']])
    from tllama.repository_layout import group_paths
    expected = {}
    for entry in plan['groups']:
        if [path.as_posix() for path in group_paths(entry['digests'])] != entry['targets']:
            raise ValueError('Journal data paths do not match their content address.')
        if not (len(entry['sources']) == len(entry['targets']) == len(entry['signatures'])
                == len(entry['metadata']) == len(entry['ownership'])):
            raise ValueError('Journal group lengths disagree.')
        for source, target in zip(entry['sources'], entry['targets']):
            if source.startswith('.tllama/') or source.startswith('Local/.uploads/'):
                raise ValueError('Journal weight source points into control/staging files.')
            safe_path(root, source)
            safe_path(root, target)
            expected[source] = target
    if expected != plan['mapping']:
        raise ValueError('Journal source mapping disagrees with its groups.')
    for relative in plan['old_cache_files']:
        if not re.fullmatch(r'\.tllama/metadata-cache/[0-9a-f]{64}(?:\.digest)?\.json', relative):
            raise ValueError('Invalid cache cleanup path in journal.')
        safe_path(root, relative)
    for relative in plan['upload_signatures']:
        if not relative.startswith('Local/.uploads/'):
            raise ValueError('Invalid upload cleanup path in journal.')
        safe_path(root, relative)
    now = inventory(root)
    definition_sources = {entry['source'] for entry in plan['definitions']}
    definition_targets = {entry['target'] for entry in plan['definitions']}
    if journal['phase'] == 'publish':
        if set(now['definitions']) != definition_sources:
            raise ValueError('Definition inventory changed before publication; inspect before resuming.')
        for entry in plan['definitions']:
            if hash_file(safe_path(root, entry['source'])) != entry['original_sha256']:
                raise ValueError('Definition changed before publication; inspect before resuming.')
    if journal['phase'] == 'cleanup' and set(now['definitions']) != definition_targets:
        raise ValueError('Definition inventory changed before cleanup; inspect before resuming.')
    if (set(now['definitions']) - {entry[key] for entry in plan['definitions'] for key in ('source', 'target')}
            or set(now['weights']) - set(plan['mapping']) - set(plan['mapping'].values())
            or set(now['uploads']) - set(plan['upload_signatures'])):
        raise ValueError('Repository acquired new definitions/weights/uploads during migration; inspect before resuming.')
    return plan


def apply_plan(root, plan=None, *, resume=False):
    root = Path(root).resolve(strict=True)
    store = RepositoryStore(root)
    with store.lock():
        format_value, journal = read_state(root)
        if journal is not None and journal['phase'] == 'complete':
            if format_value is None:
                raise ValueError('Completed migration journal has no format marker; inspect before proceeding.')
            return report(journal['plan'], 'already-converted')
        if resume:
            if journal is None:
                raise ValueError('No migration journal exists to resume.')
            plan = _validate_journal(root, journal)
        else:
            if format_value is not None or journal is not None:
                raise ValueError('Migration state already exists; use --resume or inspect the repository.')
            if plan is None or inventory(root) != plan['inventory']:
                raise ValueError('Repository inventory changed after preflight; run again.')
            for entry in plan['definitions']:
                if hash_file(safe_path(root, entry['source'])) != entry['original_sha256']:
                    raise ValueError('Definition changed after preflight; run again.')
            for entry in plan['groups']:
                if [list(file_signature(safe_path(root, value))) for value in entry['sources']] != entry['signatures']:
                    raise ValueError('Weights changed after preflight; run again.')
            control = ensure_directory(root, CONTROL, mode=0o700)
            os.chmod(control, 0o700)
            sync_directory(root)
            journal = {'journal_version': 1, 'phase': 'publish', 'plan': plan,
                       'cache_contract': CACHE_CONTRACT, 'repository_format': REPOSITORY_FORMAT}
            _validate_journal(root, journal)
            atomic_write(root, JOURNAL, json_bytes(journal))

        def phase(value):
            journal['phase'] = value
            atomic_write(root, JOURNAL, json_bytes(journal))

        def verify_targets():
            for entry in plan['groups']:
                paths = [safe_path(root, value) for value in entry['targets']]
                group = prepare_group(paths, cache_dir=store.cache_dir)
                if list(group.digests) != entry['digests']:
                    raise ValueError('Published group no longer matches the migration journal.')
                store.record_names(paths, [Path(source).name for source in entry['sources']])

        if journal['phase'] == 'publish':
            for directory in ('data', '.tllama/staging', '.tllama/metadata-cache'):
                ensure_directory(root, directory)
            restored_owners = set()
            for entry in plan['groups']:
                paths = tuple(safe_path(root, value) for value in entry['sources'])
                group = PreparedGroup(paths, tuple(tuple(value) for value in entry['signatures']),
                                      tuple(entry['digests']), tuple(entry['metadata']),
                                      tuple(path.name for path in paths))
                # Different mounted filesystems inside a repository require a
                # verified copy; never fall back to an unverified move/unlink.
                link = all(path.stat().st_dev == root.stat().st_dev for path in paths)
                targets = store.publish(group, link_owned=link)
                address = entry['targets'][0]
                if not entry['preexisting'] and address not in restored_owners:
                    for path, owner in zip(targets, entry['ownership']):
                        for member in (path, path.with_suffix('.inf')):
                            if hasattr(os, 'chown') and ownership(member)[:2] != owner[:2]:
                                os.chown(member, *owner[:2])
                                os.chmod(member, owner[2])
                                with member.open('rb') as handle:
                                    os.fsync(handle.fileno())
                    restored_owners.add(address)
            phase('definitions')
        if journal['phase'] == 'definitions':
            verify_targets()
            allowed = {}
            for entry in plan['definitions']:
                allowed.setdefault(entry['source'], set()).add(entry['original_sha256'])
                allowed.setdefault(entry['target'], set()).add(entry['rewritten_sha256'])
            for relative, hashes in allowed.items():
                path = safe_path(root, relative)
                if path.exists() and (not path.is_file() or hash_file(path) not in hashes):
                    raise ValueError(f'Definition changed during migration: {relative}')
            # Back up every original before unlinking any path: encoded target
            # names can coincide with another definition's legacy literal name.
            for index, entry in enumerate(plan['definitions']):
                backup = f'{CONTROL}/backups/{index:06d}.toml'
                path = safe_path(root, backup)
                if path.exists():
                    if hash_file(path) != entry['original_sha256']:
                        raise ValueError(f'Original definition backup changed: {backup}')
                else:
                    atomic_write(root, backup, base64.b64decode(entry['original']))
            for entry in plan['definitions']:
                path = safe_path(root, entry['source'])
                path.unlink(missing_ok=True)
                sync_directory(path.parent)
            for entry in plan['definitions']:
                atomic_write(root, entry['target'], base64.b64decode(entry['rewritten']),
                             mode=entry['mode'], times=entry['times'], owner=entry['owner'])
            phase('cleanup')
        if journal['phase'] == 'cleanup':
            verify_targets()
            for entry in plan['definitions']:
                if hash_file(safe_path(root, entry['target'])) != entry['rewritten_sha256']:
                    raise ValueError('Converted definition changed before cleanup.')
            signatures = {source: signature for entry in plan['groups']
                          for source, signature in zip(entry['sources'], entry['signatures'])}
            for source, target in plan['mapping'].items():
                if source == target:
                    continue
                path = safe_path(root, source)
                if path.exists():
                    if list(file_signature(path)) != signatures[source]:
                        raise ValueError(f'Old weight changed before removal: {source}')
                    path.unlink()
                    sync_directory(path.parent)
            keep = {f'.tllama/metadata-cache/{sha256(target.encode()).hexdigest()}{suffix}'
                    for target in plan['mapping'].values() for suffix in ('.json', '.digest.json')}
            for relative in plan['old_cache_files']:
                if relative not in keep:
                    safe_path(root, relative).unlink(missing_ok=True)
            if store.cache_dir.exists():
                sync_directory(store.cache_dir)
            for relative, signature in plan['upload_signatures'].items():
                path = safe_path(root, relative)
                if path.exists():
                    if list(file_signature(path)) != signature:
                        raise ValueError(f'Upload staging changed during migration: {relative}')
                    path.unlink()
                    sync_directory(path.parent)
            # Remove only empty original directories; preserve unrelated files.
            parents = set()
            for relative in (*plan['mapping'], *plan['inventory']['definitions'], *plan['upload_signatures']):
                parent = Path(relative).parent
                while parent.as_posix() != '.' and parent.as_posix() not in AREAS:
                    parents.add(parent.as_posix())
                    parent = parent.parent
            for relative in sorted(parents, key=lambda value: (-len(Path(value).parts), value)):
                try:
                    path = safe_path(root, relative)
                    path.rmdir()
                    sync_directory(path.parent)
                except OSError as exc:
                    if exc.errno not in {errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST}:
                        raise
            # The marker commits a complete current-format store, including
            # empty definition branches and reservation state. Local read-only
            # maintenance must work before the first daemon initialization.
            for relative in (*AREAS, '.tllama/pins'):
                ensure_directory(root, relative)
            atomic_write(root, MARKER, json_bytes(REPOSITORY_FORMAT), mode=None)
            phase('complete')
        return report(plan, 'converted')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path, help='Existing model repository root.')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--apply', action='store_true', help='Convert with all daemons stopped; default is read-only.')
    action.add_argument('--resume', action='store_true', help='Resume an interrupted conversion with all daemons stopped.')
    parser.add_argument('--json', action='store_true', help='Print the summary/mapping as JSON; never print prompts.')
    args = parser.parse_args(argv)
    try:
        root = args.repository.resolve(strict=True)
        if args.resume:
            result = apply_plan(root, resume=True)
        else:
            marker, state = read_state(root)
            if marker is not None and (state is None or state['phase'] == 'complete'):
                result = {'status': 'already-converted'}
            else:
                progress = None if args.json else lambda path: print(f'Inspecting: {path}', file=sys.stderr, flush=True)
                plan = build_plan(root, progress=progress)
                result = apply_plan(root, plan) if args.apply else report(plan, 'planned')
        if args.json:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            print(f"Migration: {result['status']}")
            for entry in result.get('definitions', []):
                print(f"Definition: {entry['source']} -> {entry['target']}")
            for entry in result.get('files', []):
                print(f"Weights: {entry['source']} -> {entry['target']}")
            if result['status'] == 'planned':
                print('Nothing written. Stop all daemons before using --apply.')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'migrate-repository-v2: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
