"""Current repository format, definition transactions and HF publication."""

import json
import os
from pathlib import Path
import re
from functools import wraps
from hashlib import sha256

from tllama.config import ConfigError
from tllama.helpers.gguf_import import shard_names
from tllama.helpers.gguf_metadata import read_gguf_metadata
from tllama.helpers.model_toml import parse_model_toml, render_model_toml, write_model_toml
from tllama.helpers.repository_files import repository_tempfile
from tllama.repository_layout import REPOSITORY_FORMAT
from tllama.repository_names import definition_path, check_portable_collisions
from tllama.repository_store import prepare_group, sync_directory
from tllama.repository_pins import temporary_directory


MARKER = '.tllama/repository.json'


def read_format(store):
    marker = store.root / MARKER
    if marker.is_symlink():
        raise ConfigError('Repository format marker must not be a symlink.')
    if not marker.exists():
        return None
    try:
        value = json.loads(marker.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ConfigError(f'Cannot read repository format marker: {exc}') from exc
    if value != REPOSITORY_FORMAT:
        raise ConfigError('Unknown repository format; use the matching TLlama version.')
    return value


def write_json(store, relative, value):
    parent = Path(relative).parent.as_posix()
    directory = store._directory(parent)
    path = store.root / relative
    if path.is_symlink():
        raise ValueError(f'Repository metadata must not be a symlink: {relative}')
    fd, temporary = repository_tempfile(directory=directory, prefix='.repository-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_format(store):
    """Initialize only an empty store. Never reinterpret legacy file names.

    A valid final marker is the offline converter's commit point: it is written
    only after all data/definition changes and cleanup. The private recovery
    journal need not be readable by the service account after an admin conversion.
    """
    with store.lock():
        if read_format(store) is None:
            if (store.root / '.tllama/migration-v2').exists():
                raise ConfigError('Repository conversion is incomplete. Keep daemons stopped and use '
                                  'tools/migrate-repository-v2.py REPOSITORY --resume.')
            for directory, directories, files in os.walk(store.root, followlinks=False):
                parent = Path(directory)
                for name in list(directories):
                    if (parent / name).is_symlink():
                        raise ConfigError('Unmarked repository has symlinked directories; inspect and convert it explicitly.')
                    if parent / name == store.root / '.tllama':
                        directories.remove(name)
                if files:
                    raise ConfigError('Legacy/nonempty repository has no format-2 marker. Stop daemons and run '
                                      'tools/migrate-repository-v2.py REPOSITORY, then --apply.')
            write_json(store, MARKER, REPOSITORY_FORMAT)
            sync_directory(store.root)
        for directory in ('Local', 'TLlama', 'HuggingFace', 'data', '.tllama/metadata-cache', '.tllama/pins'):
            store._directory(directory)


def repository_operation(function):
    @wraps(function)
    def operation(manager, *args, **kwargs):
        with manager.repository.lock():
            ensure_format(manager.repository)
            return function(manager, *args, **kwargs)
    return operation


def destination(manager, name):
    relative = definition_path(name)
    target = manager.models_dir / relative
    current = manager.models_dir.resolve()
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Definition path must not contain a symlink.')
    if not target.resolve().is_relative_to(manager.models_dir.resolve()):
        raise ValueError('Definition must stay inside the repository.')
    paths = [path.relative_to(manager.models_dir).as_posix() for path in manager._iter_repository_toml_files()]
    check_portable_collisions([*paths, relative.as_posix()])
    return target


def require_data_reference(manager, value):
    """Public mutations only publish references to canonical managed data."""
    path = Path(value)
    if path.is_absolute() or '\\' in value or path.as_posix() != value:
        raise ValueError('Managed data references must be repository-relative POSIX paths.')
    parts = path.parts
    if (len(parts) != 3 or parts[0] != 'data' or not re.fullmatch('[0-9a-f]{64}', parts[1])
            or not re.fullmatch(r'weights(?:-00001-of-\d{5})?\.gguf', parts[2])):
        raise ValueError('Model files must reference the first file of a managed data/<sha256>/ group.')
    result = manager.models_dir / path
    current = manager.repository.root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Managed data paths must not contain symlinks.')
    if not result.resolve().is_relative_to(manager.repository.root) or not result.is_file():
        raise FileNotFoundError(f'Managed data file does not exist: {value}')
    names = shard_names(result.name, read_gguf_metadata(str(result)))
    group = prepare_group([result.parent / name for name in names], cache_dir=manager.metadata_cache_dir)
    if (group.relative_paths[0].as_posix() != value
            or {item.name for item in result.parent.iterdir()} != set(names)):
        raise ValueError('Managed data group does not match its content address or complete file set.')
    return result


def origin_path(repo_id, filename):
    key = json.dumps([repo_id, filename], ensure_ascii=False, separators=(',', ':')).encode()
    return '.tllama/origins/' + sha256(key).hexdigest() + '.json'


def read_origin(manager, repo_id, filename):
    path = manager.repository.root / origin_path(repo_id, filename)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError('Pull origin record must not be a symlink.')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) and value.get('repo_id') == repo_id and value.get('filename') == filename else {}
    except (OSError, ValueError):
        return {}


def register_pull(manager, repo_id, filename, paths, metadata, digests, source_files):
    """Called with the repository lock, after whole-group publication."""
    relative = paths[0].relative_to(manager.repository.root).as_posix()
    index = read_origin(manager, repo_id, filename)
    name = None
    if not metadata.get('is_projector'):
        base = repo_id + '/' + Path(filename).name[:-5]
        candidates = [base] + [base + f'_{number:02d}' for number in range(1, 100)]
        previous = index.get('model')
        if isinstance(previous, str) and previous in candidates:
            candidates.insert(0, previous)
        else:
            previous = None
        for candidate in dict.fromkeys(candidates):
            target = destination(manager, candidate)
            if not target.exists():
                write_model_toml(target, render_model_toml(relative, metadata=metadata))
                name = candidate
                break
            spec = parse_model_toml(target.read_text(encoding='utf-8'), source=str(target))
            if spec.llm_model == relative:
                name = candidate
                break
            if candidate == previous and spec.llm_model == index.get('path'):
                spec.document['llm']['model'] = relative
                write_model_toml(target, spec.document.as_string(), overwrite=True)
                name = candidate
                break
        if name is None:
            raise ValueError('Pull definition name and its numbered variants are already taken.')
        manager._invalidate_metadata_cache_entry(name)
    write_json(manager.repository, origin_path(repo_id, filename),
               {'repo_id': repo_id, 'filename': filename, 'model': name, 'path': relative,
                'digests': list(digests), 'files': source_files})
    return str(paths[0])


def pull_sync(manager, repo_id, filename, token=None, revision=None, progress=None, file_info=None):
    """Stage downloads privately; publish/register under one repository lock."""
    ensure_format(manager.repository)
    from huggingface_hub import hf_hub_download
    from tllama.backend import _hf_tqdm_class_for
    from tllama.repository_names import encode_component
    parts = filename.split('/')
    if (not filename.lower().endswith('.gguf') or len(repo_id.split('/')) != 2 or any(part in {'', '.', '..'} or '\\' in part
            for part in [*repo_id.split('/'), *parts])):
        raise ValueError('Invalid Hugging Face repository/file reference.')
    for part in [*repo_id.split('/'), *parts]:
        encode_component(part)
    with manager.repository.lock():
        destination(manager, repo_id + '/' + parts[-1][:-5])
    index = read_origin(manager, repo_id, filename)
    # Reuse only after checking every remote shard, not just the first one.
    if (revision is None and isinstance(index.get('files'), list) and index['files']
            and isinstance(index.get('digests'), list) and len(index['files']) == len(index['digests'])
            and all(isinstance(value, str) for value in index['files'] + index['digests'])):
        reusable = True
        for number, (remote, digest) in enumerate(zip(index['files'], index['digests'])):
            info = file_info if number == 0 and file_info is not None else manager._fetch_hf_file_info_sync(repo_id, remote, token)
            if info.status != 'found' or info.sha256 != digest:
                reusable = False
                break
        if reusable:
            with manager.repository.lock():
                group = None
                try:
                    first = require_data_reference(manager, index['path'])
                    names = shard_names(first.name, read_gguf_metadata(str(first)))
                    group = prepare_group([first.parent / name for name in names], cache_dir=manager.metadata_cache_dir)
                except (OSError, KeyError, ValueError, TypeError):
                    pass
                if group is not None and list(group.digests) == index['digests']:
                    if progress is not None:
                        size = sum(path.stat().st_size for path in group.sources)
                        progress.set_total(size)
                        progress.add(size)
                    return register_pull(manager, repo_id, filename, group.sources, group.metadata[0],
                                         group.digests, index['files'])
    # completed already accumulates bytes across bars; total must likewise
    # cover the whole split group instead of being reset by each shard.
    totals = {filename: progress.snapshot()[1]} if progress is not None else {}
    class FileProgress:
        def __init__(self, remote):
            self.remote = remote

        def set_total(self, total):
            if total:
                totals[self.remote] = int(total)
                progress.set_total(sum(totals.values()))

        def add(self, count):
            progress.add(count)

    with temporary_directory(manager.repository, parent='.tllama/downloads', prefix='pull-',
                             purpose='pull') as (directory, pin):
        def download(remote):
            result = Path(hf_hub_download(repo_id=repo_id, filename=remote, revision=revision, token=token,
                                         local_dir=directory,
                                         tqdm_class=_hf_tqdm_class_for(FileProgress(remote)) if progress is not None else None))
            if result.resolve() != (directory / remote).resolve() or not result.resolve().is_relative_to(directory.resolve()):
                raise ValueError('HF downloader returned a path outside its owned staging directory.')
            return result
        first = download(filename)
        names = shard_names(first.name, read_gguf_metadata(str(first)))
        remote_parent = Path(filename).parent
        files = [(remote_parent / name).as_posix() for name in names]
        sources = [first] + [download(remote) for remote in files[1:]]
        group = prepare_group(sources)
        with manager.repository.lock():
            pin.update(groups=[group.relative_paths[0].parts[1]])
            paths = manager.repository.publish(group, link_owned=True)
            return register_pull(manager, repo_id, filename, paths, group.metadata[0], group.digests, files)
