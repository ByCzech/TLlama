"""Publish model definitions while keeping shared weights and comments intact."""

from tllama.helpers.model_toml import parse_model_toml, resolve_repo_relative_path, write_model_toml, validate_definition_name
from tllama.helpers.gguf_import import validate_group
from tllama.model_uploads import verified_blob
from tllama.helpers.metadata_cache import build_model_file_fingerprint, save_metadata_cache, save_digest_cache
from tllama.helpers.gguf_metadata import build_model_metadata_payload
import os


def definition_path(manager, name):
    validate_definition_name(name)
    path = manager._toml_path_for_reference(name)
    if path is None:
        raise ValueError('Model names support one to three path components.')
    if path.is_symlink() or not path.resolve().is_relative_to(manager.models_dir.resolve()):
        raise ValueError('Definition must stay inside the model repository without a symlink target.')
    return path


def validate_definition(manager, text):
    spec = parse_model_toml(text)
    for section, model, source in (('llm', spec.llm_model, spec.llm_from),
                                   ('mmproj', spec.mmproj_model, spec.mmproj_from)):
        if source is not None:
            raise ValueError(f'[{section}].from must be uploaded with create before publication.')
        if model is not None:
            path = resolve_repo_relative_path(model, manager.models_dir)
            if not path.is_file():
                raise FileNotFoundError(f'[{section}].model does not name an existing repository file: {model}')
    return spec


def copy_definition(manager, source, destination, *, force=False):
    source_path = definition_path(manager, source)
    target = definition_path(manager, destination)
    if source_path.resolve() == target.resolve():
        raise ValueError('Source and destination name the same definition.')
    with source_path.open('r', encoding='utf-8', newline='') as source_file:
        text = source_file.read()
    validate_definition(manager, text)
    write_model_toml(target, text, overwrite=force)
    manager._invalidate_metadata_cache_entry(destination)
    return {'status': 'success', 'source': source, 'destination': destination}


def create_definition(manager, name, text, uploads, *, force=False, dry_run=False):
    if len(text.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValueError('TOML definition exceeds the 2 MiB limit.')
    target = definition_path(manager, name)
    if target.exists() and not force:
        raise FileExistsError('Destination already exists.')
    spec = parse_model_toml(text)
    for section, model, source in (('llm', spec.llm_model, spec.llm_from),
                                   ('mmproj', spec.mmproj_model, spec.mmproj_from)):
        if source is None and model is not None and not resolve_repo_relative_path(model, manager.models_dir).is_file():
            raise FileNotFoundError(f'[{section}].model does not name an existing repository file: {model}')
    if dry_run:
        if uploads:
            raise ValueError('Preflight checks definition/name/existing references; do not supply uploads.')
        return {'status': 'ready', 'model': name}
    expected = {section for section, source in (('llm', spec.llm_from), ('mmproj', spec.mmproj_from))
                if source is not None}
    if set(uploads) != expected:
        raise ValueError('Uploads must match exactly the [llm]/[mmproj] sections containing from.')
    groups = []
    for section, files in uploads.items():
        if not 1 <= len(files) <= 2048:
            raise ValueError('An upload group requires 1..2048 files.')
        names = [item.filename for item in files]
        if any('/' in item or '\\' in item or any(ord(c) < 32 for c in item)
               or item in {'.', '..'} or not item.lower().endswith('.gguf') for item in names):
            raise ValueError('Upload filenames must be plain GGUF basenames.')
        verified = [verified_blob(manager, item.digest) for item in files]
        blobs = [path for path, fingerprint in verified]
        fingerprints = [fingerprint for path, fingerprint in verified]
        metadata = validate_group(names, blobs)
        if fingerprints != [build_model_file_fingerprint(path) for path in blobs]:
            raise ValueError('Uploaded files changed during header inspection.')
        directory = manager.local_models_dir / 'imports' / files[0].digest[7:]
        if not directory.resolve().is_relative_to(manager.models_dir.resolve()):
            raise ValueError('Local import directory escapes the model repository.')
        groups.append((section, names, blobs, directory, metadata, files, fingerprints))
        # Only registered blobs are read. from is a client-side path and
        # must never grant an HTTP caller server filesystem access.
        table = spec.document[section]
        table['from'] = str((directory / names[0]).relative_to(manager.models_dir))
        table['model'] = table.pop('from')
    if groups:
        text = spec.document.as_string()
    for section, names, blobs, directory, metadata, files, fingerprints in groups:
        directory.mkdir(parents=True, exist_ok=True)
        for filename, blob, header, uploaded, scanned in zip(names, blobs, metadata, files, fingerprints):
            destination = directory / filename
            try:
                os.link(blob, destination)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file() or not os.path.samefile(blob, destination):
                    raise FileExistsError('An imported weight filename already exists with different content.')
            fingerprint = {**scanned, 'path': str(destination.resolve())}
            save_digest_cache(manager.metadata_cache_dir, name, destination,
                              {'content_sha256': uploaded.digest[7:], 'source': 'local'}, fingerprint=fingerprint)
            save_metadata_cache(manager.metadata_cache_dir, name, destination,
                                build_model_metadata_payload(header), fingerprint=fingerprint)
    validate_definition(manager, text)
    write_model_toml(target, text, overwrite=force)
    manager._invalidate_metadata_cache_entry(name)
    return {'status': 'success', 'model': name}
