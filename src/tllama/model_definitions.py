"""Publish model definitions while keeping shared weights and comments intact."""

from tllama.helpers.model_toml import parse_model_toml, write_model_toml
from tllama.repository_store import prepare_group
from tllama.repository_runtime import destination, require_data_reference, repository_operation
from tllama.model_uploads import verified_blob
from tllama.helpers.metadata_cache import build_model_file_fingerprint


def definition_path(manager, name):
    return destination(manager, name)


def validate_definition(manager, text):
    spec = parse_model_toml(text)
    for section, model, source in (('llm', spec.llm_model, spec.llm_from),
                                   ('mmproj', spec.mmproj_model, spec.mmproj_from)):
        if source is not None:
            raise ValueError(f'[{section}].from must be uploaded with create before publication.')
        if model is not None:
            path = require_data_reference(manager, model)
            if not path.is_file():
                raise FileNotFoundError(f'[{section}].model does not name an existing repository file: {model}')
    if 'lora_path' in spec.runtime:
        require_data_reference(manager, spec.runtime['lora_path'])
    return spec


@repository_operation
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


@repository_operation
def create_definition(manager, name, text, uploads, *, force=False, dry_run=False):
    if len(text.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValueError('TOML definition exceeds the 2 MiB limit.')
    target = definition_path(manager, name)
    if target.exists() and not force:
        raise FileExistsError('Destination already exists.')
    spec = parse_model_toml(text)
    for section, model, source in (('llm', spec.llm_model, spec.llm_from),
                                   ('mmproj', spec.mmproj_model, spec.mmproj_from)):
        if source is None and model is not None:
            require_data_reference(manager, model)
    if 'lora_path' in spec.runtime:
        require_data_reference(manager, spec.runtime['lora_path'])
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
        group = prepare_group(blobs, names=names, cache_dir=manager.metadata_cache_dir)
        if tuple(item.digest[7:] for item in files) != group.digests:
            raise ValueError('Uploaded content does not match its declared digests.')
        if fingerprints != [build_model_file_fingerprint(path) for path in blobs]:
            raise ValueError('Uploaded files changed during header inspection.')
        groups.append(group)
        # Only registered blobs are read. from is a client-side path and
        # must never grant an HTTP caller server filesystem access.
        table = spec.document[section]
        table['from'] = group.relative_paths[0].as_posix()
        table['model'] = table.pop('from')
    if groups:
        text = spec.document.as_string()
    for group in groups:
        manager.repository.publish(group, link_owned=True)
    validate_definition(manager, text)
    write_model_toml(target, text, overwrite=force)
    manager._invalidate_metadata_cache_entry(name)
    return {'status': 'success', 'model': name}
