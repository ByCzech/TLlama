"""Publish model definitions while keeping shared weights and comments intact."""

from tllama.helpers.model_toml import parse_model_toml, resolve_repo_relative_path, write_model_toml


def definition_path(manager, name):
    if (not name or name != name.strip() or '\\' in name or '\0' in name
            or any(part in {'', '.', '..'} for part in name.split('/'))):
        raise ValueError('Invalid model name.')
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
    text = source_path.read_text(encoding='utf-8')
    validate_definition(manager, text)
    write_model_toml(target, text, overwrite=force)
    manager._invalidate_metadata_cache_entry(destination)
    return {'status': 'success', 'source': source, 'destination': destination}
