"""Listing sizes describe all files required by one model definition."""

import os

import gguf
import pytest

from tllama.helpers.model_toml import parse_model_toml


@pytest.fixture
def artifact(manager):
    def create(name, arch='llama', **keys):
        path = manager.local_models_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(str(path), arch)
        for key, value in keys.items():
            if isinstance(value, str):
                writer.add_string(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            else:
                writer.add_uint32(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        return path
    return create


def define(toml_file, name='model', llm='model.gguf', projector=None, extra=''):
    body = f'[llm]\nmodel="Local/{llm}"\n'
    if projector:
        body += f'[mmproj]\nmodel="Local/{projector}"\n'
    return toml_file(f'Local/{name}.toml', body + extra)


async def test_size_counts_projector_and_shared_llm_per_definition(manager, artifact, toml_file):
    llm = artifact('model.gguf')
    proj = artifact('projector.gguf', 'clip')
    define(toml_file, 'text')
    define(toml_file, 'vision', projector=proj.name)
    models = {item['id']: item for item in await manager.list_local_models_with_metadata()}
    assert models['text']['size'] == llm.stat().st_size
    assert models['vision']['size'] == llm.stat().st_size + proj.stat().st_size


async def test_size_includes_all_main_and_projector_shards(manager, artifact, toml_file):
    files = []
    for prefix, arch in [('model', 'llama'), ('projector', 'clip')]:
        for index in range(2):
            files.append(artifact(f'{prefix}-{index+1:05d}-of-00002.gguf', arch,
                                  **{'split.count': 2, 'split.no': index}))
    define(toml_file, llm=files[0].name, projector=files[2].name)
    [model] = await manager.list_local_models_with_metadata()
    assert model['size'] == sum(path.stat().st_size for path in files)
    # Header cache reuse must not freeze the sizes of continuation shards.
    with files[1].open('ab') as handle:
        handle.write(b'more tensor bytes')
    [changed] = await manager.list_local_models_with_metadata()
    assert changed['size'] == sum(path.stat().st_size for path in files)


@pytest.mark.parametrize('broken', ['missing', 'wrong-index', 'outside'])
async def test_bad_shard_set_is_isolated_to_its_definition(manager, artifact, toml_file, tmp_path, broken):
    first = artifact('split-00001-of-00002.gguf', **{'split.count':2, 'split.no':0})
    if broken == 'wrong-index':
        artifact('split-00002-of-00002.gguf', **{'split.count':2, 'split.no':0})
    elif broken == 'outside':
        outside = tmp_path / 'outside.gguf'
        outside.write_bytes(b'outside')
        (first.parent/'split-00002-of-00002.gguf').symlink_to(outside)
    artifact('model.gguf')
    define(toml_file, 'broken', llm=first.name)
    define(toml_file, 'healthy')
    assert [item['id'] for item in await manager.list_local_models_with_metadata()] == ['healthy']


def test_required_adapter_counts_once_even_when_hard_linked(manager, artifact, toml_file):
    path = artifact('model.gguf')
    link = path.with_name('adapter.gguf')
    os.link(path, link)
    toml = define(toml_file, extra='[runtime]\nlora_path="Local/adapter.gguf"\n')
    spec = parse_model_toml(toml.read_text())
    assert manager._model_artifacts(spec, path, toml).size == path.stat().st_size


def test_tags_exposes_complete_size(client, manager, artifact, toml_file, monkeypatch):
    from tllama.routers import ollama
    monkeypatch.setattr(ollama, 'model_manager', manager)
    llm, proj = artifact('model.gguf'), artifact('proj.gguf', 'clip')
    define(toml_file, projector=proj.name)
    [model] = client.get('/api/tags').json()['models']
    assert model['size'] == llm.stat().st_size + proj.stat().st_size
