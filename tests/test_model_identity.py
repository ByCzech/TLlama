"""Portable IDs preserve plain SHA-256 and hash only actual model changes."""

from hashlib import sha256
import os

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.metadata_cache import get_digest_cache_path, load_digest_cache, DIGEST_SCHEMA_VERSION
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_identity import parameter_changes, model_digest
from test_model_artifacts import artifact, define


def spec(extra=''):
    return parse_model_toml('[llm]\nmodel="Local/a.gguf"\n' + extra)


@pytest.mark.parametrize('extra', [
    '', '[sampling]\ntemperature=0.8\ntop_k=40\nmax_tokens=-1\nstop=[]\n',
    '[sampling]\nseed=-1\n', '[sampling]\nseed=4294967295\n[runtime]\nseed=4294967295\n',
    '[runtime]\nn_batch=512\nn_ubatch=512\nuse_mmap=true\nuse_mlock=false\n',
    '[runtime]\ntype_kv=1\nflash_attn=false\ncontext_shift=false\n',
    '[runtime]\nn_ctx=0\nnuma=false\nseed=-1\noffload_kqv=true\n',
    '[runtime]\nn_threads=0\nn_threads_batch=-1\nspec_type="none"\ndraft_n_max=3\n',
    '[system]\nprompt=""\n[template]\njinja=""\n[fim]\norder="auto"\n',
])
def test_default_spellings_preserve_plain_sha256(extra):
    changes = parameter_changes(spec(extra), {})
    digest = sha256(b'weights').hexdigest()
    assert changes == {}
    assert model_digest(digest, changes=changes) == digest


def test_four_composition_cases_are_distinct_and_roles_are_framed():
    llm, projector = sha256(b'llm').hexdigest(), sha256(b'projector').hexdigest()
    changed = parameter_changes(spec('[sampling]\ntop_k=10\n'), {})
    ids = [model_digest(llm), model_digest(llm, projector),
           model_digest(llm, changes=changed), model_digest(llm, projector, changed)]
    assert ids[0] == llm and len(set(ids)) == 4
    assert model_digest(llm, projector) != model_digest(projector, llm)


def test_gguf_defaults_precede_toml_comparison_including_float32():
    metadata = {'arch':'llama', 'context_length':8192, 'recommended_sampling':{'temperature':0.6000000238418579},
                'template':'GGUF template', 'tool_template':'GGUF template',
                'embedding':{'pooling_type':3},
                'metadata_raw':{'llama.rope.freq_base':10000}}
    equal = spec('[sampling]\ntemperature=0.6\n[runtime]\nn_ctx=8192\npooling_type=3\n'
                 'rope_freq_base=10000.0\n[template]\njinja="GGUF template"\n')
    assert parameter_changes(equal, metadata) == {}
    assert parameter_changes(spec('[sampling]\ntemperature=0.8\n'), metadata) != {}
    # An explicit native template overrides the tool-use variant as well.
    assert parameter_changes(equal, dict(metadata, tool_template='other tool template')) == {
        'template.jinja':'GGUF template'}


def test_normalization_sorts_keys_and_preserves_ordered_values():
    first = spec('[sampling]\ntop_p=1\ntop_k=20\nstop=["a","b"]\n')
    second = spec('# comment\n[sampling]\nstop = ["a", "b"]\ntop_k = 20\ntop_p = 1.0\n')
    assert model_digest('llm', changes=parameter_changes(first, {})) == model_digest(
        'llm', changes=parameter_changes(second, {}))
    second.stop.reverse()
    assert model_digest('llm', changes=parameter_changes(first, {})) != model_digest(
        'llm', changes=parameter_changes(second, {}))


def test_parameter_names_and_kv_types_are_part_of_the_identity():
    first = parameter_changes(spec('[runtime]\nkv_overrides={test=1}\n'), {})
    second = parameter_changes(spec('[runtime]\nkv_overrides={test=1.0}\n'), {})
    assert model_digest('llm', changes=first) != model_digest('llm', changes=second)
    assert model_digest('llm', changes={'sampling.top_k':10}) != model_digest(
        'llm', changes={'runtime.n_ctx':10})


def test_kv_aliases_and_implicit_mtp_limit_have_one_identity():
    shorthand = spec('[runtime]\ntype_kv="Q8_0"\nspec_type="draft-mtp"\n')
    explicit = spec('[runtime]\ntype_k=8\ntype_v="q8_0"\nspec_type="draft-mtp"\ndraft_n_max=3\n')
    assert parameter_changes(shorthand, {}) == parameter_changes(explicit, {})
    # Explicit all-layers differs from the default automatic fit policy.
    assert 'runtime.n_gpu_layers' in parameter_changes(spec('[runtime]\nn_gpu_layers=-1\n'), {})


@pytest.mark.parametrize('extra', ['[system]\nprompt="custom"\n', '[template]\njinja="custom"\n',
                                  '[fim]\norder="disabled"\n'])
def test_prompt_template_and_fim_changes_are_included(extra):
    assert parameter_changes(spec(extra), {})


async def test_real_listing_aliases_projector_parameters_and_cache_reuse(manager, artifact, toml_file, monkeypatch):
    llm, projector = artifact('model.gguf'), artifact('proj.gguf', 'clip')
    define(toml_file, 'plain')
    define(toml_file, 'copy')
    define(toml_file, 'vision', projector=projector.name)
    definition = define(toml_file, 'changed', extra='[sampling]\ntop_k=10\n')
    define(toml_file, 'both', projector=projector.name, extra='[sampling]\ntop_k=10\n')
    models = {item['id']:item for item in await manager.list_local_models_with_metadata()}
    assert models['plain']['digest'] == models['copy']['digest'] == sha256(llm.read_bytes()).hexdigest()
    assert len({item['digest'] for item in models.values()}) == 4
    assert models['vision']['digest'] == model_digest(sha256(llm.read_bytes()).hexdigest(),
                                                      sha256(projector.read_bytes()).hexdigest())
    definition.write_text(definition.read_text() + '# formatting only\n')
    def fail(*args):
        pytest.fail('An unchanged GGUF was rehashed')
    monkeypatch.setattr(manager, '_compute_content_sha256', fail)
    again = {item['id']:item['digest'] for item in await manager.list_local_models_with_metadata()}
    assert again == {name:item['digest'] for name, item in models.items()}


async def test_server_environment_and_paths_cannot_change_id(manager, artifact, toml_file, monkeypatch, tmp_path):
    llm = artifact('model.gguf', **{'general.sampling.top_k':12})
    define(toml_file, extra='[sampling]\ntop_k=12\n[runtime]\nn_ctx=0\n')
    [original] = await manager.list_local_models_with_metadata()
    root = tmp_path/'different-server'
    (root/'Local').mkdir(parents=True)
    (root/'Local'/'renamed.gguf').write_bytes(llm.read_bytes())
    (root/'Local'/'alias.toml').write_text('[llm]\nmodel="Local/renamed.gguf"\n'
                                         '[runtime]\nn_ctx=0\n[sampling]\ntop_k=12\n')
    other = ModelManager(BackendConfig(models_dir=str(root), context_length=512,
                                       sampling_overrides={'top_k':8},
                                       runtime_overrides={'device':'none', 'n_batch':64},
                                       kv_cache_type='q8_0'))
    monkeypatch.setenv('TLLAMA_CONTEXT_LENGTH','256')
    monkeypatch.setenv('TLLAMA_SAMPLING_TOP_K','2')
    [copied] = await other.list_local_models_with_metadata()
    assert copied['digest'] == original['digest'] == sha256(llm.read_bytes()).hexdigest()


async def test_changed_projector_and_continuation_shard_change_id(manager, artifact, toml_file):
    first = artifact('model-00001-of-00002.gguf', **{'split.count':2,'split.no':0})
    second = artifact('model-00002-of-00002.gguf', **{'split.count':2,'split.no':1})
    proj = artifact('proj.gguf','clip')
    define(toml_file, llm=first.name, projector=proj.name)
    [before] = await manager.list_local_models_with_metadata()
    with second.open('ab') as handle:
        handle.write(b'changed tensors')
    [changed_shard] = await manager.list_local_models_with_metadata()
    with proj.open('ab') as handle:
        handle.write(b'changed projector')
    [changed_proj] = await manager.list_local_models_with_metadata()
    assert len({item['digest'] for item in (before, changed_shard, changed_proj)}) == 3


async def test_real_header_defaults_and_metadata_cache_version_are_used(manager, artifact, toml_file):
    import json
    from tllama.helpers.metadata_cache import get_metadata_cache_path, SCHEMA_VERSION
    path = artifact('model.gguf', **{'llama.context_length':8192, 'llama.rope.freq_base':1000000.0,
                                    'llama.rope.scaling.type':'linear', 'llama.rope.scaling.factor':2.0,
                                    'llama.pooling_type':3, 'general.sampling.temp':0.6})
    define(toml_file, extra='[sampling]\ntemperature=0.6\n[runtime]\nn_ctx=8192\n'
                           'rope_freq_base=1000000\nrope_freq_scale=0.5\n'
                           'rope_scaling_type=1\npooling_type=3\nyarn_beta_fast=32.0\n')
    [model] = await manager.list_local_models_with_metadata()
    assert model['digest'] == sha256(path.read_bytes()).hexdigest()
    cache = json.loads(get_metadata_cache_path(manager.metadata_cache_dir,path).read_text())
    assert SCHEMA_VERSION == 9 and cache['schema_version'] == 9
    assert cache['metadata']['metadata_raw']['llama.rope.freq_base'] == 1000000.0


async def test_cache_v1_is_rebuilt_and_mid_hash_changes_cannot_poison_cache(manager, artifact, monkeypatch):
    import json
    path = artifact('model.gguf')
    await manager.get_model_digest(path)
    cache_path = get_digest_cache_path(manager.metadata_cache_dir, path)
    document = json.loads(cache_path.read_text())
    document['schema_version'] = 1
    document['digest']['content_sha256'] = 'old unverified digest'
    cache_path.write_text(json.dumps(document))
    stat = path.stat()
    os.utime(cache_path, ns=(stat.st_mtime_ns,stat.st_mtime_ns))
    assert DIGEST_SCHEMA_VERSION == 2 and load_digest_cache(manager.metadata_cache_dir,path) is None
    original = manager._compute_content_sha256
    calls = []
    def change_during_hash(file):
        result = original(file)
        calls.append(result)
        if len(calls) == 1:
            with path.open('ab') as handle:
                handle.write(b'replaced during hashing')
        return result
    monkeypatch.setattr(manager,'_compute_content_sha256',change_during_hash)
    digest = await manager.get_model_digest(path)
    assert len(calls) == 2 and digest['content_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert load_digest_cache(manager.metadata_cache_dir,path) == digest


async def test_unreadable_component_does_not_fall_back_to_plain_id(manager, artifact, toml_file, monkeypatch):
    artifact('model.gguf')
    proj = artifact('proj.gguf','clip')
    define(toml_file, projector=proj.name)
    original = manager._compute_content_sha256
    def hash_file(path):
        if path == proj:
            raise OSError('unreadable projector')
        return original(path)
    monkeypatch.setattr(manager,'_compute_content_sha256',hash_file)
    [model] = await manager.list_local_models_with_metadata()
    assert model['digest'] == ''


async def test_toml_changed_while_hashing_is_retried_without_stale_id(manager, artifact, toml_file, monkeypatch):
    path = artifact('model.gguf')
    definition = define(toml_file)
    original = manager._compute_content_sha256
    calls = []
    def change_definition(file):
        calls.append(file)
        definition.write_text(definition.read_text()+'[sampling]\ntop_k=20\n')
        return original(file)
    monkeypatch.setattr(manager,'_compute_content_sha256',change_definition)
    [model] = await manager.list_local_models_with_metadata()
    assert calls == [path]
    assert model['digest'] == model_digest(sha256(path.read_bytes()).hexdigest(),
                                           changes={'sampling.top_k':20})


async def test_tags_and_loaded_generation_share_id_and_ps_never_rehashes(public, monkeypatch):
    from tllama.routers import ollama
    from test_native_reporting import ps
    manager, client, name, toml = public
    monkeypatch.setattr(ollama,'model_manager',manager)
    toml.write_text(toml.read_text()+'\ntop_k=10\n')
    # Cold admission computes the same model ID even with a request context
    # override; the request is not part of the persisted definition.
    await manager.preload_native_model(name, num_ctx=4096, keep_alive=-1)
    [listed] = (await client.get('/api/tags')).json()['models']
    [loaded] = await ps(client)
    assert loaded['digest'] == listed['digest'] != ''
    toml.write_text(toml.read_text().replace('top_k=10','top_k=20'))
    def fail(*args):
        pytest.fail('ps touched model files')
    monkeypatch.setattr(manager,'_listing_identity',fail)
    monkeypatch.setattr(manager,'_compute_content_sha256',fail)
    [still_loaded] = await ps(client)
    assert still_loaded['digest'] == loaded['digest']


# Native subprocess integration uses the shared owned-worker fixture.
from test_native_public_api import public
from test_native_manager import native_manager
