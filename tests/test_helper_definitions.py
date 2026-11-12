"""Helper TOMLs are independent references, never independent inference workers."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tllama.helpers.model_toml import parse_model_toml, TomlModelError
from tllama.model_definitions import create_definition, copy_definition
from tllama.model_uploads import blob_path
from tllama.repository_check import check_repository
from tllama.repository_cleanup import definition_groups, collect_pending
from tllama.repository_names import definition_path
from tllama.repository_pins import Reservation
from tllama.repository_prune import prune_repository
from tllama.repository_store import prepare_group
from test_model_create import gguf_file


def reference(role, value, key='model'):
    return f'[{role}]\n{key} = {json.dumps(value)}\n'


def publish(manager, tmp_path, arch='llama', count=1):
    sources = [gguf_file(tmp_path / (f'{arch}-{index+1:05d}-of-{count:05d}.gguf'
                                    if count > 1 else f'{arch}.gguf'), arch=arch,
                         shard=index if count > 1 else None, count=count if count > 1 else None)
               for index in range(count)]
    paths = manager.repository.publish(prepare_group(sources))
    return paths, paths[0].relative_to(manager.repository.root).as_posix()


@pytest.mark.parametrize('role', ['mmproj', 'draft'])
@pytest.mark.parametrize('key', ['model', 'from'])
def test_standalone_helper_type_is_derived_from_its_only_role(role, key):
    text = '# Keep spelling and comments\n' + reference(role, 'source.gguf', key)
    spec = parse_model_toml(text)
    assert spec.model_type == role and spec.llm_model is None and spec.llm_from is None
    assert getattr(spec, role + '_' + key) == 'source.gguf'
    assert spec.document.as_string() == text


@pytest.mark.parametrize('body, message', [
    ('', 'expected'),
    ('[draft]\n', 'either'),
    ('[draft]\nmodel="a"\nfrom="b"\n', 'both'),
    ('[draft]\nmodel=123\n', 'string'),
    ('[draft]\nmodel="a"\nstrategy="dflash"\n', 'not a key'),
    ('[mmproj]\nmodel="a"\n[draft]\nmodel="b"\n', 'exactly one'),
    ('[draft]\nmodel="a"\n[runtime]\nspec_type="none"\n', 'inference settings'),
    ('[mmproj]\nmodel="a"\nwarmup=false\n', 'inference settings'),
])
def test_ambiguous_or_unused_helper_settings_are_rejected(body, message):
    with pytest.raises(TomlModelError, match=message):
        parse_model_toml(body)


def test_main_can_reference_one_projector_and_one_draft_without_enabling_it():
    text = ''.join(reference(role, role + '.gguf') for role in ('llm', 'mmproj', 'draft'))
    spec = parse_model_toml(text)
    assert spec.model_type == 'main' and spec.runtime == {}
    assert spec.components == (('llm', 'llm.gguf', None), ('mmproj', 'mmproj.gguf', None),
                               ('draft', 'draft.gguf', None))


@pytest.mark.parametrize('role, arch', [('mmproj', 'clip'), ('draft', 'qwen3')])
@pytest.mark.parametrize('count', [1, 2])
def test_standalone_helper_protects_complete_group_from_prune_and_is_removable(
        manager, tmp_path, role, arch, count):
    paths, relative = publish(manager, tmp_path, arch, count)
    create_definition(manager, 'helper', reference(role, relative), {})
    definition = manager.models_dir / definition_path('helper')
    assert definition_groups(definition) == {paths[0].parent.name: [path.name for path in paths]}
    checked = check_repository(manager.repository)
    assert checked['ok'] and checked['groups'][0]['usage'] == 'referenced'
    preview = prune_repository(manager.repository)
    assert preview['ok'] and preview['candidates'] == []
    assert manager._list_local_models_sync() == []
    with pytest.raises(ValueError, match=f'{role} helper'):
        manager._resolve_native_definition_sync('helper', None)
    assert manager.native_snapshot() == []
    copy_definition(manager, 'helper', 'helper-alias')
    manager.delete_model_definition('helper')
    assert all(path.is_file() for path in paths)
    last = manager.delete_model_definition('helper-alias')
    assert not last['cleanup']['warnings']
    assert not paths[0].parent.exists()


@pytest.mark.parametrize('role, arch', [('mmproj', 'clip'), ('draft', 'qwen3')])
@pytest.mark.parametrize('delete_first', ['helper', 'main'])
def test_main_and_helper_definitions_are_independent_data_references(
        manager, tmp_path, role, arch, delete_first):
    main_paths, main = publish(manager, tmp_path)
    helper_paths, helper = publish(manager, tmp_path, arch)
    create_definition(manager, 'helper', reference(role, helper), {})
    create_definition(manager, 'main', reference('llm', main) + reference(role, helper), {})
    first = manager.delete_model_definition(delete_first)
    assert not first['cleanup']['warnings']
    assert all(path.is_file() for path in helper_paths)
    manager.delete_model_definition('main' if delete_first == 'helper' else 'helper')
    assert not main_paths[0].parent.exists() and not helper_paths[0].parent.exists()


@pytest.mark.parametrize('role, arch', [('mmproj', 'clip'), ('draft', 'qwen3')])
def test_composite_create_adds_no_implicit_helper_definition(manager, tmp_path, role, arch):
    main_paths, main = publish(manager, tmp_path)
    helper_paths, helper = publish(manager, tmp_path, arch, count=2)
    create_definition(manager, 'main', reference('llm', main) + reference(role, helper), {})
    assert list(manager.models_dir.rglob('*.toml')) == [manager.models_dir / definition_path('main')]
    result = manager.delete_model_definition('main')
    assert set(result['cleanup']['removed_groups']) == {main_paths[0].parent.name, helper_paths[0].parent.name}


def test_reservation_defers_deletion_of_last_helper_definition(manager, tmp_path):
    paths, helper = publish(manager, tmp_path, 'qwen3', count=2)
    create_definition(manager, 'helper', reference('draft', helper), {})
    with Reservation(manager.repository, 'test', groups=[paths[0].parent.name]):
        cleanup = manager.delete_model_definition('helper')['cleanup']
        assert cleanup['deferred_groups'] == [paths[0].parent.name]
        assert all(path.is_file() for path in paths)
    collect_pending(manager.repository)
    assert not paths[0].parent.exists()


def test_unimported_draft_does_not_make_reference_inventory_look_complete(manager, tmp_path):
    paths, main = publish(manager, tmp_path)
    create_definition(manager, 'main', reference('llm', main), {})
    unfinished = manager.models_dir / definition_path('unfinished')
    unfinished.write_text(reference('llm', main) + reference('draft', '/client/draft.gguf', 'from'))
    with pytest.raises(ValueError, match='Unpublished'):
        definition_groups(unfinished)
    result = manager.delete_model_definition('main')
    assert result['cleanup']['warnings'] and paths[0].is_file()


@pytest.mark.parametrize('reference_path', ['Local/draft.gguf', 'data/' + '0' * 64 + '/weights.gguf'])
def test_create_rejects_unmanaged_or_missing_draft_before_publication(manager, tmp_path, reference_path):
    _, main = publish(manager, tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        create_definition(manager, 'main', reference('llm', main) + reference('draft', reference_path), {})
    assert not list(manager.models_dir.rglob('*.toml'))


@pytest.mark.parametrize('role, arch', [('mmproj', 'clip'), ('draft', 'qwen3')])
def test_server_import_publishes_only_explicitly_named_helper(manager, tmp_path, role, arch):
    source = gguf_file(tmp_path / f'{role}.gguf', arch=arch)
    digest = 'sha256:' + sha256(source.read_bytes()).hexdigest()
    manager.repository._directory('.tllama/uploads')
    blob_path(manager, digest).write_bytes(source.read_bytes())
    create_definition(manager, 'helper', reference(role, '/client/source.gguf', 'from'),
                      {role: [SimpleNamespace(filename=source.name, digest=digest)]})
    text = manager.read_model_definition('helper')
    spec = parse_model_toml(text)
    assert spec.model_type == role and getattr(spec, role + '_from') is None
    relative = getattr(spec, role + '_model')
    assert (manager.models_dir / relative).read_bytes() == source.read_bytes()
    assert Path(relative).parts[0] == 'data'
    assert len(list(manager.models_dir.rglob('*.toml'))) == 1
    assert (manager.models_dir / relative).with_suffix('.inf').read_text() == source.name + '\n'
