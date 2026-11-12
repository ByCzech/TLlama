"""A standalone projector pull owns a discoverable, independent definition."""

import json
from pathlib import Path

import pytest

from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import copy_definition
from tllama.repository_check import check_repository
from tllama.repository_names import definition_path
from tllama.repository_prune import prune_repository
from tllama.repository_runtime import origin_path
from tllama.routers import ollama
from test_repository_runtime import hf


def definition(manager, name='ns/repo/mmproj'):
    return manager.models_dir / definition_path(name)


@pytest.mark.parametrize('stream', [False, True])
def test_public_pull_registers_projector_and_show_explains_its_use(manager, hf, client, monkeypatch, stream):
    hf.add('mmproj.gguf', arch='clip')
    monkeypatch.setattr(ollama, 'model_manager', manager)
    response = client.post('/api/pull', json={'model': 'ns/repo/mmproj', 'stream': stream})
    assert response.status_code == 200, response.text
    data = (json.loads(response.text.splitlines()[-1]) if stream else response.json())
    assert set(data) == {'status', 'path'} and data['status'] == 'success'
    path = Path(data['path'])
    relative = path.relative_to(manager.repository.root).as_posix()
    show = client.post('/api/show', json={'model': 'ns/repo/mmproj'})
    assert show.status_code == 200, show.text
    assert show.json()['model_definition'] == {'type': 'mmproj', 'artifacts': {'mmproj': relative}}
    assert show.json()['capabilities'] == []
    assert set(parse_model_toml(show.json()['modelfile']).document) == {'mmproj'}
    assert client.get('/api/tags').json()['models'] == []
    assert [row['name'] for row in client.get('/api/tags?type=mmproj').json()['models']] == ['ns/repo/mmproj']
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('split', [False, True])
def test_projector_definition_protects_all_shards_from_prune_and_is_removed_explicitly(manager, hf, split):
    files = ['mmproj-00001-of-00002.gguf', 'mmproj-00002-of-00002.gguf'] if split else ['mmproj.gguf']
    for number, name in enumerate(files):
        hf.add(name, arch='clip', shard=number if split else None, count=2 if split else None)
    path = Path(manager._pull_hf_file_sync('ns/repo', files[0]))
    name = 'ns/repo/' + Path(files[0]).stem
    checked = check_repository(manager.repository, hashes=True)
    assert checked['ok'] and not checked['issues']
    assert checked['groups'][0]['references'] == [name]
    assert checked['groups'][0]['usage'] == 'referenced'
    pruned = prune_repository(manager.repository, apply=True)
    assert pruned['ok'] and not pruned['removed_groups'] and not pruned['candidates']
    members = sorted(path.parent.glob('*.gguf'))
    assert len(members) == len(files)
    for member, original in zip(members, files):
        assert member.with_suffix('.inf').read_text().splitlines() == [original]
    result = manager.delete_model_definition(name)
    assert result['cleanup']['removed_groups'] == [path.parent.name]
    assert not path.parent.exists() and not definition(manager, name).exists()


@pytest.mark.parametrize('legacy', [False, True])
def test_repeat_pull_recreates_missing_definition_and_inf_from_verified_origin(manager, hf, legacy):
    filename = 'nested/Projector:Q8%.gguf'
    hf.add(filename, arch='clip')
    path = Path(manager._pull_hf_file_sync('ns/repo', filename))
    target = definition(manager, 'ns/repo/Projector:Q8%')
    inode = path.stat().st_ino
    target.unlink()
    path.with_suffix('.inf').unlink()
    origin = manager.repository.root / origin_path('ns/repo', filename)
    if legacy:
        record = json.loads(origin.read_text())
        record['model'] = None  # 1.4.x registered projector data without a TOML.
        origin.write_text(json.dumps(record))
    assert manager._pull_hf_file_sync('ns/repo', filename) == str(path)
    spec = parse_model_toml(target.read_text())
    assert spec.model_type == 'mmproj' and spec.mmproj_model == path.relative_to(manager.repository.root).as_posix()
    assert path.stat().st_ino == inode and len(hf.calls) == 1
    assert json.loads(origin.read_text())['model'] == 'ns/repo/Projector:Q8%'
    assert path.with_suffix('.inf').read_text().splitlines() == [Path(filename).name]


@pytest.mark.parametrize('metadata_found', [False, True])
def test_repeat_projector_pull_preserves_verbatim_customization(manager, hf, monkeypatch, metadata_found):
    hf.add('mmproj.gguf', arch='clip')
    path = manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf')
    target = definition(manager)
    custom = '# User comment\r\n' + target.read_text().replace('\n', '\r\n')
    target.write_bytes(custom.encode())
    if not metadata_found:
        from tllama.backend import HfFileLookup, HF_LOOKUP_UNKNOWN
        monkeypatch.setattr(manager, '_fetch_hf_file_info_sync', lambda *args: HfFileLookup(HF_LOOKUP_UNKNOWN))
    assert manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf') == path
    assert target.read_bytes() == custom.encode()
    assert len(hf.calls) == (1 if metadata_found else 2)


@pytest.mark.parametrize('keep_alias', [False, True])
def test_changed_projector_updates_its_owned_pointer_and_preserves_other_users(manager, hf, keep_alias):
    hf.add('mmproj.gguf', arch='clip')
    old = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
    target = definition(manager)
    target.write_text('# Keep comment\n' + target.read_text())
    if keep_alias:
        copy_definition(manager, 'ns/repo/mmproj', 'snapshot')
    hf.files['ns/repo', 'mmproj.gguf'] += b'changed payload'
    new = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
    assert new != old and new.is_file()
    assert '# Keep comment' in target.read_text()
    assert parse_model_toml(target.read_text()).mmproj_model == new.relative_to(manager.repository.root).as_posix()
    assert old.is_file() is keep_alias
    if keep_alias:
        assert parse_model_toml(manager.read_model_definition('snapshot')).mmproj_model == old.relative_to(manager.repository.root).as_posix()


def test_repointed_helper_is_not_overwritten_and_numbered_name_stays_stable(manager, hf):
    hf.add('mmproj.gguf', arch='clip')
    hf.add('other.gguf', arch='clip')
    hf.files['ns/repo', 'other.gguf'] += b'another projector'
    original = manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf')
    manager._pull_hf_file_sync('ns/repo', 'other.gguf')
    target = definition(manager)
    custom = definition(manager, 'ns/repo/other').read_text()
    target.write_text(custom)
    for _ in range(2):
        assert manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf') == original
    assert target.read_text() == custom
    assert parse_model_toml(definition(manager, 'ns/repo/mmproj_01').read_text()).model_type == 'mmproj'
    assert len(list(manager.models_dir.rglob('*.toml'))) == 3
    assert len(hf.calls) == 2


@pytest.mark.parametrize('other_role', ['llm', 'draft'])
def test_same_data_in_another_role_does_not_replace_standalone_helper_registration(manager, hf, other_role):
    hf.add('mmproj.gguf', arch='clip')
    path = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
    target = definition(manager)
    relative = path.relative_to(manager.repository.root).as_posix()
    custom = f'[{other_role}]\nmodel = {json.dumps(relative)}\n'
    target.write_text(custom)
    manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf')
    assert target.read_text() == custom
    assert parse_model_toml(definition(manager, 'ns/repo/mmproj_01').read_text()).model_type == 'mmproj'
    assert len(list((manager.models_dir / 'data').iterdir())) == 1


def test_distinct_origins_share_data_and_keep_their_own_helper_names(manager, hf):
    hf.add('mmproj.gguf', arch='clip')
    hf.add('alias.gguf', repo='other/repo', arch='clip')
    first = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
    assert manager._pull_hf_file_sync('other/repo', 'alias.gguf') == str(first)
    assert first.with_suffix('.inf').read_text().splitlines() == ['mmproj.gguf', 'alias.gguf']
    assert {row['id'] for row in manager._list_local_models_sync('mmproj')} == {'ns/repo/mmproj', 'other/repo/alias'}
    assert len(list((manager.models_dir / 'data').iterdir())) == 1
    manager.delete_model_definition('ns/repo/mmproj')
    assert first.is_file()
    manager.delete_model_definition('other/repo/alias')
    assert not first.parent.exists()


def test_main_copy_references_pulled_projector_independently(manager, hf):
    hf.add('main.gguf')
    hf.add('mmproj.gguf', arch='clip')
    manager._pull_hf_file_sync('ns/repo', 'main.gguf')
    projector = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
    copy_definition(manager, 'ns/repo/main', 'vision', mmproj='ns/repo/mmproj')
    manager.delete_model_definition('ns/repo/mmproj')
    assert projector.is_file()
    assert parse_model_toml(manager.read_model_definition('vision')).mmproj_model == projector.relative_to(manager.repository.root).as_posix()
    manager.delete_model_definition('vision')
    assert not projector.parent.exists()


@pytest.mark.parametrize('filename, arch, model_type', [
    ('anonymous.gguf', 'clip', 'mmproj'), ('mmproj-misleading.gguf', 'llama', 'main'),
])
def test_registration_uses_gguf_role_instead_of_filename(manager, hf, filename, arch, model_type):
    hf.add(filename, arch=arch)
    manager._pull_hf_file_sync('ns/repo', filename)
    assert parse_model_toml(definition(manager, 'ns/repo/' + Path(filename).stem).read_text()).model_type == model_type


@pytest.mark.parametrize('existing', [False, True])
def test_failed_helper_definition_write_does_not_commit_origin_or_remove_old_data(manager, hf, monkeypatch, existing):
    hf.add('mmproj.gguf', arch='clip')
    origin = manager.repository.root / origin_path('ns/repo', 'mmproj.gguf')
    old = old_definition = old_origin = None
    if existing:
        old = Path(manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf'))
        old_definition, old_origin = definition(manager).read_bytes(), origin.read_bytes()
        hf.files['ns/repo', 'mmproj.gguf'] += b'new projector'
    def fail(*args, **kwargs):
        raise OSError('cannot publish helper TOML')
    monkeypatch.setattr('tllama.repository_runtime.write_model_toml', fail)
    with pytest.raises(OSError, match='cannot publish helper TOML'):
        manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf')
    if existing:
        assert old.is_file()
        assert definition(manager).read_bytes() == old_definition
        assert origin.read_bytes() == old_origin
    else:
        assert not definition(manager).exists() and not origin.exists()
