"""Compose complete shared data groups, committing only one requested TOML."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_cleanup import QUEUE
from tllama.repository_names import definition_path
from tllama.routers import ollama
from test_helper_definitions import publish, reference


@pytest.fixture
def compose_repo(manager, tmp_path, monkeypatch, client):
    refs, paths = {}, {}
    for role, arch, count in [('llm', 'llama', 1), ('old-mmproj', 'clip', 1),
                              ('mmproj', 'clip', 2), ('old-draft', 'qwen3', 1), ('draft', 'qwen3', 2)]:
        paths[role], refs[role] = publish(manager, tmp_path, arch, count)
    text = ('# Keep original formatting\n' + reference('llm', refs['llm']) +
            reference('mmproj', refs['old-mmproj']) + 'warmup=false # keep projector settings\n' +
            reference('draft', refs['old-draft']) +
            '[runtime]\nn_ctx=4096\nspec_type="none"\n[sampling]\ntemperature=0.4\n' +
            '[system]\nprompt="Czech" # keep system\n')
    create_definition(manager, 'main', text, {})
    for role in ('mmproj', 'draft'):
        create_definition(manager, role, reference(role, refs[role]), {})
    monkeypatch.setattr(ollama, 'model_manager', manager)
    return SimpleNamespace(manager=manager, client=client, refs=refs, paths=paths, text=text)


@pytest.mark.parametrize('roles', [('mmproj',), ('draft',), ('mmproj', 'draft')])
def test_copy_replaces_only_requested_roles_and_preserves_all_settings(compose_repo, roles):
    repo = compose_repo
    before = {path: (path.stat().st_ino, path.read_bytes())
              for paths in repo.paths.values() for path in paths}
    response = repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed',
                                                  **{role: role for role in roles}})
    assert response.status_code == 200, response.text
    assert response.json() == {'status': 'success', 'source': 'main', 'destination': 'composed'}
    changed = repo.manager.read_model_definition('composed')
    spec, old = parse_model_toml(changed), parse_model_toml(repo.text)
    assert spec.llm_model == old.llm_model
    for role in ('mmproj', 'draft'):
        assert getattr(spec, role + '_model') == repo.refs[role if role in roles else 'old-' + role]
    assert spec.runtime == old.runtime and spec.sampling == old.sampling
    assert spec.system_prompt == old.system_prompt and spec.mmproj_context == old.mmproj_context
    assert '# Keep original formatting' in changed and '# keep system' in changed
    assert '# keep projector settings' in changed
    assert repo.manager.read_model_definition('main') == repo.text
    assert {path: (path.stat().st_ino, path.read_bytes()) for path in before} == before
    assert len(list(repo.manager.models_dir.rglob('*.toml'))) == 4
    assert repo.manager.native_snapshot() == []


def test_copy_resolves_helper_names_once_then_keeps_direct_paths(compose_repo):
    repo = compose_repo
    copy_definition(repo.manager, 'main', 'composed', mmproj='mmproj', draft='draft')
    for role in ('mmproj', 'draft'):
        repo.manager.delete_model_definition(role)
        # Reusing the old helper name with different content must not change
        # the already assembled model.
        create_definition(repo.manager, role, reference(role, repo.refs['old-' + role]), {})
    spec = parse_model_toml(repo.manager.read_model_definition('composed'))
    assert spec.mmproj_model == repo.refs['mmproj'] and spec.draft_model == repo.refs['draft']
    assert all(path.is_file() for role in ('mmproj', 'draft') for path in repo.paths[role])
    repo.manager.delete_model_definition('composed')
    assert all(not repo.paths[role][0].parent.exists() for role in ('mmproj', 'draft'))
    assert (repo.manager.models_dir / repo.refs['llm']).is_file()


@pytest.mark.parametrize('role, helper, status', [
    ('mmproj', 'missing', 404), ('mmproj', 'draft', 400), ('draft', 'mmproj', 400), ('draft', 'main', 400),
])
def test_invalid_helper_does_not_replace_destination_or_schedule_cleanup(compose_repo, role, helper, status):
    repo = compose_repo
    copy_definition(repo.manager, 'main', 'composed')
    path = repo.manager.models_dir / definition_path('composed')
    before = path.read_bytes()
    response = repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed',
                                                  'force': True, role: helper})
    assert response.status_code == status, response.text
    assert path.read_bytes() == before
    assert not (repo.manager.repository.root / QUEUE).exists()
    assert all(path.is_file() for paths in repo.paths.values() for path in paths)


def test_second_bad_helper_cannot_leave_first_binding_partly_published(compose_repo):
    response = compose_repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed',
                                                          'mmproj': 'mmproj', 'draft': 'missing'})
    assert response.status_code == 404
    assert compose_repo.manager.read_model_definition('composed') is None


def test_attach_adds_missing_sections_without_creating_helper_aliases(compose_repo):
    repo = compose_repo
    original = reference('llm', repo.refs['llm']) + '[system]\nprompt="Keep me"\n'
    create_definition(repo.manager, 'plain', original, {})
    copy_definition(repo.manager, 'plain', 'composed', mmproj='mmproj', draft='draft')
    spec = parse_model_toml(repo.manager.read_model_definition('composed'))
    assert spec.mmproj_model == repo.refs['mmproj'] and spec.draft_model == repo.refs['draft']
    assert spec.system_prompt == 'Keep me'
    assert len(list(repo.manager.models_dir.rglob('*.toml'))) == 5


def test_attachment_does_not_import_client_side_main_source(compose_repo):
    repo = compose_repo
    (repo.manager.models_dir / definition_path('pending')).write_text(reference('llm', '/client/model.gguf', 'from'))
    response = repo.client.post('/api/copy', json={'source': 'pending', 'destination': 'composed', 'draft': 'draft'})
    assert response.status_code == 400 and '[llm].from must be uploaded' in response.text
    assert repo.manager.read_model_definition('composed') is None


@pytest.mark.parametrize('role, value', [('mmproj', None), ('draft', None), ('mmproj', ''), ('draft', ''),
    ('mmproj', 1), ('draft', False), ('mmproj', []), ('draft', {}), ('mmproj', '../bad'), ('draft', '/path')])
def test_invalid_helper_fields_fail_before_repository_work(compose_repo, monkeypatch, role, value):
    def no_copy(*args, **kwargs):
        raise AssertionError('Invalid helper fields must not reach repository mutation')
    monkeypatch.setattr(ollama, 'copy_definition', no_copy)
    response = compose_repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed', role: value})
    assert response.status_code == 400, response.text


@pytest.mark.parametrize('broken', ['unimported', 'projector-as-draft', 'model-as-projector', 'missing-shard'])
def test_unusable_helpers_are_rejected_before_target_publication(compose_repo, broken):
    repo = compose_repo
    role = 'draft'
    if broken == 'unimported':
        text = reference('draft', '/client/draft.gguf', 'from')
    elif broken == 'projector-as-draft':
        text = reference('draft', repo.refs['mmproj'])
    elif broken == 'model-as-projector':
        role = 'mmproj'
        text = reference(role, repo.refs['llm'])
    else:
        repo.paths['draft'][1].unlink()
        text = reference('draft', repo.refs['draft'])
    bad = repo.manager.models_dir / definition_path('bad')
    bad.write_text(text)
    response = repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed', role: 'bad'})
    assert response.status_code in {400, 404}, response.text
    assert repo.manager.read_model_definition('composed') is None


def test_helper_sources_cannot_accept_attached_helpers(compose_repo):
    response = compose_repo.client.post('/api/copy', json={'source': 'draft', 'destination': 'composed', 'mmproj': 'mmproj'})
    assert response.status_code == 400 and 'main model source' in response.text


def test_replacing_unusable_old_binding_validates_the_final_definition(compose_repo):
    repo = compose_repo
    path = repo.manager.models_dir / definition_path('main')
    path.write_text(reference('llm', repo.refs['llm']) + reference('mmproj', '/client/not-present.gguf', 'from'))
    copy_definition(repo.manager, 'main', 'composed', mmproj='mmproj')
    assert parse_model_toml(repo.manager.read_model_definition('composed')).mmproj_model == repo.refs['mmproj']
    assert path.read_text().endswith('from = "/client/not-present.gguf"\n')


def test_attaching_preserves_crlf_and_projector_context(compose_repo):
    repo = compose_repo
    path = repo.manager.models_dir / definition_path('main')
    path.write_bytes(repo.text.replace('\n', '\r\n').encode())
    copy_definition(repo.manager, 'main', 'composed', mmproj='mmproj')
    raw = (repo.manager.models_dir / definition_path('composed')).read_bytes()
    assert b'\n' not in raw.replace(b'\r\n', b'')
    assert b'warmup=false # keep projector settings\r\n' in raw


def test_forced_attachment_reclaims_only_retired_destination_data(compose_repo, tmp_path):
    repo = compose_repo
    old_paths, old = publish(repo.manager, tmp_path, 'gemma')
    create_definition(repo.manager, 'composed', reference('llm', old), {})
    result = copy_definition(repo.manager, 'main', 'composed', force=True, draft='draft')
    assert result['cleanup']['removed_groups'] == [old_paths[0].parent.name]
    assert not old_paths[0].parent.exists()
    assert all(path.is_file() for paths in repo.paths.values() for path in paths)


def test_attachment_conflict_keeps_existing_target(compose_repo):
    repo = compose_repo
    copy_definition(repo.manager, 'main', 'composed')
    before = repo.manager.read_model_definition('composed')
    response = repo.client.post('/api/copy', json={'source': 'main', 'destination': 'composed', 'draft': 'draft'})
    assert response.status_code == 409
    assert repo.manager.read_model_definition('composed') == before


def test_aliased_root_and_encoded_helper_names_keep_repo_relative_paths(compose_repo, tmp_path):
    repo = compose_repo
    name = 'ByCzech/Projektor:Q8%'
    copy_definition(repo.manager, 'mmproj', name)
    alias = tmp_path / 'models-alias'
    alias.symlink_to(repo.manager.repository.root, target_is_directory=True)
    aliased = ModelManager(BackendConfig(models_dir=str(alias)))
    copy_definition(aliased, 'main', 'Český/Model:Q4%', mmproj=name, draft='draft')
    spec = parse_model_toml(aliased.read_model_definition('Český/Model:Q4%'))
    assert spec.mmproj_model == repo.refs['mmproj'] and spec.draft_model == repo.refs['draft']
    assert not Path(spec.mmproj_model).is_absolute() and not Path(spec.draft_model).is_absolute()


def test_changing_helper_name_without_changing_weights_preserves_model_id(compose_repo):
    repo = compose_repo
    create_definition(repo.manager, 'same-projector', reference('mmproj', repo.refs['old-mmproj']), {})
    before = repo.manager._listing_identity('main')
    copy_definition(repo.manager, 'main', 'composed', mmproj='same-projector')
    assert repo.manager._listing_identity('composed') == before


def test_compose_and_helper_delete_are_serialized_across_managers(compose_repo):
    repo = compose_repo
    other = ModelManager(repo.manager.config)
    barrier = threading.Barrier(2)
    def compose():
        barrier.wait(timeout=5)
        try:
            copy_definition(repo.manager, 'main', 'composed', mmproj='mmproj')
            return 'copied'
        except FileNotFoundError:
            return 'missing'
    def delete():
        barrier.wait(timeout=5)
        other.delete_model_definition('mmproj')
    with ThreadPoolExecutor(max_workers=2) as pool:
        copied, deleted = pool.submit(compose), pool.submit(delete)
        result = copied.result(timeout=10)
        deleted.result(timeout=10)
    path = repo.manager.models_dir / definition_path('composed')
    assert path.is_file() == (result == 'copied')
    assert all(file.is_file() == (result == 'copied') for file in repo.paths['mmproj'])
