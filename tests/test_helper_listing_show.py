"""Read-only helper discovery uses definitions without starting native workers."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_names import definition_path
from tllama.routers import ollama, openai
from test_helper_definitions import publish, reference


@pytest.fixture
def listing_repo(manager, tmp_path, monkeypatch, client, request):
    if getattr(request, 'param', False):
        alias = tmp_path / 'repo-alias'
        alias.symlink_to(manager.repository.root, target_is_directory=True)
        manager = ModelManager(BackendConfig(models_dir=str(alias)))
        manager.ensure_storage()
    monkeypatch.setattr(ollama, 'model_manager', manager)
    monkeypatch.setattr(openai, 'model_manager', manager)
    paths, refs = {}, {}
    for role, arch in [('llm', 'llama'), ('mmproj', 'clip'), ('draft', 'qwen3')]:
        paths[role], refs[role] = publish(manager, tmp_path, arch, count=2 if role != 'llm' else 1)
    for role in ('mmproj', 'draft'):
        create_definition(manager, role, reference(role, refs[role]), {})
    create_definition(manager, 'plain', reference('llm', refs['llm']), {})
    full = '# Keep this comment\n' + ''.join(reference(role, refs[role]) for role in refs)
    create_definition(manager, 'full', full, {})
    return SimpleNamespace(manager=manager, client=client, paths=paths, refs=refs, full=full)


@pytest.mark.parametrize('listing_repo', [False, True], indirect=True)
@pytest.mark.parametrize('kind, expected', [
    (None, {'plain', 'full'}), ('main', {'plain', 'full'}),
    ('all', {'plain', 'full', 'mmproj', 'draft'}), ('mmproj', {'mmproj'}), ('draft', {'draft'}),
])
def test_tags_filters_helper_roles_and_reports_complete_sizes(listing_repo, kind, expected):
    repo = listing_repo
    response = repo.client.get('/api/tags', params={'type': kind} if kind is not None else {})
    assert response.status_code == 200, response.text
    data = response.json()
    assert {row['name'] for row in data['models']} == expected
    assert data['server_features'] == ['repository_helpers']
    for row in data['models']:
        role = row['name'] if row['name'] in {'mmproj', 'draft'} else 'main'
        assert row['model_definition'] == {'type': role}
        files = (repo.paths['llm'] if row['name'] == 'plain' else repo.paths[row['name']]
                 if role != 'main' else [path for paths in repo.paths.values() for path in paths])
        assert row['size'] == sum(path.stat().st_size for path in files)
        assert len(row['digest']) == 64
    assert repo.manager.native_snapshot() == []


@pytest.mark.parametrize('listing_repo', [False, True], indirect=True)
@pytest.mark.parametrize('name', ['plain', 'full', 'mmproj', 'draft'])
def test_show_has_repo_relative_artifacts_and_verbatim_toml(listing_repo, name):
    repo = listing_repo
    response = repo.client.post('/api/show', json={'model': name})
    assert response.status_code == 200, response.text
    data = response.json()
    role = name if name in {'mmproj', 'draft'} else 'main'
    expected = repo.refs if name == 'full' else {name if role != 'main' else 'llm': repo.refs[name if role != 'main' else 'llm']}
    assert data['model_definition'] == {'type': role, 'artifacts': expected}
    assert all(not Path(path).is_absolute() for path in data['model_definition']['artifacts'].values())
    assert data['modelfile'] == repo.manager.read_model_definition(name)
    assert parse_model_toml(data['modelfile']).model_type == role
    if role != 'main':
        assert data['capabilities'] == [] and data['template'] == '' and data['parameters'] == ''
        assert not ({'renderer', 'parser', 'template_source', 'thinking'} & data.keys())
        assert data['model_info']['general.architecture'] == ('clip' if role == 'mmproj' else 'qwen3')
    elif name == 'full':
        assert 'completion' in data['capabilities'] and 'vision' in data['capabilities']
    assert repo.manager.native_snapshot() == []


def test_openai_model_list_excludes_helpers(listing_repo):
    data = listing_repo.client.get('/v1/models').json()
    assert {row['id'] for row in data['data']} == {'plain', 'full'}
    assert all('model_definition' not in row for row in data['data'])


@pytest.mark.parametrize('kind', ['other', '', 'MAIN'])
def test_invalid_filter_is_rejected_before_repository_scan(listing_repo, monkeypatch, kind):
    def no_scan():
        raise AssertionError('Invalid filter must not inspect model storage')
    monkeypatch.setattr(listing_repo.manager, '_iter_repository_toml_files', no_scan)
    assert listing_repo.client.get('/api/tags', params={'type': kind}).status_code == 400


@pytest.mark.parametrize('role', ['mmproj', 'draft'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('path, body', [
    ('/api/generate', {'prompt': 'hello', 'stream': False}),
    ('/api/chat', {'messages': [{'role': 'user', 'content': 'hello'}], 'stream': False}),
    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hello'}], 'stream': False}),
])
def test_helper_inference_fails_before_native_start(listing_repo, role, path, body, stream):
    response = listing_repo.client.post(path, json={'model': role, **body, 'stream': stream})
    assert response.status_code == 400, response.text
    assert f'{role} helper' in response.text
    assert listing_repo.manager.native_snapshot() == []


def test_non_generation_main_models_are_not_filtered_out(listing_repo):
    create_definition(listing_repo.manager, 'embedding', reference('llm', listing_repo.refs['llm']) +
                      '[runtime]\nembedding=true\npooling_type=1\n', {})
    data = listing_repo.client.get('/api/tags').json()
    rows = {row['name']: row for row in data['models']}
    assert set(rows) == {'plain', 'full', 'embedding'}
    assert rows['embedding']['model_definition']['type'] == 'main'


async def test_helper_metadata_never_inherits_global_embedding_or_template(listing_repo, monkeypatch):
    manager = listing_repo.manager
    original = manager._get_raw_model_metadata
    async def with_template(name):
        return {**await original(name), 'template': 'a generation template', 'tool_template': 'tool template'}
    monkeypatch.setattr(manager, '_get_raw_model_metadata', with_template)
    manager.config.runtime_overrides['embedding'] = True
    metadata = await manager.get_model_metadata('draft')
    assert metadata['template'] == '' and metadata['tool_template'] == ''
    assert metadata['embedding_configured'] is False and metadata['has_projector'] is False


def test_show_fallback_still_describes_helper_role(listing_repo, monkeypatch):
    async def no_metadata(name):
        return None
    monkeypatch.setattr(listing_repo.manager, 'get_model_metadata', no_metadata)
    response = listing_repo.client.post('/api/show', json={'model': 'draft'})
    assert response.status_code == 200
    data = response.json()
    assert data['model_definition'] == {'type': 'draft', 'artifacts': {'draft': listing_repo.refs['draft']}}
    assert data['capabilities'] == [] and data['template'] == ''


async def test_definition_changing_to_helper_during_enrichment_is_not_leaked_in_main_list(listing_repo, monkeypatch):
    manager = listing_repo.manager
    original = manager.get_model_metadata
    async def changed(name):
        if name == 'plain':
            (manager.models_dir / definition_path(name)).write_text(reference('draft', listing_repo.refs['draft']))
        return await original(name)
    monkeypatch.setattr(manager, 'get_model_metadata', changed)
    rows = await manager.list_local_models_with_metadata()
    assert {row['id'] for row in rows} == {'full'}


def test_helper_aliases_share_id_but_roles_are_distinct(listing_repo):
    manager = listing_repo.manager
    copy_definition(manager, 'draft', 'draft-alias')
    assert manager._listing_identity('draft') == manager._listing_identity('draft-alias')
    main = listing_repo.paths['llm'][0]
    assert manager._listing_identity('plain')[0] == sha256(main.read_bytes()).hexdigest()
    helper = manager._listing_identity('draft')[0]
    create_definition(manager, 'draft-other-name', reference('draft', listing_repo.refs['draft']), {})
    assert manager._listing_identity('draft-other-name')[0] == helper
    assert len({manager._listing_identity(name)[0] for name in ('plain', 'full', 'mmproj', 'draft')}) == 4


def test_broken_helper_is_isolated_from_other_models(listing_repo):
    listing_repo.paths['draft'][1].unlink()
    data = listing_repo.client.get('/api/tags', params={'type': 'all'}).json()
    assert {row['name'] for row in data['models']} == {'plain', 'mmproj'}
