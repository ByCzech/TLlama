"""Remove only attached, unshared and unambiguous helper definitions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_cleanup import collect_pending, QUEUE
from tllama.repository_names import definition_path
from tllama.repository_pins import Reservation
from tllama.routers import ollama
from tllama.schemas.model_management import DeleteRequest, ShowRequest, PullRequest
from test_helper_definitions import publish, reference
from test_repository_worker_pins import protected


@pytest.fixture
def removal_repo(manager, tmp_path, monkeypatch, client, request):
    if getattr(request, 'param', False):
        alias = tmp_path / 'models-alias'
        alias.symlink_to(manager.repository.root, target_is_directory=True)
        manager = ModelManager(BackendConfig(models_dir=str(alias)))
        manager.ensure_storage()
    paths, refs = {}, {}
    for role, arch in [('llm', 'llama'), ('mmproj', 'clip'), ('draft', 'qwen3')]:
        paths[role], refs[role] = publish(manager, tmp_path, arch, count=1 if role == 'llm' else 2)
        if role != 'llm':
            create_definition(manager, role, reference(role, refs[role]), {})
    create_definition(manager, 'main', ''.join(reference(role, value) for role, value in refs.items()), {})
    monkeypatch.setattr(ollama, 'model_manager', manager)
    return SimpleNamespace(manager=manager, client=client, paths=paths, refs=refs, tmp_path=tmp_path)


@pytest.mark.parametrize('removal_repo', [False, True], indirect=True)
@pytest.mark.parametrize('cascade', [False, True])
def test_last_main_preserves_or_removes_independent_helper_definitions(removal_repo, cascade):
    repo = removal_repo
    unrelated, unused = publish(repo.manager, repo.tmp_path, 'gemma')
    create_definition(repo.manager, 'unrelated-helper', reference('draft', unused), {})
    orphan, _ = publish(repo.manager, repo.tmp_path, 'qwen2')
    response = repo.client.request('DELETE', '/api/delete', json={'model': 'main', 'remove_unused_helpers': cascade})
    assert response.status_code == 200, response.text
    result = response.json()['cleanup']
    assert result['removed_helpers'] == (['draft', 'mmproj'] if cascade else [])
    assert result['unused_helpers'] == ([] if cascade else ['draft', 'mmproj'])
    assert not repo.paths['llm'][0].parent.exists()
    for role in ('mmproj', 'draft'):
        assert (repo.manager.models_dir / definition_path(role)).is_file() is not cascade
        assert repo.paths[role][0].parent.exists() is not cascade
    assert unrelated[0].is_file() and orphan[0].is_file()
    assert repo.manager.read_model_definition('unrelated-helper') is not None
    assert not result['warnings']


def test_helpers_used_by_another_main_are_preserved(removal_repo):
    repo = removal_repo
    create_definition(repo.manager, 'other', reference('llm', repo.refs['llm']) + reference('mmproj', repo.refs['mmproj']), {})
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
    assert result['removed_helpers'] == ['draft'] and result['unused_helpers'] == []
    assert repo.manager.read_model_definition('mmproj') is not None
    assert all(path.is_file() for role in ('llm', 'mmproj') for path in repo.paths[role])
    assert not repo.paths['draft'][0].parent.exists()


def test_any_other_main_data_reference_protects_helper_definition(removal_repo):
    repo = removal_repo
    create_definition(repo.manager, 'other', reference('llm', repo.refs['draft']), {})
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
    assert result['removed_helpers'] == ['mmproj']
    assert repo.manager.read_model_definition('draft') is not None
    assert all(path.is_file() for path in repo.paths['draft'])


def test_no_helper_toml_means_last_main_always_reclaims_all_groups(removal_repo):
    repo = removal_repo
    for role in ('mmproj', 'draft'):
        first = repo.manager.delete_model_definition(role)
        assert first['kept_model_file'] == repo.refs[role]
    result = repo.manager.delete_model_definition('main')['cleanup']
    assert result['removed_helpers'] == result['unused_helpers'] == []
    assert all(not paths[0].parent.exists() for paths in repo.paths.values())


@pytest.mark.parametrize('cascade', [False, True])
def test_ambiguous_aliases_are_kept_for_explicit_removal(removal_repo, cascade):
    repo = removal_repo
    copy_definition(repo.manager, 'draft', 'draft-alias')
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=cascade)['cleanup']
    assert result['removed_helpers'] == (['mmproj'] if cascade else [])
    assert result['unused_helpers'] == (['draft', 'draft-alias'] if cascade else ['draft', 'draft-alias', 'mmproj'])
    assert all(repo.manager.read_model_definition(name) is not None for name in ('draft', 'draft-alias'))
    assert all(path.is_file() for path in repo.paths['draft'])
    if cascade:
        assert any('draft, draft-alias' in warning for warning in result['warnings'])
    else:
        assert result['warnings'] == []


@pytest.mark.parametrize('broken', ['syntax', 'from', 'symlink', 'shards'])
def test_incomplete_inventory_never_authorizes_helper_deletion(removal_repo, broken):
    repo = removal_repo
    bad = repo.manager.models_dir / definition_path('bad')
    if broken == 'syntax':
        bad.write_text('broken toml')
    elif broken == 'from':
        bad.write_text(reference('draft', '/client/file.gguf', 'from'))
    elif broken == 'symlink':
        outside = repo.tmp_path / 'outside.toml'
        outside.write_text(reference('draft', repo.refs['draft']))
        bad.symlink_to(outside)
    else:
        bad.write_text(reference('draft', repo.refs['draft'].replace('of-00002', 'of-00003')))
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
    assert result['removed_helpers'] == []
    assert any('Helper cleanup skipped' in warning for warning in result['warnings'])
    assert all(repo.manager.read_model_definition(role) is not None for role in ('mmproj', 'draft'))
    assert all(path.is_file() for role in ('mmproj', 'draft') for path in repo.paths[role])


def test_failed_helper_unlink_preserves_reference_and_reports_partial_result(removal_repo, monkeypatch):
    repo = removal_repo
    protected_path = repo.manager.models_dir / definition_path('draft')
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == protected_path:
            raise PermissionError('cannot remove helper')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
    assert result['removed_helpers'] == ['mmproj'] and result['unused_helpers'] == ['draft']
    assert protected_path.is_file() and all(path.is_file() for path in repo.paths['draft'])
    assert any('cannot remove helper' in warning for warning in result['warnings'])


def test_corrupt_reclaim_queue_keeps_helper_definitions(removal_repo):
    repo = removal_repo
    (repo.manager.repository.root / QUEUE).write_text('invalid queue')
    result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
    assert result['removed_helpers'] == [] and result['unused_helpers'] == ['draft', 'mmproj']
    assert all(repo.manager.read_model_definition(role) is not None for role in ('mmproj', 'draft'))
    assert result['warnings']


def test_reservations_defer_physical_cleanup_after_helper_definition_removal(removal_repo):
    repo = removal_repo
    groups = [paths[0].parent.name for paths in repo.paths.values()]
    with Reservation(repo.manager.repository, 'test', groups=groups):
        result = repo.manager.delete_model_definition('main', remove_unused_helpers=True)['cleanup']
        assert result['removed_helpers'] == ['draft', 'mmproj']
        assert set(result['deferred_groups']) == set(groups)
        assert all(path.is_file() for paths in repo.paths.values() for path in paths)
    collect_pending(repo.manager.repository)
    assert all(not paths[0].parent.exists() for paths in repo.paths.values())


async def test_live_worker_keeps_projector_data_after_cascade(protected):
    manager = protected
    projector = manager.test.group('clip', split=True)
    create_definition(manager, 'projector', reference('mmproj', projector), {})
    create_definition(manager, 'a', reference('llm', manager.test.first) + reference('mmproj', projector), {}, force=True)
    async with manager.acquire_native_model('a') as lease:
        result = (await manager.delete_model('a', remove_unused_helpers=True))['cleanup']
        assert result['removed_helpers'] == ['projector']
        assert manager.read_model_definition('projector') is None
        assert (manager.models_dir / projector).is_file() and result['deferred_groups']
    await manager.wait_native_closed(lease.generation)
    collect_pending(manager.repository)
    assert not (manager.models_dir / projector).parent.exists()


def test_concurrent_copy_and_cascade_serialize_the_reference_decision(removal_repo):
    repo = removal_repo
    other = ModelManager(repo.manager.config)
    barrier = threading.Barrier(2)
    def copy():
        barrier.wait(timeout=5)
        try:
            copy_definition(other, 'main', 'copy')
            return 'copied'
        except FileNotFoundError:
            return 'missing'
    def delete():
        barrier.wait(timeout=5)
        return repo.manager.delete_model_definition('main', remove_unused_helpers=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        copying, deleting = pool.submit(copy), pool.submit(delete)
        copied = copying.result(timeout=10)
        result = deleting.result(timeout=10)['cleanup']
    assert result['removed_helpers'] == ([] if copied == 'copied' else ['draft', 'mmproj'])
    assert all(path.is_file() == (copied == 'copied') for paths in repo.paths.values() for path in paths)


@pytest.mark.parametrize('value', [None, 0, 1, 'true', 'false', [], {}])
def test_delete_flag_is_strictly_boolean_before_any_operation(removal_repo, monkeypatch, value):
    async def no_delete(*args, **kwargs):
        raise AssertionError('Invalid delete controls must not reach mutation')
    monkeypatch.setattr(removal_repo.manager, 'delete_model', no_delete)
    response = removal_repo.client.request('DELETE', '/api/delete', json={'model': 'main', 'remove_unused_helpers': value})
    assert response.status_code == 400


def test_delete_extension_does_not_leak_into_show_or_pull_schema():
    assert 'remove_unused_helpers' in DeleteRequest.model_fields
    assert 'remove_unused_helpers' not in ShowRequest.model_fields
    assert 'remove_unused_helpers' not in PullRequest.model_fields
