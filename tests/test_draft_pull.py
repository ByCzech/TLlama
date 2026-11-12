"""DFlash header architecture selects a stored draft, never a runnable LLM."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tllama.backend import ModelManager, HfFileLookup, HF_LOOKUP_UNKNOWN, PullProgress
from tllama.config import BackendConfig
from tllama.helpers.metadata_cache import get_digest_cache_path
from tllama.helpers.model_toml import parse_model_toml, render_model_toml
from tllama.model_definitions import copy_definition
from tllama.repository_check import check_repository
from tllama.repository_names import definition_path
from tllama.repository_prune import prune_repository
from tllama.repository_runtime import origin_path
from tllama.routers import ollama
from test_repository_runtime import hf


@pytest.mark.parametrize('alias_root', [False, True])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('remote_info', [False, True])
def test_draft_pull_reuse_shards_info_and_prune_protection(manager, hf, tmp_path, monkeypatch,
                                                        alias_root, split, remote_info):
    filenames = (['nested/anonymous-00001-of-00002.gguf', 'nested/anonymous-00002-of-00002.gguf']
                 if split else ['nested/anonymous.gguf'])
    for number, name in enumerate(filenames):
        hf.add(name, arch='dflash', shard=number if split else None, count=2 if split else None)
    if alias_root:
        alias = tmp_path / 'models-alias'
        alias.symlink_to(manager.repository.root, target_is_directory=True)
        aliased = ModelManager(BackendConfig(models_dir=str(alias)))
        monkeypatch.setattr(aliased, '_fetch_hf_file_info_sync', manager._fetch_hf_file_info_sync)
        manager = aliased
    path = Path(manager._pull_hf_file_sync('ns/repo', filenames[0]))
    name = 'ns/repo/' + Path(filenames[0]).stem
    target = manager.models_dir / definition_path(name)
    custom = ('# Keep user comment\r\n' + target.read_text().replace('\n', '\r\n')).encode()
    target.write_bytes(custom)
    spec = parse_model_toml(custom.decode())
    assert spec.model_type == 'draft' and set(spec.document) == {'draft'}
    assert spec.draft_model == path.relative_to(manager.repository.root).as_posix()
    assert manager._list_local_models_sync() == []
    assert [row['id'] for row in manager._list_local_models_sync('draft')] == [name]
    members = sorted(path.parent.glob('*.gguf'))
    inodes = [member.stat().st_ino for member in members]
    digest_cache = get_digest_cache_path(manager.metadata_cache_dir, path)
    cached_digest = digest_cache.read_bytes()
    for member in members:
        member.with_suffix('.inf').unlink()
    if not remote_info:
        monkeypatch.setattr(manager, '_fetch_hf_file_info_sync', lambda *args: HfFileLookup(HF_LOOKUP_UNKNOWN))
    progress = PullProgress()
    assert manager._pull_hf_file_sync('ns/repo', filenames[0], progress=progress) == str(path)
    assert target.read_bytes() == custom
    assert [member.stat().st_ino for member in members] == inodes
    assert len(hf.calls) == len(filenames) * (1 if remote_info else 2)
    assert progress.snapshot() == (sum(member.stat().st_size for member in members),) * 2
    assert json.loads(digest_cache.read_bytes())['digest'] == json.loads(cached_digest)['digest']
    for member, filename in zip(members, filenames):
        assert member.with_suffix('.inf').read_text().splitlines() == [Path(filename).name]
    report = check_repository(manager.repository, hashes=True)
    assert report['ok'] and not report['issues']
    assert report['groups'][0]['references'] == [name]
    assert prune_repository(manager.repository, apply=True)['removed_groups'] == []
    manager.delete_model_definition(name)
    assert not path.parent.exists()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('arch, filename, kind', [('dflash', 'anonymous', 'draft'),
                                               ('llama', 'DFlash2-draft', 'main')])
def test_http_pull_uses_architecture_and_show_reports_role(manager, hf, client, monkeypatch, stream,
                                                       arch, filename, kind):
    hf.add(filename + '.gguf', arch=arch)
    monkeypatch.setattr(ollama, 'model_manager', manager)
    name = 'ns/repo/' + filename
    response = client.post('/api/pull', json={'model': name, 'stream': stream})
    assert response.status_code == 200, response.text
    result = json.loads(response.text.splitlines()[-1]) if stream else response.json()
    assert set(result) == {'status', 'path'} and result['status'] == 'success'
    shown = client.post('/api/show', json={'model': name})
    assert shown.status_code == 200, shown.text
    role = 'llm' if kind == 'main' else 'draft'
    assert shown.json()['model_definition'] == {'type': kind, 'artifacts': {
        role: Path(result['path']).relative_to(manager.repository.root).as_posix()}}
    assert shown.json()['model_info']['general.architecture'] == arch
    if kind == 'draft':
        assert shown.json()['capabilities'] == []
        assert client.get('/api/tags').json()['models'] == []
        assert [row['name'] for row in client.get('/api/tags?type=draft').json()['models']] == [name]
        rejected = client.post('/api/generate', json={'model': name, 'prompt': 'hello', 'stream': False})
        assert rejected.status_code == 400 and 'draft helper' in rejected.text
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('keep_alias', [False, True])
def test_changed_draft_updates_owned_binding_but_keeps_shared_old_weights(manager, hf, keep_alias):
    hf.add('draft.gguf', arch='dflash')
    first = Path(manager._pull_hf_file_sync('ns/repo', 'draft.gguf'))
    if keep_alias:
        copy_definition(manager, 'ns/repo/draft', 'snapshot')
    target = manager.models_dir / definition_path('ns/repo/draft')
    target.write_text('# Keep comment\n' + target.read_text())
    hf.files['ns/repo', 'draft.gguf'] += b'changed weights'
    second = Path(manager._pull_hf_file_sync('ns/repo', 'draft.gguf'))
    assert first != second and first.is_file() is keep_alias
    assert '# Keep comment' in target.read_text()
    assert parse_model_toml(target.read_text()).draft_model == second.relative_to(manager.repository.root).as_posix()
    assert parse_model_toml(target.read_text()).runtime == {}


def test_legacy_llm_definition_is_preserved_and_gets_an_independent_draft_alias(manager, hf):
    hf.add('draft.gguf', arch='dflash')
    path = Path(manager._pull_hf_file_sync('ns/repo', 'draft.gguf'))
    target = manager.models_dir / definition_path('ns/repo/draft')
    relative = path.relative_to(manager.repository.root).as_posix()
    legacy = render_model_toml(relative, metadata=manager._read_metadata_for_path(path))
    target.write_text(legacy)
    for _ in range(2):
        assert manager._pull_hf_file_sync('ns/repo', 'draft.gguf') == str(path)
    assert target.read_text() == legacy
    helper = manager.models_dir / definition_path('ns/repo/draft_01')
    assert parse_model_toml(helper.read_text()).draft_model == relative
    assert json.loads((manager.repository.root / origin_path('ns/repo', 'draft.gguf')).read_text())['model'] == 'ns/repo/draft_01'
    assert len(hf.calls) == 1 and len(list(path.parent.parent.iterdir())) == 1


def test_identical_drafts_share_data_but_keep_original_names_and_definitions(manager, hf):
    raw = hf.add('original.gguf', arch='dflash')
    hf.add('renamed.gguf', arch='dflash', repo='other/repo')
    first = Path(manager._pull_hf_file_sync('ns/repo', 'original.gguf'))
    assert manager._pull_hf_file_sync('other/repo', 'renamed.gguf') == str(first)
    assert first.parent.name == sha256(raw).hexdigest()
    assert first.with_suffix('.inf').read_text().splitlines() == ['original.gguf', 'renamed.gguf']
    assert len(manager._list_local_models_sync('draft')) == 2
    manager.delete_model_definition('ns/repo/original')
    assert first.is_file()
    manager.delete_model_definition('other/repo/renamed')
    assert not first.parent.exists()


def test_composite_model_keeps_pulled_draft_after_helper_removal_without_enabling_speculation(manager, hf):
    hf.add('main.gguf')
    hf.add('draft.gguf', arch='dflash')
    manager._pull_hf_file_sync('ns/repo', 'main.gguf')
    draft = Path(manager._pull_hf_file_sync('ns/repo', 'draft.gguf'))
    copy_definition(manager, 'ns/repo/main', 'composite', draft='ns/repo/draft')
    spec = parse_model_toml(manager.read_model_definition('composite'))
    assert spec.draft_model == draft.relative_to(manager.repository.root).as_posix()
    assert 'spec_type' not in spec.runtime
    manager.delete_model_definition('ns/repo/draft')
    assert draft.is_file()
    manager.delete_model_definition('composite')
    assert not draft.parent.exists() and manager.native_snapshot() == []
