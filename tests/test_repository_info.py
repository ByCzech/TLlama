"""Original names remain optional provenance, never extra weights or identity."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_check import check_repository
from tllama.repository_cleanup import collect_pending, QUEUE
from tllama.repository_info import info_path
from tllama.repository_inspect import inspect_repository
from tllama.repository_prune import prune_repository
from tllama.repository_store import RepositoryStore, prepare_group
from test_repository_store import weight, split
from test_repository_cleanup import repo, text_for
from test_repository_runtime import hf
from test_model_create import create_client
from tllama.client_import import create_from_file


def test_original_names_merge_without_new_weights_or_changed_identity(tmp_path):
    source = weight(tmp_path / 'input' / 'Qwen:IQ3%.gguf')
    other = source.with_name('můj-qwen.gguf')
    other.write_bytes(source.read_bytes())
    store = RepositoryStore(tmp_path / 'models')
    [path] = store.publish(prepare_group([source]))
    before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    assert store.publish(prepare_group([other])) == (path,)
    info = info_path(path)
    assert info.read_text().splitlines() == ['Qwen:IQ3%.gguf', 'můj-qwen.gguf']
    assert path.parent.name == sha256(source.read_bytes()).hexdigest()
    assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == before
    assert len(list((store.root / 'data').iterdir())) == 1
    stamp = info.stat().st_mtime_ns
    store.publish(prepare_group([other]))
    assert info.stat().st_mtime_ns == stamp


def test_complete_shards_keep_separate_original_names(tmp_path):
    sources = split(tmp_path / 'input', prefix='Qwen:Q4%')
    paths = RepositoryStore(tmp_path / 'models').publish(prepare_group(sources))
    assert paths[0].parent == paths[1].parent
    for source, path in zip(sources, paths):
        assert info_path(path).read_text() == source.name + '\n'


def test_simultaneous_publishers_do_not_lose_different_original_names(tmp_path):
    sources = [weight(tmp_path / 'input' / f'model-{n}.gguf') for n in range(8)]
    root = tmp_path / 'models'
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda source: RepositoryStore(root).publish(prepare_group([source])), sources))
    assert len({paths[0] for paths in results}) == 1
    assert set(info_path(results[0][0]).read_text().splitlines()) == {source.name for source in sources}


def test_missing_infos_are_silent_and_check_does_not_read_or_hash_companions(repo):
    data = repo.test_publish(count=2)
    create_definition(repo, 'a', text_for(data.relative), {})
    for path in data.paths:
        info_path(path).unlink()
    report = check_repository(repo.repository, hashes=True)
    assert report['ok'] and report['issues'] == []
    copy_definition(repo, 'a', 'copy')
    # Even a large human annotation is not a weight, a hash input, or a size.
    info_path(data.paths[0]).write_text('old name.gguf\n' * 1000)
    report = check_repository(repo.repository, hashes=True)
    assert report['ok'] and report['issues'] == []
    stats = inspect_repository(repo.repository)
    assert stats['summary']['valid_group_weight_bytes'] == sum(p.stat().st_size for p in data.paths)


@pytest.mark.parametrize('which', ['matching-symlink', 'unrelated-info'])
def test_unsafe_or_unrelated_companions_block_cleanup(repo, tmp_path, which):
    data = repo.test_publish()
    outside = tmp_path / 'outside.inf'
    outside.write_text('keep')
    if which == 'matching-symlink':
        info_path(data.paths[0]).unlink()
        info_path(data.paths[0]).symlink_to(outside)
    else:
        (data.paths[0].parent / 'unexpected.inf').write_text('keep')
    assert not check_repository(repo.repository)['ok']
    assert not prune_repository(repo.repository, apply=True)['ok']
    assert data.paths[0].exists() and outside.read_text() == 'keep'


def test_shared_definition_keeps_infos_until_last_rm(repo):
    data = repo.test_publish(count=2)
    create_definition(repo, 'a', text_for(data.relative), {})
    copy_definition(repo, 'a', 'b')
    repo.delete_model_definition('a')
    assert all(info_path(path).exists() for path in data.paths)
    repo.delete_model_definition('b')
    assert not data.paths[0].parent.exists()


def test_prune_lists_and_removes_companions_with_orphan_weights(repo):
    data = repo.test_publish(count=2)
    preview = prune_repository(repo.repository)
    assert preview['ok'] and preview['candidates']
    remove = preview['candidates'][0]['remove_paths']
    assert all(info_path(path).relative_to(repo.models_dir).as_posix() in remove for path in data.paths)
    assert prune_repository(repo.repository, apply=True)['ok']
    assert not data.paths[0].parent.exists()


def test_interrupted_cleanup_can_resume_after_infos_were_removed(repo, monkeypatch):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    unlink = Path.unlink
    def fail(path, *args, **kwargs):
        if path == data.paths[0]:
            raise PermissionError('injected weight unlink failure')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail)
        result = repo.delete_model_definition('a')
        assert result['cleanup']['deferred_groups'] == [data.group]
    assert not info_path(data.paths[0]).exists()
    assert (repo.models_dir / QUEUE).exists()
    assert collect_pending(repo.repository)['removed_groups'] == [data.group]
    assert not data.paths[0].parent.exists()


def test_pull_reuse_recreates_missing_info_and_records_another_origin(manager, hf):
    hf.add('nested/Qwen:Q4%.gguf')
    hf.add('renamed.gguf', repo='other/repo')
    path = Path(manager._pull_hf_file_sync('ns/repo', 'nested/Qwen:Q4%.gguf'))
    info = info_path(path)
    assert info.read_text() == 'Qwen:Q4%.gguf\n'
    info.unlink()
    assert Path(manager._pull_hf_file_sync('ns/repo', 'nested/Qwen:Q4%.gguf')) == path
    assert hf.calls == [('ns/repo', 'nested/Qwen:Q4%.gguf')]
    assert info.read_text() == 'Qwen:Q4%.gguf\n'
    assert Path(manager._pull_hf_file_sync('other/repo', 'renamed.gguf')) == path
    assert info.read_text().splitlines() == ['Qwen:Q4%.gguf', 'renamed.gguf']


async def test_create_reuses_uploaded_bytes_but_records_each_original_name(create_client, manager, tmp_path):
    source = weight(tmp_path / 'client' / 'Qwen:Q4%.gguf')
    other = source.with_name('alias.gguf')
    other.write_bytes(source.read_bytes())
    definitions = []
    for number, model in enumerate((source, other)):
        path = tmp_path / f'create-{number}.toml'
        path.write_text('[llm]\nfrom=' + json.dumps(str(model)) + '\n')
        await create_from_file(create_client, f'model-{number}', path, as_json=True)
        definitions.append(parse_model_toml(manager.read_model_definition(f'model-{number}')))
    assert definitions[0].llm_model == definitions[1].llm_model
    installed = manager.models_dir / definitions[0].llm_model
    assert info_path(installed).read_text().splitlines() == ['Qwen:Q4%.gguf', 'alias.gguf']
    assert len(list((manager.models_dir / 'data').iterdir())) == 1


def test_failed_info_replace_preserves_existing_names_and_weight(tmp_path, monkeypatch):
    source = weight(tmp_path / 'input/model.gguf')
    store = RepositoryStore(tmp_path / 'models')
    [path] = store.publish(prepare_group([source]))
    info = info_path(path)
    before = info.read_bytes(), path.read_bytes(), path.stat().st_ino
    def fail(*args):
        raise OSError('injected info replace failure')
    monkeypatch.setattr('tllama.repository_info.os.replace', fail)
    with pytest.raises(OSError):
        store.record_names([path], ['new.gguf'])
    assert (info.read_bytes(), path.read_bytes(), path.stat().st_ino) == before
    assert not list(path.parent.glob('*.tmp'))
