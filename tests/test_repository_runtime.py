"""The active runtime uses one explicit format and complete shared data groups."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from tllama.backend import ModelManager, HfFileLookup, HF_LOOKUP_FOUND, HF_LOOKUP_MISSING, PullProgress
from tllama.config import BackendConfig, ConfigError
from tllama.helpers import metadata_cache
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_definitions import create_definition, copy_definition
from tllama.repository_layout import REPOSITORY_FORMAT
from tllama.repository_names import definition_path
from tllama.repository_runtime import origin_path, MARKER
from tllama.repository_store import prepare_group
from test_model_create import gguf_file


def publish(manager, tmp_path, *, arch='llama'):
    source = gguf_file(tmp_path / (arch + '.gguf'), arch=arch)
    target, = manager.repository.publish(prepare_group([source]))
    return target.relative_to(manager.models_dir).as_posix()


def text_for(relative):
    return '# User definition\n[llm]\nmodel=' + json.dumps(relative) + '\n'


@pytest.fixture
def hf(manager, monkeypatch, tmp_path):
    files, calls, lookups = {}, [], []

    def add(filename, *, arch='llama', shard=None, count=None, repo='ns/repo'):
        source = gguf_file(tmp_path / 'remote' / repo / filename, arch=arch, shard=shard, count=count)
        files[repo, filename] = source.read_bytes()
        return files[repo, filename]

    def download(*, repo_id, filename, revision, token, local_dir, tqdm_class):
        calls.append((repo_id, filename))
        raw = files[repo_id, filename]
        if isinstance(raw, Exception):
            raise raw
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if tqdm_class is not None:
            with tqdm_class(total=len(raw)) as bar:
                bar.update(len(raw))
        return str(target)

    def lookup(repo_id, filename, token=None, revision=None):
        lookups.append((repo_id, filename))
        raw = files.get((repo_id, filename))
        if not isinstance(raw, bytes):
            return HfFileLookup(HF_LOOKUP_MISSING)
        return HfFileLookup(HF_LOOKUP_FOUND, sha256(raw).hexdigest(), len(raw))

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(manager, '_fetch_hf_file_info_sync', lookup)
    return SimpleNamespace(add=add, files=files, calls=calls, lookups=lookups, download=download)


def test_empty_repository_is_initialized_explicitly_and_idempotently(tmp_path):
    manager = ModelManager(BackendConfig(models_dir=str(tmp_path / 'new-store')))
    assert not manager.models_dir.exists()
    manager.ensure_storage()
    assert json.loads((manager.models_dir / MARKER).read_text()) == REPOSITORY_FORMAT
    before = (manager.models_dir / MARKER).stat()
    manager.ensure_storage()
    assert (manager.models_dir / MARKER).stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize('filename', ['Local/old.toml', 'Local/old.gguf', 'data/unknown/weights.gguf'])
def test_unmarked_nonempty_repository_requires_offline_conversion(tmp_path, filename):
    root = tmp_path / 'legacy'
    file = root / filename
    file.parent.mkdir(parents=True)
    file.write_bytes(b'keep original')
    manager = ModelManager(BackendConfig(models_dir=str(root)))
    with pytest.raises(ConfigError, match='migrate-repository-v2.py'):
        manager.ensure_storage()
    assert file.read_bytes() == b'keep original'
    assert not (root / MARKER).exists()


@pytest.mark.parametrize('marker', ['not JSON', '{"version":999}', 'null'])
def test_unknown_or_broken_marker_is_never_reinterpreted(manager, marker):
    (manager.models_dir / MARKER).write_text(marker)
    with pytest.raises(ConfigError):
        manager.ensure_storage()
    assert (manager.models_dir / MARKER).read_text() == marker


@pytest.mark.parametrize('name', ['NUL', 'Český:Q4%', 'ByCzech/Model:Q4%', 'ns/repo/model:Q4%'])
async def test_create_copy_read_list_identity_and_delete_use_the_same_codec(manager, tmp_path, name):
    relative = publish(manager, tmp_path)
    text = text_for(relative)
    create_definition(manager, name, text, {})
    physical = manager.models_dir / definition_path(name)
    assert physical.read_text() == text
    assert ':' not in physical.name and '%' in physical.name
    assert manager.read_model_definition(name) == text
    rows = await manager.list_local_models_with_metadata()
    assert {row['id'] for row in rows} == {name}
    before = manager._listing_identity(name)
    copy_definition(manager, name, 'Copied:100%')
    assert manager._listing_identity('Copied:100%') == before
    manager.delete_model_definition(name)
    assert not physical.exists()
    assert manager.read_model_definition('Copied:100%') == text
    assert (manager.models_dir / relative).exists()


@pytest.mark.parametrize('names', [('ByCzech/model', 'byczech/other'), ('Žena/model', 'Z\u030cena/other')])
def test_portable_parent_collisions_are_rejected_even_on_linux(manager, tmp_path, names):
    text = text_for(publish(manager, tmp_path))
    create_definition(manager, names[0], text, {})
    with pytest.raises(ValueError, match='collision'):
        create_definition(manager, names[1], text, {})
    assert len(list(manager.models_dir.rglob('*.toml'))) == 1


def test_two_managers_serialize_portable_name_checks(manager, tmp_path):
    text = text_for(publish(manager, tmp_path))
    other = ModelManager(manager.config)
    barrier = threading.Barrier(2)
    def write(job):
        instance, name = job
        barrier.wait(timeout=5)
        try:
            create_definition(instance, name, text, {})
            return 'created'
        except ValueError as exc:
            assert 'collision' in str(exc)
            return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, [(manager, 'A/model'), (other, 'a/other')])) == ['created', 'rejected']
    assert len(list(manager.models_dir.rglob('*.toml'))) == 1


def test_delete_cannot_follow_a_definition_parent_symlink(manager, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    victim = outside / 'model.toml'
    victim.write_text('keep this')
    (manager.tllama_models_dir / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        manager.delete_model_definition('escape/model')
    assert victim.read_text() == 'keep this'


def test_new_definition_cannot_point_to_arbitrary_legacy_weight(manager, tmp_path):
    gguf_file(manager.local_models_dir / 'old.gguf')
    with pytest.raises(ValueError, match='managed data'):
        create_definition(manager, 'bad', text_for('Local/old.gguf'), {})
    assert not list(manager.models_dir.rglob('*.toml'))


@pytest.mark.parametrize('broken', ['wrong-hash', 'extra-file', 'symlink'])
def test_new_definition_cannot_publish_invalid_data_reference(manager, tmp_path, broken):
    relative = publish(manager, tmp_path)
    target = manager.models_dir / relative
    if broken == 'wrong-hash':
        replacement = target.parent.with_name('0' * 64)
        target.parent.rename(replacement)
        relative = (replacement / target.name).relative_to(manager.models_dir).as_posix()
    elif broken == 'extra-file':
        (target.parent / 'unexpected').write_text('unknown')
    else:
        other = tmp_path / 'moved'
        target.parent.rename(other)
        target.parent.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        create_definition(manager, 'bad', text_for(relative), {})
    assert not list(manager.models_dir.rglob('*.toml'))


def test_pull_uses_private_staging_and_warms_final_caches(manager, hf, monkeypatch):
    raw = hf.add('nested/model:Q4%.gguf')
    target = Path(manager._pull_hf_file_sync('ns/repo', 'nested/model:Q4%.gguf', token='do-not-store'))
    assert target == manager.models_dir / 'data' / sha256(raw).hexdigest() / 'weights.gguf'
    definition = manager.models_dir / definition_path('ns/repo/model:Q4%')
    assert parse_model_toml(definition.read_text()).llm_model == target.relative_to(manager.models_dir).as_posix()
    assert not list((manager.models_dir / '.tllama/downloads').iterdir())
    assert not list(manager.hf_models_dir.rglob('*.gguf'))
    assert 'do-not-store' not in (manager.models_dir / origin_path('ns/repo', 'nested/model:Q4%.gguf')).read_text()
    def fail(*args, **kwargs):
        raise AssertionError('Published paths must already have measured caches')
    monkeypatch.setattr(manager, '_compute_content_sha256', fail)
    monkeypatch.setattr(manager, '_get_model_metadata_sync', fail)
    assert manager._listing_identity('ns/repo/model:Q4%')[0] == sha256(raw).hexdigest()
    assert metadata_cache.load_digest_cache(manager.metadata_cache_dir, target)['source'] == 'local'


def test_repeat_pull_reuses_measured_group_and_preserves_edited_definition(manager, hf):
    hf.add('model.gguf')
    path = manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    definition = manager.models_dir / definition_path('ns/repo/model')
    custom = definition.read_text() + '# User customization\n[system]\nprompt="Czech"\n'
    definition.write_text(custom)
    inode = Path(path).stat().st_ino
    progress = PullProgress()
    assert manager._pull_hf_file_sync('ns/repo', 'model.gguf', progress=progress) == path
    assert definition.read_text() == custom and Path(path).stat().st_ino == inode
    assert hf.calls == [('ns/repo', 'model.gguf')]
    assert progress.snapshot() == (Path(path).stat().st_size,) * 2


@pytest.mark.parametrize('root_kind', ['direct', 'symlink', 'parent-symlink', 'relative'])
@pytest.mark.parametrize('arch', ['llama', 'clip'])
@pytest.mark.parametrize('split', [False, True])
def test_repeat_pull_uses_store_root_for_every_member(manager, hf, monkeypatch, tmp_path,
                                                     root_kind, arch, split):
    root = manager.repository.root
    configured = root
    if root_kind == 'symlink':
        configured = tmp_path / 'models-alias'
        configured.symlink_to(root, target_is_directory=True)
    elif root_kind == 'parent-symlink':
        parent = tmp_path / 'parent-alias'
        parent.symlink_to(root.parent, target_is_directory=True)
        configured = parent / root.name
    elif root_kind == 'relative':
        monkeypatch.chdir(root.parent)
        configured = Path(root.name)
    aliased = ModelManager(BackendConfig(models_dir=str(configured)))
    aliased.ensure_storage()
    monkeypatch.setattr(aliased, '_fetch_hf_file_info_sync', manager._fetch_hf_file_info_sync)
    names = (['nested/model-00001-of-00002.gguf', 'nested/model-00002-of-00002.gguf']
             if split else ['nested/model.gguf'])
    for number, filename in enumerate(names):
        hf.add(filename, arch=arch, shard=number if split else None, count=2 if split else None)
    first = Path(aliased._pull_hf_file_sync('ns/repo', names[0]))
    members = sorted(first.parent.glob('*.gguf'))
    signatures = {path.name: (path.stat().st_ino, path.read_bytes()) for path in members}
    definition = root / definition_path('ns/repo/' + Path(names[0]).stem)
    custom = None
    if arch == 'llama':
        custom = definition.read_text() + '# Preserve user settings\n[system]\nprompt="Czech"\n'
        definition.write_text(custom)
    origin = root / origin_path('ns/repo', names[0])
    before = json.loads(origin.read_text())
    progress = PullProgress()
    second = Path(aliased._pull_hf_file_sync('ns/repo', names[0], progress=progress))
    assert first == second and first.is_relative_to(aliased.models_dir)
    assert aliased.models_dir == aliased.repository.root == configured.absolute()
    assert first.resolve().is_relative_to(root)
    assert {path.name: (path.stat().st_ino, path.read_bytes()) for path in members} == signatures
    assert hf.calls == [('ns/repo', name) for name in names]
    assert hf.lookups == [('ns/repo', name) for name in names]
    assert json.loads(origin.read_text()) == before
    assert before['path'] == first.relative_to(aliased.models_dir).as_posix()
    assert progress.snapshot() == (sum(path.stat().st_size for path in members),) * 2
    assert len(list((root / 'data').iterdir())) == 1
    for path, name in zip(members, names):
        assert path.with_suffix('.inf').read_text().splitlines() == [Path(name).name]
    if arch == 'llama':
        assert definition.read_text() == custom
        assert parse_model_toml(custom).llm_model == before['path']
        copy_definition(aliased, 'ns/repo/' + Path(names[0]).stem, 'copy')
        assert aliased.read_model_definition('copy') == custom
    else:
        spec = parse_model_toml(definition.read_text())
        assert spec.model_type == 'mmproj' and spec.mmproj_model == before['path']
        assert before['model'] == 'ns/repo/' + Path(names[0]).stem
        assert not aliased._list_local_models_sync()
    assert not list((root / '.tllama/downloads').iterdir())


@pytest.mark.parametrize('component', ['data', 'group', 'weight'])
def test_alias_root_does_not_allow_symlinks_inside_managed_data(manager, tmp_path, component):
    import shutil

    relative = publish(manager, tmp_path)
    original = manager.repository.root / relative
    content = original.read_bytes()
    target = {'data': original.parent.parent, 'group': original.parent, 'weight': original}[component]
    outside = tmp_path / 'outside-managed-data'
    shutil.move(str(target), str(outside))
    target.symlink_to(outside, target_is_directory=component != 'weight')
    alias = tmp_path / 'models-alias'
    alias.symlink_to(manager.repository.root, target_is_directory=True)
    aliased = ModelManager(BackendConfig(models_dir=str(alias)))
    with pytest.raises(ValueError, match='symlink'):
        create_definition(aliased, 'bad', text_for(relative), {})
    assert original.read_bytes() == content
    assert not list(manager.repository.root.rglob('*.toml'))


def test_changed_source_updates_only_its_own_definition_pointer(manager, hf):
    hf.add('model.gguf')
    old = manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    copy_definition(manager, 'ns/repo/model', 'snapshot')
    definition = manager.models_dir / definition_path('ns/repo/model')
    definition.write_text(definition.read_text() + '# Keep this\n[system]\nprompt="Czech"\n')
    hf.add('model.gguf', arch='qwen3')
    new = manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    spec = parse_model_toml(definition.read_text())
    assert old != new and spec.system_prompt == 'Czech' and '# Keep this' in definition.read_text()
    assert spec.llm_model == Path(new).relative_to(manager.models_dir).as_posix()
    assert parse_model_toml(manager.read_model_definition('snapshot')).llm_model == Path(old).relative_to(manager.models_dir).as_posix()
    assert Path(old).exists()  # The snapshot alias still references the old group.


def test_pull_preserves_a_definition_that_the_user_repointed(manager, hf, tmp_path):
    hf.add('model.gguf')
    manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    definition = manager.models_dir / definition_path('ns/repo/model')
    custom = text_for(publish(manager, tmp_path, arch='qwen3'))
    definition.write_text(custom)
    manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    assert definition.read_text() == custom
    assert (manager.models_dir / definition_path('ns/repo/model_01')).is_file()
    assert len(hf.calls) == 1


@pytest.mark.parametrize('bad_record', ['{not JSON', '{"filename": "different"}'])
def test_broken_origin_record_cannot_stop_safe_republication(manager, hf, bad_record):
    hf.add('model.gguf')
    first = manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    (manager.models_dir / origin_path('ns/repo', 'model.gguf')).write_text(bad_record)
    assert manager._pull_hf_file_sync('ns/repo', 'model.gguf') == first
    assert len(list(manager.models_dir.rglob('*.toml'))) == 1


def test_nested_basename_collision_stays_stable_across_repeated_pulls(manager, hf):
    hf.add('a/model.gguf')
    hf.add('b/model.gguf', arch='qwen3')
    first = manager._pull_hf_file_sync('ns/repo', 'a/model.gguf')
    second = manager._pull_hf_file_sync('ns/repo', 'b/model.gguf')
    assert first != second
    assert manager._pull_hf_file_sync('ns/repo', 'b/model.gguf') == second
    assert len(hf.calls) == 2
    assert {row['id'] for row in manager._list_local_models_sync()} == {'ns/repo/model', 'ns/repo/model_01'}


def test_pulls_from_different_origins_share_identical_weights(manager, hf):
    hf.add('model.gguf')
    hf.add('alias.gguf', repo='other/repo')
    first = manager._pull_hf_file_sync('ns/repo', 'model.gguf')
    second = manager._pull_hf_file_sync('other/repo', 'alias.gguf')
    assert first == second
    assert len(list((manager.models_dir / 'data').iterdir())) == 1
    assert len(list(manager.models_dir.rglob('*.toml'))) == 2


def test_projector_pull_registers_a_helper_without_becoming_a_main_model(manager, hf):
    raw = hf.add('mmproj.gguf', arch='clip')
    target = manager._pull_hf_file_sync('ns/repo', 'mmproj.gguf')
    assert Path(target).read_bytes() == raw
    assert manager._list_local_models_sync() == []
    definition = manager.models_dir / definition_path('ns/repo/mmproj')
    spec = parse_model_toml(definition.read_text())
    assert spec.model_type == 'mmproj'
    assert spec.mmproj_model == Path(target).relative_to(manager.repository.root).as_posix()
    assert [row['id'] for row in manager._list_local_models_sync('mmproj')] == ['ns/repo/mmproj']


def test_pull_completes_shards_and_rechecks_every_remote_member(manager, hf):
    first = hf.add('split/model-00001-of-00002.gguf', shard=0, count=2)
    second = hf.add('split/model-00002-of-00002.gguf', shard=1, count=2)
    progress = PullProgress()
    path = Path(manager._pull_hf_file_sync('ns/repo', 'split/model-00001-of-00002.gguf', progress=progress))
    assert sorted(p.name for p in path.parent.glob('*.gguf')) == ['weights-00001-of-00002.gguf', 'weights-00002-of-00002.gguf']
    assert progress.snapshot() == (len(first) + len(second),) * 2
    assert len(hf.calls) == 2
    assert manager._pull_hf_file_sync('ns/repo', 'split/model-00001-of-00002.gguf') == str(path)
    assert len(hf.calls) == 2 and len(hf.lookups) == 2
    hf.add('split/model-00002-of-00002.gguf', shard=1, count=2, arch='qwen3')
    changed = manager._pull_hf_file_sync('ns/repo', 'split/model-00001-of-00002.gguf')
    assert changed != str(path) and len(hf.calls) == 4


@pytest.mark.parametrize('failure', ['network', 'bad-header', 'continuation'])
def test_failed_pull_never_publishes_partial_data_or_definition(manager, hf, failure):
    filename = 'model.gguf'
    if failure == 'network':
        filename = 'model-00001-of-00002.gguf'
        hf.add(filename, shard=0, count=2)
        hf.files['ns/repo', 'model-00002-of-00002.gguf'] = OSError('network failed')
    elif failure == 'bad-header':
        hf.files['ns/repo', filename] = b'not GGUF'
    else:
        filename = 'model-00002-of-00002.gguf'
        hf.add(filename, shard=1, count=2)
    with pytest.raises(Exception):
        manager._pull_hf_file_sync('ns/repo', filename)
    assert not list((manager.models_dir / 'data').iterdir())
    assert not list(manager.models_dir.rglob('*.toml'))
    assert not list((manager.models_dir / '.tllama/downloads').iterdir())


async def test_cancelled_pull_joins_owned_publication_before_return(manager, hf, monkeypatch):
    hf.add('model.gguf')
    entered, release = threading.Event(), threading.Event()
    def blocked(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return hf.download(**kwargs)
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'hf_hub_download', blocked)
    task = asyncio.create_task(manager.pull_hf_file('ns/repo', 'model.gguf'))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(list(manager.models_dir.rglob('*.toml'))) == 1
    assert not list((manager.models_dir / '.tllama/downloads').iterdir())
