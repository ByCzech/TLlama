"""Real file publication, cache reuse, failures and cross-process ownership."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import gguf
import pytest

from tllama.helpers import metadata_cache as cache
from tllama.model_identity import file_group_digest
from tllama.repository_store import RepositoryStore, measured_digest, prepare_group


def weight(path, *, index=0, count=1, arch='llama'):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), arch)
    if count != 1:
        writer.add_uint16('split.count', count)
        writer.add_uint16('split.no', index)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


def split(root, prefix='old'):
    return [weight(root / f'{prefix}-{n+1:05d}-of-00002.gguf', index=n, count=2) for n in range(2)]


def test_prepare_is_read_only_and_copy_publication_is_independent(tmp_path):
    source = weight(tmp_path / 'input' / 'original.gguf')
    store = RepositoryStore(tmp_path / 'models')
    group = prepare_group([source], cache_dir=store.cache_dir)
    assert not store.root.exists()
    before = source.stat()
    [target] = store.publish(group)
    assert target == store.root / 'data' / sha256(source.read_bytes()).hexdigest() / 'weights.gguf'
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert not os.path.samefile(source, target)
    source.write_bytes(b'changed original')
    assert target.read_bytes() != source.read_bytes()
    assert not list(store.root.rglob('*.toml'))
    assert not (store.root / '.tllama/repository.json').exists()


def test_shards_are_published_together_with_warm_caches(tmp_path, monkeypatch):
    sources = split(tmp_path / 'inputs')
    store = RepositoryStore(tmp_path / 'models')
    group = prepare_group(sources)
    observed = []
    rename = os.rename
    def publish(source, target):
        assert not Path(target).exists()
        assert {path.name for path in Path(source).iterdir()} == {
            'weights-00001-of-00002.gguf', 'weights-00002-of-00002.gguf',
            'weights-00001-of-00002.inf', 'weights-00002-of-00002.inf'}
        observed.append(Path(target))
        return rename(source, target)
    monkeypatch.setattr('tllama.repository_store.os.rename', publish)
    targets = store.publish(group)
    assert len(observed) == 1 and targets[0].parent == targets[1].parent == observed[0]
    assert observed[0].name == file_group_digest(list(group.digests))
    for target, digest, metadata in zip(targets, group.digests, group.metadata):
        assert cache.load_digest_cache(store.cache_dir, target) == {'content_sha256': digest, 'source': 'local'}
        assert cache.load_metadata_cache(store.cache_dir, target) == metadata
        assert cache.get_digest_cache_path(store.cache_dir, target).stat().st_mtime_ns == target.stat().st_mtime_ns


def test_same_bytes_different_source_names_share_one_group_without_replacement(tmp_path):
    source = weight(tmp_path / 'input' / 'first.gguf')
    other = tmp_path / 'input' / 'renamed.gguf'
    other.write_bytes(source.read_bytes())
    store = RepositoryStore(tmp_path / 'models')
    targets = store.publish(prepare_group([source]))
    inode = targets[0].stat().st_ino
    assert store.publish(prepare_group([other])) == targets
    assert targets[0].stat().st_ino == inode
    assert len(list((store.root / 'data').iterdir())) == 1


def test_owned_links_preserve_inodes_and_reject_external_sources(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    [target] = store.publish(prepare_group([source]), link_owned=True)
    assert source.exists() and os.path.samefile(source, target)
    external = weight(tmp_path / 'external.gguf', arch='other')
    with pytest.raises(ValueError, match='repository-owned'):
        store.publish(prepare_group([external]), link_owned=True)
    assert len(list((store.root / 'data').iterdir())) == 1


def test_hash_named_uploads_use_original_names_only_for_shard_validation(tmp_path):
    sources = split(tmp_path / 'inputs')
    names = [path.name for path in sources]
    renamed = []
    for source in sources:
        target = source.parent / sha256(source.read_bytes()).hexdigest()
        source.rename(target)
        renamed.append(target)
    group = prepare_group(renamed, names=names)
    targets = RepositoryStore(tmp_path / 'models').publish(group)
    assert targets[0].name == 'weights-00001-of-00002.gguf'


@pytest.mark.parametrize('failure', ['missing', 'reversed', 'wrong-index', 'bad-header', 'symlink'])
def test_bad_inputs_cannot_be_prepared_as_a_publishable_group(tmp_path, failure):
    sources = split(tmp_path / 'inputs')
    if failure == 'missing':
        sources = sources[:1]
    elif failure == 'reversed':
        sources.reverse()
    elif failure == 'wrong-index':
        weight(sources[1], index=0, count=2)
    elif failure == 'bad-header':
        sources[1].write_bytes(b'not a GGUF file')
    else:
        original = sources[1].with_suffix('.original')
        sources[1].rename(original)
        sources[1].symlink_to(original)
    with pytest.raises(ValueError):
        prepare_group(sources)
    assert not (tmp_path / 'models').exists()


@pytest.mark.parametrize('failure', ['copy', 'rename', 'changed-source'])
def test_failed_publication_exposes_no_partial_group_and_preserves_sources(tmp_path, monkeypatch, failure):
    sources = split(tmp_path / 'inputs')
    group = prepare_group(sources)
    store = RepositoryStore(tmp_path / 'models')
    def fail(*args, **kwargs):
        raise OSError('injected publication failure')
    if failure == 'copy':
        monkeypatch.setattr('tllama.repository_store.shutil.copystat', fail)
    elif failure == 'rename':
        monkeypatch.setattr('tllama.repository_store.os.rename', fail)
    else:
        sources[1].write_bytes(b'changed after preparation')
    with pytest.raises((OSError, ValueError)):
        store.publish(group)
    assert not (store.root / group.relative_paths[0].parent).exists()
    assert all(path.exists() for path in sources)
    assert not list((store.root / '.tllama/staging').glob('*'))


@pytest.mark.parametrize('failure', ['missing-shard', 'extra-file', 'wrong-bytes', 'symlink'])
def test_existing_broken_group_is_never_replaced(tmp_path, failure):
    group = prepare_group(split(tmp_path / 'inputs'))
    store = RepositoryStore(tmp_path / 'models')
    targets = store.publish(group)
    if failure == 'missing-shard':
        targets[1].unlink()
    elif failure == 'extra-file':
        (targets[0].parent / 'unexpected').touch()
    elif failure == 'wrong-bytes':
        targets[1].write_bytes(b'corrupt stored bytes')
    else:
        targets[1].unlink()
        targets[1].symlink_to(group.sources[1])
    before = targets[0].stat().st_ino
    with pytest.raises(ValueError):
        store.publish(group)
    assert targets[0].stat().st_ino == before


def test_copy_verifies_prepared_digest_before_publication(tmp_path):
    source = weight(tmp_path / 'source.gguf')
    group = replace(prepare_group([source]), digests=('0' * 64,))
    store = RepositoryStore(tmp_path / 'models')
    with pytest.raises(ValueError, match='digest'):
        store.publish(group)
    assert not list((store.root / 'data').iterdir())


@pytest.mark.parametrize('kind', ['data', '.tllama', '.tllama/staging', '.tllama/repository.lock'])
def test_storage_links_cannot_redirect_publication_outside_repository(tmp_path, kind):
    source = weight(tmp_path / 'input.gguf')
    store = RepositoryStore(tmp_path / 'models')
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = store.root / kind
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        store.publish(prepare_group([source]))
    assert not list(outside.iterdir())


def test_repository_root_alias_is_supported(tmp_path):
    root = tmp_path / 'real'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    source = weight(tmp_path / 'input.gguf')
    store = RepositoryStore(alias)
    [target] = store.publish(prepare_group([source]))
    assert target.is_relative_to(root)


def test_local_digest_and_metadata_caches_are_reused_without_hashing(tmp_path, monkeypatch):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    digest = sha256(source.read_bytes()).hexdigest()
    cache.save_digest_cache(store.cache_dir, 'old', source, {'content_sha256': digest, 'source': 'local'})
    cache.save_metadata_cache(store.cache_dir, 'old', source, {'preserved': 'schema-9 payload'})
    def forbidden(*args):
        raise AssertionError('Must reuse a valid locally measured digest')
    monkeypatch.setattr('tllama.repository_store.hash_file', forbidden)
    group = prepare_group([source], cache_dir=store.cache_dir)
    [target] = store.publish(group, link_owned=True)
    assert cache.load_metadata_cache(store.cache_dir, target) == {'preserved': 'schema-9 payload'}
    assert store.publish(prepare_group([source], cache_dir=store.cache_dir), link_owned=True) == (target,)


@pytest.mark.parametrize('source', ['hf', 'unknown', None])
def test_unmeasured_digest_claims_are_rehashed(tmp_path, source):
    store = RepositoryStore(tmp_path / 'models')
    path = weight(store.root / 'Local' / 'old.gguf')
    cache.save_digest_cache(store.cache_dir, 'old', path, {'content_sha256': '0' * 64, 'source': source})
    assert measured_digest(path, store.cache_dir) == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('schema', [0, 1, 999])
def test_old_or_unknown_digest_schemas_are_ignored_without_rewriting_them(tmp_path, schema):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    cache.save_digest_cache(store.cache_dir, 'old', source, {'content_sha256': '0' * 64, 'source': 'local'})
    file = cache.get_digest_cache_path(store.cache_dir, source)
    value = json.loads(file.read_text())
    value['schema_version'] = schema
    file.write_text(json.dumps(value))
    os.utime(file, ns=(source.stat().st_mtime_ns,) * 2)
    old = file.read_bytes()
    group = prepare_group([source], cache_dir=store.cache_dir)
    assert group.digests == (sha256(source.read_bytes()).hexdigest(),)
    assert file.read_bytes() == old


@pytest.mark.parametrize('schema', [0, 8, 999])
def test_old_or_unknown_metadata_is_rebuilt_from_header(tmp_path, schema):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    cache.save_metadata_cache(store.cache_dir, 'old', source, {'wrong': True})
    file = cache.get_metadata_cache_path(store.cache_dir, source)
    value = json.loads(file.read_text())
    value['schema_version'] = schema
    file.write_text(json.dumps(value))
    os.utime(file, ns=(source.stat().st_mtime_ns,) * 2)
    group = prepare_group([source], cache_dir=store.cache_dir)
    assert group.metadata[0]['arch'] == 'llama'
    assert 'wrong' not in group.metadata[0]


def test_cache_and_source_timestamp_changes_invalidate_measured_digest(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    cache.save_digest_cache(store.cache_dir, 'old', source, {'content_sha256': '0' * 64, 'source': 'local'})
    stamp = source.stat().st_mtime_ns + 1
    os.utime(source, ns=(stamp, stamp))
    assert measured_digest(source, store.cache_dir) == sha256(source.read_bytes()).hexdigest()


def test_changed_file_cannot_acquire_a_digest_of_old_bytes(tmp_path, monkeypatch):
    source = weight(tmp_path / 'input.gguf')
    def change(path):
        result = sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b'replaced while hashing')
        return result
    monkeypatch.setattr('tllama.repository_store.hash_file', change)
    with pytest.raises(ValueError, match='changed'):
        measured_digest(source)


def test_lock_is_reentrant_and_bounds_thread_contention(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    result = []
    def contend():
        try:
            with store.lock(timeout=0):
                result.append('acquired')
        except TimeoutError:
            result.append('blocked')
    with store.lock():
        with store.lock(timeout=0):
            thread = threading.Thread(target=contend)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
    assert result == ['blocked']
    with store.lock(timeout=0):
        pass


def test_lock_is_shared_between_processes_and_released_on_exit(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    script = '''
import sys
from tllama.repository_store import RepositoryStore
try:
    with RepositoryStore(sys.argv[1]).lock(timeout=0):
        print('acquired')
except TimeoutError:
    print('blocked')
'''
    def child():
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        return subprocess.run([sys.executable, '-c', script, str(store.root)], env=env,
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    with store.lock():
        assert child() == 'blocked'
    assert child() == 'acquired'


def test_competing_processes_reuse_one_complete_group(tmp_path):
    sources = split(tmp_path / 'inputs')
    store = RepositoryStore(tmp_path / 'models')
    script = '''
import sys
from tllama.repository_store import RepositoryStore, prepare_group
group = prepare_group(sys.argv[2:])
print('ready', flush=True)
sys.stdin.readline()
paths = RepositoryStore(sys.argv[1]).publish(group)
print(paths[0].stat().st_ino)
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    children = []
    try:
        for _ in range(2):
            children.append(subprocess.Popen([sys.executable, '-c', script, str(store.root),
                                              *(str(path) for path in sources)], env=env,
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        # Release both processes before waiting for either result. No unbounded
        # readline or timing-dependent sleep is needed for the barrier.
        for child in children:
            child.stdin.write('go\n')
            child.stdin.flush()
        results = [child.communicate(timeout=10) for child in children]
        assert all(child.returncode == 0 for child in children), results
        assert all(output.splitlines()[0] == 'ready' for output, _ in results)
        assert len({output.splitlines()[1] for output, _ in results}) == 1
        [directory] = list((store.root / 'data').iterdir())
        assert len(list(directory.glob('*.gguf'))) == 2
        assert not list((store.root / '.tllama/staging').iterdir())
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_cache_format_from_older_releases_is_not_upgraded_in_place(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    cache.save_digest_cache(store.cache_dir, 'old', source, {'content_sha256': '0' * 64, 'source': 'local'})
    file = cache.get_digest_cache_path(store.cache_dir, source)
    value = json.loads(file.read_text())
    value.pop('cache_format_version')
    value['model']['path'] = str(source)
    value['model']['mtime_ns'] = source.stat().st_mtime_ns
    file.write_text(json.dumps(value))
    os.utime(file, ns=(source.stat().st_mtime_ns,) * 2)
    before = file.read_bytes()
    group = prepare_group([source], cache_dir=store.cache_dir)
    assert group.digests == (sha256(source.read_bytes()).hexdigest(),)
    assert file.read_bytes() == before


def test_invalid_cache_json_is_a_miss_not_a_failed_conversion(tmp_path):
    store = RepositoryStore(tmp_path / 'models')
    source = weight(store.root / 'Local' / 'old.gguf')
    store.cache_dir.mkdir(parents=True)
    cache.get_digest_cache_path(store.cache_dir, source).write_text('broken JSON')
    cache.get_metadata_cache_path(store.cache_dir, source).write_text('[]')
    group = prepare_group([source], cache_dir=store.cache_dir)
    assert group.digests == (sha256(source.read_bytes()).hexdigest(),)
    assert group.metadata[0]['arch'] == 'llama'


@pytest.mark.parametrize('timeout', [-1, float('nan'), float('inf'), True])
def test_invalid_lock_timeout_has_no_side_effects(tmp_path, timeout):
    store = RepositoryStore(tmp_path / 'models')
    with pytest.raises(ValueError):
        with store.lock(timeout=timeout):
            pass
    assert not store.root.exists()
