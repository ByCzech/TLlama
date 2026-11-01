"""Offline conversion preserves models, handles legacy caches and resumes I/O."""

import base64
import errno
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.backend import ModelManager
from tllama.helpers import metadata_cache as cache
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_identity import file_group_digest
from tllama.repository_layout import REPOSITORY_FORMAT
from tllama.repository_names import definition_name
from tllama.repository_store import RepositoryStore
from test_repository_store import weight, split

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/migrate-repository-v2.py'
_spec = importlib.util.spec_from_file_location('migration_tool', SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def define(root, name='model', target='Local/old.gguf', extra='', area='Local'):
    path = root / area / (name + '.toml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# Keep this comment\n[llm]\nmodel=' + json.dumps(target) + ' # path comment\n' + extra,
                    encoding='utf-8')
    return path


def basic(tmp_path):
    root = tmp_path / 'models'
    source = weight(root / 'Local' / 'old.gguf')
    definition = define(root)
    return root, source, definition


@pytest.mark.parametrize('interrupted', [False, True])
def test_conversion_finishes_all_empty_runtime_directories_before_marker(tmp_path, monkeypatch, interrupted):
    from tllama.repository_check import check_repository
    from tllama.repository_inspect import inspect_repository
    root, _, _ = basic(tmp_path)
    if interrupted:
        original = tool.ensure_directory
        def fail_pins(root, relative, **kwargs):
            if relative == '.tllama/pins':
                raise PermissionError('injected final-directory failure')
            return original(root, relative, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(tool, 'ensure_directory', fail_pins)
            with pytest.raises(PermissionError):
                tool.apply_plan(root, tool.build_plan(root))
        assert not (root / tool.MARKER).exists()
        tool.apply_plan(root, resume=True)
    else:
        tool.apply_plan(root, tool.build_plan(root))
    for relative in ('Local', 'TLlama', 'HuggingFace', '.tllama/pins'):
        assert (root / relative).is_dir()
    before = snapshot(root)
    assert check_repository(RepositoryStore(root))['ok']
    assert inspect_repository(RepositoryStore(root))['ok']
    assert snapshot(root) == before


def snapshot(root):
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


def test_conversion_records_all_original_names_without_splitting_identical_content(tmp_path):
    root, source, _ = basic(tmp_path)
    alias = root / 'jiný:Q4%.gguf'
    alias.write_bytes(source.read_bytes())
    plan = tool.build_plan(root)
    result = tool.apply_plan(root, plan)
    assert result['data_groups'] == 1
    target = root / plan['mapping']['Local/old.gguf']
    assert target == root / plan['mapping'][alias.name]
    assert target.with_suffix('.inf').read_text().splitlines() == ['old.gguf', 'jiný:Q4%.gguf']
    assert target.parent.name == result['definitions'][0]['id']


def test_original_name_with_newline_is_rejected_in_read_only_preflight(tmp_path):
    root, _, _ = basic(tmp_path)
    weight(root / 'invalid\nname.gguf', arch='other')
    before = snapshot(root)
    with pytest.raises(ValueError, match='single-line basenames'):
        tool.build_plan(root)
    assert snapshot(root) == before
    assert not (root / '.tllama').exists()


def test_conversion_records_the_original_name_of_each_shard_and_lora(tmp_path):
    root = tmp_path / 'models'
    sources = split(root / 'Local', prefix='Qwen:IQ3%')
    adapter = weight(root / 'Local' / 'adapter.bin', arch='adapter')
    define(root, target=sources[0].relative_to(root).as_posix(), extra='[runtime]\nlora_path="Local/adapter.bin"\n')
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    for source in [*sources, adapter]:
        target = root / plan['mapping'][source.relative_to(root).as_posix()]
        assert target.with_suffix('.inf').read_text() == source.name + '\n'


def test_resume_recovers_original_names_after_cleanup_removed_all_sources(tmp_path, monkeypatch):
    root, source, _ = basic(tmp_path)
    atomic = tool.atomic_write
    def interrupt(root, relative, raw, **kwargs):
        if relative == tool.MARKER:
            raise OSError('interrupted before marker')
        return atomic(root, relative, raw, **kwargs)
    plan = tool.build_plan(root)
    with monkeypatch.context() as patch:
        patch.setattr(tool, 'atomic_write', interrupt)
        with pytest.raises(OSError, match='interrupted before marker'):
            tool.apply_plan(root, plan)
    assert not source.exists()
    # This represents an older converter with the same journal contract.
    for companion in (root / 'data').rglob('*.inf'):
        companion.unlink()
    assert tool.apply_plan(root, resume=True)['status'] == 'converted'
    target = root / plan['mapping']['Local/old.gguf']
    assert target.with_suffix('.inf').read_text() == 'old.gguf\n'


def test_empty_orphan_error_identifies_the_file_not_the_previous_progress_line(tmp_path, capsys):
    root, _, _ = basic(tmp_path)
    empty = root / 'orphan-empty.gguf'
    empty.touch()
    before = snapshot(root)
    assert tool.main([str(root)]) == 1
    output = capsys.readouterr()
    assert 'Inspecting: orphan-empty.gguf' in output.err
    assert 'Cannot inspect GGUF orphan-empty.gguf:' in output.err
    assert snapshot(root) == before


def test_empty_continuation_error_identifies_the_actual_shard(tmp_path, capsys):
    root = tmp_path / 'models'
    sources = split(root / 'Local')
    sources[1].write_bytes(b'')
    define(root, target=sources[0].relative_to(root).as_posix())
    before = snapshot(root)
    assert tool.main([str(root)]) == 1
    assert str(sources[1]) in capsys.readouterr().err
    assert snapshot(root) == before


def test_preview_is_read_only_and_reports_real_content_identity(tmp_path, capsys):
    root, source, _ = basic(tmp_path)
    before = snapshot(root)
    assert tool.main([str(root), '--json']) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report['status'] == 'planned' and report['data_groups'] == 1
    assert report['definitions'][0]['id'] == sha256(source.read_bytes()).hexdigest()
    assert output.err == '' and snapshot(root) == before
    assert not (root / '.tllama').exists()


def test_apply_moves_weights_keeps_inode_comments_modes_times_and_ids(tmp_path):
    root, source, definition = basic(tmp_path)
    before = source.stat()
    definition.chmod(0o640)
    stamp = 1_700_000_000_123_456_789
    os.utime(definition, ns=(stamp, stamp))
    original = definition.read_bytes()
    plan = tool.build_plan(root)
    result = tool.apply_plan(root, plan)
    assert result['status'] == 'converted'
    target = root / plan['mapping']['Local/old.gguf']
    assert not source.exists() and target.stat().st_ino == before.st_ino
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    text = definition.read_text()
    assert '# Keep this comment' in text and '# path comment' in text
    assert parse_model_toml(text).llm_model == target.relative_to(root).as_posix()
    assert stat.S_IMODE(definition.stat().st_mode) == 0o640
    assert definition.stat().st_mtime_ns == stamp
    assert (root / tool.CONTROL / 'backups/000000.toml').read_bytes() == original
    assert stat.S_IMODE((root / tool.JOURNAL).stat().st_mode) == 0o600
    assert json.loads((root / tool.MARKER).read_text()) == REPOSITORY_FORMAT
    assert result['definitions'][0]['id'] == target.parent.name


@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='Requires POSIX root ownership control')
def test_administrative_conversion_preserves_definition_owner_and_group(tmp_path):
    root, _, definition = basic(tmp_path)
    try:
        os.chown(definition, 65534, 65534)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.EPERM, errno.EACCES}:
            pytest.skip('Filesystem/user namespace cannot represent another POSIX owner')
        raise
    definition.chmod(0o640)
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    assert (definition.stat().st_uid, definition.stat().st_gid) == (65534, 65534)
    assert stat.S_IMODE(definition.stat().st_mode) == 0o640


def test_atomic_definition_write_sets_owner_before_restoring_access_bits(tmp_path, monkeypatch):
    if not hasattr(os, 'fchown'):
        pytest.skip('POSIX ownership API unavailable')
    root = tmp_path / 'models'
    root.mkdir()
    calls = []
    chmod = tool.os.chmod
    def owner(fd, uid, gid):
        calls.append(('owner', uid, gid))
    def mode(path, value):
        calls.append(('mode', value))
        return chmod(path, value)
    monkeypatch.setattr(tool.os, 'fchown', owner)
    monkeypatch.setattr(tool.os, 'chmod', mode)
    tool.atomic_write(root, 'Local/model.toml', b'private content', mode=0o640, owner=[65534, 65534])
    assert calls == [('owner', 65534, 65534), ('mode', 0o640)]
    assert stat.S_IMODE((root / 'Local/model.toml').stat().st_mode) == 0o640


def test_deduplication_does_not_silently_change_access_permissions(tmp_path):
    root, source, _ = basic(tmp_path)
    other = root / 'other.gguf'
    other.write_bytes(source.read_bytes())
    other.chmod(0o600)
    source.chmod(0o644)
    with pytest.raises(ValueError, match='different owners/access bits'):
        tool.build_plan(root)


def test_aliases_projector_lora_and_orphans_keep_independent_shared_groups(tmp_path):
    root, main, _ = basic(tmp_path)
    projector = weight(root / 'HuggingFace/ns/repo/projector.gguf', arch='clip')
    adapter = weight(root / 'TLlama/adapters/adapter.bin', arch='llama-adapter')
    orphan = weight(root / 'misc' / 'orphan.gguf', arch='orphan')
    define(root, 'vision', extra='[mmproj]\nmodel="HuggingFace/ns/repo/projector.gguf"\n'
           '[runtime]\nlora_path="TLlama/adapters/adapter.bin"\n')
    plan = tool.build_plan(root)
    assert len(plan['definitions']) == 2
    result = tool.apply_plan(root, plan)
    assert result['data_groups'] == 4
    specs = [parse_model_toml(path.read_text()) for path in (root / 'Local').glob('*.toml')]
    assert len({spec.llm_model for spec in specs}) == 1
    vision = next(spec for spec in specs if spec.mmproj_model)
    assert len({Path(vision.llm_model).parent, Path(vision.mmproj_model).parent,
                Path(vision.runtime['lora_path']).parent}) == 3
    assert len(list(root.rglob('*.toml'))) == 4  # Two live definitions plus private backups.
    assert not orphan.exists() and not adapter.exists() and not projector.exists() and not main.exists()


def test_split_sets_stay_together_and_preserve_group_digest(tmp_path):
    root = tmp_path / 'models'
    sources = split(root / 'HuggingFace/ns/repo/deep')
    define(root, target=sources[0].relative_to(root).as_posix())
    expected = file_group_digest([sha256(path.read_bytes()).hexdigest() for path in sources])
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    [directory] = list((root / 'data').iterdir())
    assert directory.name == expected
    assert {path.name for path in directory.glob('*.gguf')} == {
        'weights-00001-of-00002.gguf', 'weights-00002-of-00002.gguf'}
    assert plan['definitions'][0]['id'] == expected


def test_identical_copies_are_deduplicated_but_not_registered_as_new_models(tmp_path):
    root, source, _ = basic(tmp_path)
    other = root / 'other' / 'copied.gguf'
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())
    plan = tool.build_plan(root)
    result = tool.apply_plan(root, plan)
    assert result['data_groups'] == 1 and result['weight_files'] == 2
    assert len(result['definitions']) == 1
    assert len(list((root / 'data').rglob('*.gguf'))) == 1
    assert not source.exists() and not other.exists()


def test_whole_repository_symlink_is_supported(tmp_path):
    root, source, _ = basic(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    result = tool.apply_plan(alias, tool.build_plan(alias))
    assert result['status'] == 'converted' and not source.exists()


def test_uppercase_legacy_definition_extension_is_canonicalized(tmp_path):
    root, _, definition = basic(tmp_path)
    definition.rename(definition.with_suffix('.TOML'))
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    assert definition.is_file() and not definition.with_suffix('.TOML').exists()


def test_duplicate_canonical_definition_target_is_not_overwritten(tmp_path):
    root, _, definition = basic(tmp_path)
    definition.with_suffix('.TOML').write_bytes(definition.read_bytes())
    with pytest.raises(ValueError, match='same canonical target'):
        tool.build_plan(root)


def test_literal_percent_and_encoded_target_overlap_do_not_overwrite_definitions(tmp_path):
    root, _, original = basic(tmp_path)
    original.unlink()
    first = define(root, 'model:55b', extra='[system]\nprompt="first"\n')
    second = define(root, 'model%3A55b', extra='[system]\nprompt="second"\n')
    old = {path.name: path.read_bytes() for path in (first, second)}
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    models = {definition_name(path.relative_to(root).as_posix()): parse_model_toml(path.read_text()).system_prompt
              for path in (root / 'Local').glob('*.toml')}
    assert models == {'model:55b': 'first', 'model%3A55b': 'second'}
    backups = {path.read_bytes() for path in (root / tool.CONTROL / 'backups').iterdir()}
    assert backups == set(old.values())


def test_namespace_names_are_encoded_and_old_empty_directories_removed(tmp_path):
    root = tmp_path / 'models'
    weight(root / 'Local/old.gguf')
    define(root, 'CON:tag', area='HuggingFace/author./repo?')
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    target = root / 'HuggingFace/author%2E/repo%3F/CON%3Atag.toml'
    assert target.is_file()
    assert not (root / 'HuggingFace/author.').exists()


def test_pending_from_is_preserved_and_not_imported_from_outside_repo(tmp_path):
    root = tmp_path / 'models'
    path = root / 'Local' / 'pending.toml'
    path.parent.mkdir(parents=True)
    original = b'[llm]\nfrom="/not/a/server/import.gguf"\n'
    path.write_bytes(original)
    plan = tool.build_plan(root)
    assert plan['definitions'][0]['id'] is None and not plan['groups']
    tool.apply_plan(root, plan)
    assert path.read_bytes() == original


@pytest.mark.parametrize('failure', ['broken-toml', 'missing-weight', 'case-collision', 'bad-depth', 'orphan-shard'])
def test_preflight_errors_leave_every_file_unchanged(tmp_path, failure):
    root, _, definition = basic(tmp_path)
    if failure == 'broken-toml':
        definition.write_text('not TOML')
    elif failure == 'missing-weight':
        definition.write_text('[llm]\nmodel="Local/missing.gguf"\n')
    elif failure == 'case-collision':
        define(root, 'Model')
    elif failure == 'bad-depth':
        define(root, 'nested/model')
    else:
        weight(root / 'Local/bad-00002-of-00002.gguf', index=1, count=2)
    before = snapshot(root)
    with pytest.raises((ValueError, OSError)):
        tool.build_plan(root)
    assert snapshot(root) == before and not (root / '.tllama').exists()


def test_collision_is_found_before_hashing_weights(tmp_path, monkeypatch):
    root, _, _ = basic(tmp_path)
    define(root, 'Model')
    def forbidden(*args, **kwargs):
        raise AssertionError('Name preflight must precede multi-gigabyte hashing')
    monkeypatch.setattr(tool, 'prepare_group', forbidden)
    with pytest.raises(ValueError, match='collision'):
        tool.build_plan(root)


@pytest.mark.parametrize('schema', [1, 0, 999])
def test_legacy_digest_cache_is_discarded_and_current_cache_regenerated(tmp_path, schema):
    root, source, _ = basic(tmp_path)
    cache_dir = root / '.tllama/metadata-cache'
    cache.save_digest_cache(cache_dir, 'model', source, {'source': 'local', 'content_sha256': '0' * 64})
    old = cache.get_digest_cache_path(cache_dir, source)
    document = json.loads(old.read_text())
    document['schema_version'] = schema
    old.write_text(json.dumps(document))
    os.utime(old, ns=(source.stat().st_mtime_ns,) * 2)
    absolute_cache = cache_dir / (sha256(str(source).encode()).hexdigest() + '.json')
    absolute_cache.write_text('older-format JSON that cannot be reused')
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    target = root / plan['mapping']['Local/old.gguf']
    assert not old.exists() and not absolute_cache.exists()
    assert cache.load_digest_cache(cache_dir, target)['content_sha256'] == sha256(target.read_bytes()).hexdigest()
    assert cache.load_metadata_cache(cache_dir, target)['arch'] == 'llama'


def test_valid_current_cache_is_reused_without_rehash_and_comments_are_not_references(tmp_path, monkeypatch):
    root, source, definition = basic(tmp_path)
    cache_dir = root / '.tllama/metadata-cache'
    expected = sha256(source.read_bytes()).hexdigest()
    cache.save_digest_cache(cache_dir, 'model', source, {'source': 'local', 'content_sha256': expected})
    definition.write_text(definition.read_text() + '# Leave Local/old.gguf in this comment\n')
    def forbidden(*args):
        raise AssertionError('A valid locally measured digest must be reused')
    monkeypatch.setattr('tllama.repository_store.hash_file', forbidden)
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    assert '# Leave Local/old.gguf in this comment' in definition.read_text()
    assert plan['definitions'][0]['id'] == expected


@pytest.mark.parametrize('kind', ['old-metadata', 'old-format', 'corrupt-json'])
def test_pre_131_cache_documents_do_not_prevent_full_conversion(tmp_path, kind):
    root, source, _ = basic(tmp_path)
    cache_dir = root / '.tllama/metadata-cache'
    cache.save_metadata_cache(cache_dir, 'model', source, {'wrong': True})
    file = cache.get_metadata_cache_path(cache_dir, source)
    if kind == 'corrupt-json':
        file.write_text('{not JSON')
    else:
        document = json.loads(file.read_text())
        if kind == 'old-metadata':
            document['schema_version'] = 8
        else:
            document.pop('cache_format_version')
            document['model']['mtime_ns'] = source.stat().st_mtime_ns
            document['model']['path'] = str(source)
        file.write_text(json.dumps(document))
    os.utime(file, ns=(source.stat().st_mtime_ns,) * 2)
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    target = root / plan['mapping']['Local/old.gguf']
    assert not file.exists()
    assert cache.load_metadata_cache(cache_dir, target)['arch'] == 'llama'


def test_real_model_identity_is_unchanged_for_weights_projector_and_parameters(tmp_path):
    root, _, definition = basic(tmp_path)
    weight(root / 'Local/projector.gguf', arch='clip')
    weight(root / 'Local/adapter.gguf', arch='adapter')
    definition.write_text(definition.read_text() + '[mmproj]\nmodel="Local/projector.gguf"\n'
                          '[runtime]\nlora_path="Local/adapter.gguf"\nn_ctx=4096\n'
                          '[sampling]\ntemperature=0.5\n[system]\nprompt="Keep this prompt"\n')
    manager = ModelManager(BackendConfig(models_dir=str(root)))
    before = manager._listing_identity('model')
    plan = tool.build_plan(root)
    assert plan['definitions'][0]['id'] == before[0]
    tool.apply_plan(root, plan)
    # Resolve the converted definition through the actual format-2 runtime.
    manager.ensure_storage()
    assert manager._listing_identity('model') == before


@pytest.mark.parametrize('change', ['definition', 'weights', 'new-model'])
def test_changes_after_preview_abort_before_first_publication(tmp_path, change):
    root, source, definition = basic(tmp_path)
    plan = tool.build_plan(root)
    if change == 'definition':
        definition.write_text(definition.read_text() + '# Changed after preview\n')
    elif change == 'weights':
        source.write_bytes(b'changed after preview')
    else:
        define(root, 'another')
    with pytest.raises(ValueError, match='changed after preflight'):
        tool.apply_plan(root, plan)
    assert not (root / tool.JOURNAL).exists()
    assert not (root / 'data').exists() and source.exists()


@pytest.mark.parametrize('failure', ['published-group', 'definition-write', 'source-unlink', 'marker-written'])
def test_interrupted_migration_resumes_without_duplicate_or_lost_data(tmp_path, monkeypatch, failure):
    root, source, definition = basic(tmp_path)
    plan = tool.build_plan(root)
    expected = source.read_bytes()
    if failure == 'published-group':
        original = RepositoryStore.publish
        def fail(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise OSError('injected after group publication')
        monkeypatch.setattr(RepositoryStore, 'publish', fail)
    elif failure in {'definition-write', 'marker-written'}:
        original = tool.atomic_write
        def fail(root, relative, raw, **kwargs):
            if failure == 'definition-write' and relative == 'Local/model.toml':
                raise OSError('injected before definition publication')
            result = original(root, relative, raw, **kwargs)
            if failure == 'marker-written' and relative == tool.MARKER:
                raise OSError('injected after format marker publication')
            return result
        monkeypatch.setattr(tool, 'atomic_write', fail)
    else:
        original = Path.unlink
        def fail(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == source:
                raise OSError('injected after source removal')
            return result
        monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(OSError, match='injected'):
        tool.apply_plan(root, plan)
    assert (root / tool.JOURNAL).exists()
    if failure == 'marker-written':
        # The marker commits a complete conversion; the private journal may
        # still need its final status written, but the repository is usable.
        ModelManager(BackendConfig(models_dir=str(root))).ensure_storage()
    else:
        with pytest.raises(ConfigError, match='conversion is incomplete'):
            ModelManager(BackendConfig(models_dir=str(root))).ensure_storage()
    monkeypatch.undo()
    assert tool.apply_plan(root, resume=True)['status'] == 'converted'
    converted = root / parse_model_toml(definition.read_text()).llm_model
    assert converted.read_bytes() == expected and not source.exists()
    assert len(list((root / 'data').iterdir())) == 1
    assert tool.apply_plan(root, resume=True)['status'] == 'already-converted'


def test_new_definition_blocks_resumed_destructive_cleanup(tmp_path, monkeypatch):
    root, source, _ = basic(tmp_path)
    plan = tool.build_plan(root)
    original = tool.atomic_write
    def fail(root, relative, raw, **kwargs):
        if relative == 'Local/model.toml':
            raise OSError('injected')
        return original(root, relative, raw, **kwargs)
    monkeypatch.setattr(tool, 'atomic_write', fail)
    with pytest.raises(OSError):
        tool.apply_plan(root, plan)
    monkeypatch.undo()
    define(root, 'new-model')
    with pytest.raises(ValueError, match='acquired new'):
        tool.apply_plan(root, resume=True)
    assert source.exists()


def test_recreated_old_definition_blocks_cleanup_after_cutover(tmp_path, monkeypatch):
    root, source, definition = basic(tmp_path)
    renamed = definition.with_name('model:55b.toml')
    definition.rename(renamed)
    plan = tool.build_plan(root)
    unlink = Path.unlink
    def fail(path, *args, **kwargs):
        if path == source:
            raise OSError('injected before cleanup')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(OSError):
        tool.apply_plan(root, plan)
    monkeypatch.undo()
    renamed.write_bytes(base64.b64decode(plan['definitions'][0]['original']))
    with pytest.raises(ValueError, match='inventory changed before cleanup'):
        tool.apply_plan(root, resume=True)
    assert source.exists()


def test_missing_marker_after_completion_is_not_reported_as_converted(tmp_path):
    root, _, _ = basic(tmp_path)
    tool.apply_plan(root, tool.build_plan(root))
    (root / tool.MARKER).unlink()
    with pytest.raises(ValueError, match='no format marker'):
        tool.apply_plan(root, resume=True)


@pytest.mark.parametrize('failure', ['path-traversal', 'text-checksum', 'future-journal', 'future-cache-contract'])
def test_invalid_resume_journal_cannot_remove_arbitrary_files(tmp_path, monkeypatch, failure):
    root, source, _ = basic(tmp_path)
    plan = tool.build_plan(root)
    def fail(*args, **kwargs):
        raise OSError('injected')
    monkeypatch.setattr(RepositoryStore, 'publish', fail)
    with pytest.raises(OSError):
        tool.apply_plan(root, plan)
    monkeypatch.undo()
    file = root / tool.JOURNAL
    state = json.loads(file.read_text())
    if failure == 'path-traversal':
        state['plan']['groups'][0]['sources'][0] = '../outside.gguf'
    elif failure == 'text-checksum':
        state['plan']['definitions'][0]['original'] = base64.b64encode(b'corrupt').decode()
    elif failure == 'future-journal':
        state['journal_version'] = 999
    else:
        state['cache_contract']['metadata'] = 999
    file.write_text(json.dumps(state))
    outside = tmp_path / 'outside.gguf'
    outside.write_bytes(b'keep me')
    with pytest.raises(ValueError):
        tool.apply_plan(root, resume=True)
    assert outside.read_bytes() == b'keep me' and source.exists()


def test_staging_upload_links_are_reclaimed_and_unrelated_files_are_preserved(tmp_path):
    root, source, _ = basic(tmp_path)
    uploads = root / 'Local/.uploads'
    uploads.mkdir()
    os.link(source, uploads / sha256(source.read_bytes()).hexdigest())
    retained = root / 'Local/notes.txt'
    retained.write_text('user notes')
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    target = root / plan['mapping']['Local/old.gguf']
    assert not uploads.exists() and target.stat().st_nlink == 1
    assert retained.read_text() == 'user notes'


def test_crlf_and_untouched_prompt_content_survive_conversion(tmp_path):
    root, _, definition = basic(tmp_path)
    text = '# Comment\r\n[llm]\r\nmodel="Local/old.gguf"\r\n[system]\r\nprompt="Private Czech žluťoučký"\r\n'
    definition.write_bytes(text.encode())
    plan = tool.build_plan(root)
    tool.apply_plan(root, plan)
    raw = definition.read_bytes()
    assert b'\n' not in raw.replace(b'\r\n', b'')
    assert parse_model_toml(raw.decode()).system_prompt == 'Private Czech žluťoučký'


def test_cli_from_checkout_and_completed_conversion_are_idempotent(tmp_path):
    root, _, _ = basic(tmp_path)
    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), str(root), '--json', *args],
                              capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(run().stdout)['status'] == 'planned'
    assert json.loads(run('--apply').stdout)['status'] == 'converted'
    assert json.loads(run('--apply').stdout)['status'] == 'already-converted'
    assert json.loads(run('--resume').stdout)['status'] == 'already-converted'


@pytest.mark.parametrize('kind', ['file', 'directory', 'cache'])
def test_symlinks_are_preflight_errors(tmp_path, kind):
    root, source, _ = basic(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    if kind == 'file':
        source.unlink()
        source.symlink_to(weight(outside / 'model.gguf'))
    elif kind == 'directory':
        (root / 'linked').symlink_to(outside, target_is_directory=True)
    else:
        (root / '.tllama').mkdir()
        (root / '.tllama/metadata-cache').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='[Ss]ymlink'):
        tool.build_plan(root)


def test_unknown_repository_format_is_not_reinterpreted_as_legacy(tmp_path):
    root, _, _ = basic(tmp_path)
    (root / '.tllama').mkdir()
    (root / tool.MARKER).write_text('{"version":999}')
    before = snapshot(root)
    with pytest.raises(ValueError, match='Unknown repository format'):
        tool.build_plan(root)
    assert snapshot(root) == before


def test_startup_guard_does_not_need_access_to_private_journal(tmp_path):
    root, _, _ = basic(tmp_path)
    control = root / tool.CONTROL
    control.mkdir(parents=True, mode=0o700)
    with pytest.raises(ConfigError, match='conversion is incomplete'):
        ModelManager(BackendConfig(models_dir=str(root))).ensure_storage()
