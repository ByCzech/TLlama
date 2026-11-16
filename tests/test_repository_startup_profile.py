"""The baseline must observe real cache work without changing its validity."""

import importlib.util
import json
from pathlib import Path

import gguf
import pytest

from tllama.worker import NativeWorker


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'repository-startup-profile.py'
SPEC = importlib.util.spec_from_file_location('repository_startup_diagnostic', SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / 'models'
    local = root / 'Local'
    local.mkdir(parents=True)

    def no_worker(*args, **kwargs):
        pytest.fail('Local startup profiling must never start a worker')

    monkeypatch.setattr(NativeWorker, 'start', no_worker)
    return root


def add_weights(root, name, *, shards=1, index=0):
    path = root / 'Local' / name
    writer = gguf.GGUFWriter(path, 'llama')
    writer.add_context_length(4096)
    if shards > 1:
        writer.add_uint32('split.count', shards)
        writer.add_uint32('split.no', index)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


def define(root, name, weights):
    path = root / 'Local' / (name + '.toml')
    path.write_text(f'[llm]\nmodel="Local/{weights}"\n[runtime]\ndevice="none"\n')
    return path


def run(root, output, *options):
    code = diagnostic.main(['--models-dir', str(root), '--repeat', '2',
                            '--output', str(output), *options])
    return code, json.loads(output.read_text())


def files(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('shards', [1, 2])
def test_cold_then_warm_shared_weights_and_alias_keep_cache_and_sources(repository, tmp_path, shards):
    names = (['a.gguf'] if shards == 1 else
             [f'a-{index + 1:05d}-of-00002.gguf' for index in range(shards)])
    for index, name in enumerate(names):
        add_weights(repository, name, shards=shards, index=index)
    define(repository, 'a', names[0])
    define(repository, 'b', names[0])
    original = files(repository)
    code, cold = run(repository, tmp_path / 'cold.json', '--trace', '--model', 'a')
    assert code == 0, cold
    first, repeated = [row for row in cold['phases'] if row['phase'].startswith('list:')]
    assert first['models'] == repeated['models'] == 2
    assert first['operations']['gguf.header_read'] == shards
    assert first['operations']['gguf.full_hash'] == shards
    assert repeated['operations'].get('gguf.header_read', 0) == 0
    assert repeated['operations'].get('gguf.full_hash', 0) == 0
    assert repeated['operations'].get('metadata.save_attempt', 0) == 0
    assert repeated['operations'].get('digest.save_attempt', 0) == 0
    warmed = files(repository)
    assert all(warmed[name] == value for name, value in original.items())

    alias = tmp_path / 'configured-alias'
    alias.symlink_to(repository, target_is_directory=True)
    code, warm = run(alias, tmp_path / 'warm.json', '--trace', '--model', 'a')
    assert code == 0, warm
    assert warm['models_dir'] == str(alias)
    assert all(row['operations'].get('gguf.header_read', 0) == 0 and
               row['operations'].get('gguf.full_hash', 0) == 0 and
               row['operations'].get('metadata.save_attempt', 0) == 0 and
               row['operations'].get('digest.save_attempt', 0) == 0 for row in warm['phases'])
    assert files(repository) == warmed


def test_timing_mode_does_not_install_counters(repository, tmp_path, monkeypatch):
    add_weights(repository, 'a.gguf')
    define(repository, 'a', 'a.gguf')
    monkeypatch.setattr(diagnostic.Measurements, 'wrap', lambda *args, **kwargs: pytest.fail('instrumented'))
    code, report = run(repository, tmp_path / 'timing.json', '--model', 'a')
    assert code == 0, report
    assert report['instrumented'] is False
    assert all(row['ok'] and row['seconds'] >= 0 and 'operations' not in row for row in report['phases'])
    assert [row['phase'] for row in report['phases']] == [
        'configuration', 'manager', 'list:1', 'list:2', 'inventory',
        'metadata:a:1', 'prepare:a:1', 'metadata:a:2', 'prepare:a:2']


def test_failure_restores_tracing_and_is_not_reported_as_success(repository, tmp_path, monkeypatch):
    from tllama import backend
    original_stat = Path.stat
    original_parse = backend.parse_model_toml

    async def broken(*args, **kwargs):
        raise OSError('cannot prepare model')

    add_weights(repository, 'a.gguf')
    define(repository, 'a', 'a.gguf')
    monkeypatch.setattr(backend.ModelManager, '_resolve_native_definition', broken)
    code, report = run(repository, tmp_path / 'error.json', '--trace', '--model', 'a')
    assert code == 1 and report['ok'] is False
    assert report['error'] == 'OSError: cannot prepare model'
    assert report['phases'][-1]['phase'] == 'prepare:a:1' and not report['phases'][-1]['ok']
    assert Path.stat is original_stat and backend.parse_model_toml is original_parse


def test_existing_output_and_missing_repository_are_not_overwritten(repository, tmp_path):
    output = tmp_path / 'existing.json'
    output.write_text('keep me')
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(['--models-dir', str(repository), '--output', str(output)])
    assert exc.value.code == 2 and output.read_text() == 'keep me'
    missing = tmp_path / 'missing'
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(['--models-dir', str(missing), '--output', str(tmp_path / 'no.json')])
    assert exc.value.code == 2 and not missing.exists() and not (tmp_path / 'no.json').exists()
