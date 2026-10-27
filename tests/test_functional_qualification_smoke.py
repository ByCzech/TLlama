"""Public exact-build qualification over owned native TCP subprocess fixtures."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import gguf
import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.worker import WorkerError
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed
from test_native_fim import write_vocab


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'functional-qualification-smoke.py'
smoke = runpy.run_path(str(SCRIPT))
BUILD = 'b99999-deadbeef'


def arguments(tmp_path, profile='tools', *extra):
    return ['--profile', profile, '--model', 'a', '--models-dir', str(tmp_path / 'models'),
            '--device', 'none', '--expected-fingerprint', BUILD,
            '--output', str(tmp_path / 'capture'), *extra]


def setup_files(tmp_path, diagnosed, profile):
    root = tmp_path / 'stub'
    root.mkdir()
    report = deepcopy(diagnosed)
    report['runtime_fingerprint'] = BUILD
    for interval in report['weight_storage']['intervals']:
        interval['assignment']['device_ids'] = ['cpu']
    (root / 'memory-report').write_text(json.dumps(report))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    model = models / 'a.gguf'
    if profile.startswith('fim'):
        write_vocab(model, 'negative' if profile == 'fim-unsupported' else 'codegemma')
    elif profile == 'embeddings':
        writer = gguf.GGUFWriter(str(model), arch='bert')
        writer.add_uint32('bert.pooling_type', 1)
        writer.add_uint32('bert.embedding_length', 8)
        writer.add_uint32('bert.context_length', 256)
        writer.add_token_list([f'token{i}' for i in range(512)])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
    else:
        model.write_bytes(b'fake test weights')
    definition = '[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="Vulkan99"\nn_ctx=8192\n'
    (models / 'a.toml').write_text(definition)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + SCRIPT.parents[1].joinpath('tests/native_worker_stub.py').read_text())
    executable.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TLLAMA_', 'LLAMA_ARG_'))}
    env.update(PYTHONPATH=str(SCRIPT.parents[1] / 'src'), STUB_ROOT=str(root), STUB_BUILD=BUILD,
               STUB_PUBLIC_CTX='256' if profile == 'embeddings' else '4096')
    if profile == 'tools':
        env['STUB_TOOLS'] = 'normal'
    elif profile.startswith('fim'):
        env.update(STUB_FIM='negative' if profile == 'fim-unsupported' else 'codegemma', STUB_PUBLIC_FIM='1')
    elif profile == 'embeddings':
        env.update(STUB_EMBEDDINGS='normal', STUB_PUBLIC_EMBEDDINGS='1', STUB_EMBEDDING_POOLING='1')
    return root, models, executable, env, definition


@pytest.mark.parametrize('profile,mtp', [('text', False), ('text', True), ('tools', False),
                                       ('fim', False), ('fim-unsupported', False), ('embeddings', False)])
def test_public_reference_profiles_owned_cleanup_and_unmodified_store(tmp_path, diagnosed, profile, mtp):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, profile)
    if mtp:
        env['STUB_PUBLIC_MTP'] = '1'
    command = [sys.executable, str(SCRIPT), *arguments(tmp_path, profile, '--executable', str(executable),
        '--transport', 'tcp', '--load-timeout', '5', '--request-timeout', '5',
        *(['--require-mtp'] if mtp else []))]
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture' / 'result.json').read_text())
    assert result['ok'] and run.returncode == 0, (result, run.stdout, run.stderr)
    assert result['protocol_ok'] and not result['review_required']
    assert result['public_api'] == 'exercised' and result['mmap']
    rows = [json.loads(line) for line in (tmp_path / 'capture' / 'raw.jsonl').read_text().splitlines()]
    memory = [r for r in rows if r['stage'] == 'qualification-memory']
    assert len(memory) == 2 and {r['observation'] for r in memory} == {'loaded', 'before-close'}
    assert all(r['raw_native']['http_status'] == 200 for r in memory)
    if profile == 'text':
        recovery = next(r for r in rows if r['stage'] == 'after-stream-cancel-start')
        assert recovery['request_timeout'] == 5
    assert len([r for r in rows if r['stage'] == 'worker-cleanup']) == 1
    assert (models / 'a.toml').read_text() == definition
    assert not (models.parent / '.tllama').exists()
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    assert len(launches) == 1 and all(not Path(r['cwd']).exists() for r in launches)
    launch = launches[0]['argv']
    assert launch[launch.index('--device') + 1] == 'none'
    assert launch[launch.index('--ctx-size') + 1] == ('256' if profile == 'embeddings' else '4096')
    assert launch[launch.index('--spec-type') + 1] == ('draft-mtp' if mtp else 'none')
    if profile == 'embeddings':
        assert '--embeddings' in launch and any(r['stage'] == 'sdk-skipped' for r in rows)
        assert not result['sdks']
    if mtp:
        assert len([r for r in rows if r['stage'] == 'public-mtp-check']) == 8
    before = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert repeat.returncode != 0 and before == (root / 'launch').read_bytes()


@pytest.mark.parametrize('profile,fault,protocol_ok', [
    ('tools', 'wrong-build', False), ('tools', 'no-memory', False), ('tools', 'auth-memory', False),
    ('tools', 'wrong-memory-build', False), ('tools', 'bad-weight-reference', False),
    ('tools', 'semantic-bad', True), ('tools', 'no-done', False),
    ('embeddings', 'drift', True), ('embeddings', 'bad-usage', False),
])
def test_failures_and_semantic_review_cannot_pass(tmp_path, diagnosed, profile, fault, protocol_ok):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, profile)
    if fault == 'wrong-build':
        env['STUB_BUILD'] = 'wrong'
    elif fault == 'no-memory':
        (root / 'memory-report').unlink()
    elif fault == 'auth-memory':
        (root / 'memory-status').write_text('401')
    elif fault in {'wrong-memory-build', 'bad-weight-reference'}:
        data = json.loads((root / 'memory-report').read_text())
        if fault == 'wrong-memory-build':
            data['runtime_fingerprint'] = 'other-build'
        else:
            data['weight_storage']['intervals'][0]['buffer_id'] = 'absent-buffer'
        (root / 'memory-report').write_text(json.dumps(data))
    else:
        env['STUB_TOOLS' if profile == 'tools' else 'STUB_EMBEDDINGS'] = fault
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, profile,
        '--executable', str(executable), '--transport', 'tcp', '--load-timeout', '5', '--request-timeout', '5')],
        env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture' / 'result.json').read_text())
    assert run.returncode == 1 and not result['ok'] and result['review_required'], (result, run.stderr)
    assert result['protocol_ok'] is protocol_ok, (result, run.stderr)
    assert all(check['ok'] for check in result['checks'] if check['stage'] == 'worker-cleanup')
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    assert launches and all(not Path(r['cwd']).exists() for r in launches)
    assert (models / 'a.toml').read_text() == definition and not (models.parent / '.tllama').exists()


@pytest.mark.parametrize('profile', ['text', 'embeddings'])
def test_preview_is_exact_ignores_global_overrides_and_writes_no_capture_or_store_cache(tmp_path, diagnosed, profile):
    root, models, executable, env, _ = setup_files(tmp_path, diagnosed, profile)
    env.update(TLLAMA_RUNTIME_DEVICE='ROCm99', TLLAMA_RUNTIME_SPEC_TYPE='draft-mtp',
               TLLAMA_RUNTIME_DRAFT_N_MAX='7')
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, profile,
        '--executable', str(executable), '--dry-run', '--no-mmap')], env=env,
        capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr + run.stdout
    result = json.loads(run.stdout)
    argv = result['argv']
    assert argv[argv.index('--load-mode') + 1] == 'none'
    assert argv[argv.index('--device') + 1] == 'none'
    assert result['expected_fingerprint'] == BUILD
    assert ('--embeddings' in argv) is (profile == 'embeddings')
    assert not (root / 'launch').exists() and not (tmp_path / 'capture').exists()
    assert not (models.parent / '.tllama').exists()


@pytest.mark.parametrize('extra', [['--expected-fingerprint', ' '], ['--ctx', '0'],
    ['--request-timeout', 'nan'], ['--load-timeout', '-1'], ['--with-sdks'],
    ['--require-mtp'], ['--fim-no-prompt-cache']])
def test_invalid_inputs_never_create_capture(tmp_path, extra):
    assert smoke['main'](arguments(tmp_path, 'tools', *extra)) == 1
    assert not (tmp_path / 'capture').exists()


def test_file_changes_cannot_be_hidden_by_successful_feature_results(tmp_path):
    args = smoke['parser']().parse_args(arguments(tmp_path))
    smoke['validate'](args)
    args.output.mkdir()
    capture = smoke['Capture'](args)
    file = tmp_path / 'definition.toml'
    file.write_text('original')
    capture.identity(file)
    file.write_text('changed definition')
    with pytest.raises(WorkerError, match='file-identity'):
        capture.identity(file)
    assert capture.checks[-1]['ok'] is False


def test_candidate_pin_does_not_admit_decision_or_change_global_policy(tmp_path):
    from tllama.worker_config import WorkerLaunchSpec, RUNTIME_FINGERPRINT, SUPPORTED_RUNTIME_FINGERPRINTS
    from tllama.native_config import ResolvedWorkerConfig
    from tllama.residency import ModelDefinition
    args = smoke['parser']().parse_args(arguments(tmp_path))
    launch = WorkerLaunchSpec(model=tmp_path / 'model.gguf', device='none', decision_type='laya')
    definition = ModelDefinition('a', ResolvedWorkerConfig(launch, {}, {}, {}, None, None, None))
    with pytest.raises(WorkerError, match='Decision requires separate'):
        smoke['pin_definition'](definition, args)
    accepted = smoke['pin_definition'](replace(definition,
        resolved=replace(definition.resolved, launch=replace(launch, decision_type=None))), args)
    assert accepted.resolved.launch.runtime_fingerprint == BUILD
    assert accepted.resolved.launch.version_policy == 'strict'
    assert accepted.resolved.launch.expected_runtime_fingerprint == BUILD
    assert RUNTIME_FINGERPRINT != BUILD and BUILD not in SUPPORTED_RUNTIME_FINGERPRINTS


def test_external_model_headers_scanned_once_without_store_cache(tmp_path, diagnosed, monkeypatch):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, 'embeddings')
    args = smoke['parser']().parse_args(arguments(tmp_path, 'embeddings',
        '--output', str(tmp_path / 'isolated' / 'capture')))
    smoke['validate'](args)
    args.output.mkdir(parents=True)
    capture = smoke['Capture'](args)
    module = smoke['module_for']('embeddings')
    manager = smoke['manager_type'](args, capture, module)(
        BackendConfig(models_dir=str(models.parent), runtime_executable=executable))
    original = ModelManager._get_model_metadata_sync
    scans = []
    def scan(self, path):
        scans.append(path)
        return original(self, path)
    monkeypatch.setattr(ModelManager, '_get_model_metadata_sync', scan)
    first = manager._resolve_native_definition_sync('a', args.ctx)
    second = manager._resolve_native_definition_sync('a', args.ctx)
    assert first == second and len(scans) == 1
    assert first.metadata['arch'] == 'bert'
    assert not list(manager.metadata_cache_dir.glob('*'))
    assert not (models.parent / '.tllama').exists()
    assert (models / 'a.toml').read_text() == definition


def header_manager(tmp_path):
    args = smoke['parser']().parse_args(arguments(tmp_path))
    smoke['validate'](args)
    args.output.mkdir()
    capture = smoke['Capture'](args)
    manager = smoke['manager_type'](args, capture, smoke['module_for']('tools'))(
        BackendConfig(models_dir=str(tmp_path / 'models')))
    return manager


def test_header_cache_concurrent_reads_and_returned_mutation(tmp_path, monkeypatch):
    manager = header_manager(tmp_path)
    model = tmp_path / 'weights.gguf'
    model.write_bytes(b'original')
    scans = []
    def scan(self, path):
        scans.append(path)
        return {'nested': {'values': [1, 2]}}
    monkeypatch.setattr(ModelManager, '_get_model_metadata_sync', scan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(manager._get_model_metadata_sync, [str(model)] * 8))
    assert len(scans) == 1
    values[0]['nested']['values'].append(3)
    assert manager._get_model_metadata_sync(str(model)) == {'nested': {'values': [1, 2]}}


@pytest.mark.parametrize('change', ['replace', 'during-scan', 'missing-result'])
def test_header_cache_retries_changed_or_unsuccessful_scans(tmp_path, monkeypatch, change):
    manager = header_manager(tmp_path)
    model = tmp_path / 'weights.gguf'
    model.write_bytes(b'original')
    scans = []
    def scan(self, path):
        scans.append(path)
        if len(scans) == 1:
            if change == 'during-scan':
                model.write_bytes(b'changed while scanning')
            if change == 'missing-result':
                return None
        return {'scan': len(scans)}
    monkeypatch.setattr(ModelManager, '_get_model_metadata_sync', scan)
    manager._get_model_metadata_sync(str(model))
    if change == 'replace':
        replacement = tmp_path / 'replacement.gguf'
        replacement.write_bytes(b'original')
        stat = model.stat()
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(model)
    assert manager._get_model_metadata_sync(str(model)) == {'scan': 2}
    assert manager._get_model_metadata_sync(str(model)) == {'scan': 2}
    assert len(scans) == 2


@pytest.mark.parametrize('profile', ['text', 'tools'])
def test_startup_error_survives_cleanup_with_reaped_nonzero_process(tmp_path, diagnosed, profile):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, profile)
    env['STUB_DEVICE_FAILURES'] = json.dumps({'none': 'invalid device: CUDA0'})
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, profile,
        '--executable', str(executable), '--transport', 'tcp', '--load-timeout', '5', '--request-timeout', '5')],
        env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture' / 'result.json').read_text())
    assert run.returncode == 1 and not result['ok'] and not result['protocol_ok']
    rows = [json.loads(line) for line in (tmp_path / 'capture' / 'raw.jsonl').read_text().splitlines()]
    assert any('invalid device: CUDA0' in row.get('error', '') for row in rows)
    assert all(c['ok'] for c in result['checks'] if c['stage'] == 'worker-cleanup')
    assert not any(c['stage'] == 'worker-exit' for c in result['checks'])
    if profile == 'text':
        assert 'invalid device: CUDA0' in result['error']


async def test_cleanup_failure_keeps_primary_error_and_restores_managers(tmp_path):
    from tllama import main
    from tllama.routers import ollama, openai
    args = smoke['parser']().parse_args(arguments(tmp_path, 'text'))
    smoke['validate'](args)
    args.output.mkdir()
    capture = smoke['Capture'](args)
    definition = tmp_path / 'definition.toml'
    definition.write_text('original')
    class Manager:
        def __init__(self, config=None):
            pass
        def _toml_path_for_reference(self, name):
            return definition
        async def shutdown(self):
            raise WorkerError('cleanup also failed')
    async def fail(args):
        main.model_manager = ollama.model_manager = openai.model_manager = object()
        raise WorkerError('original startup failure')
    module = SimpleNamespace(ModelManager=Manager, load_daemon_config=lambda: SimpleNamespace(backend=None),
                             exercise=fail)
    capture.managers.append(Manager())
    previous = main.model_manager, ollama.model_manager, openai.model_manager
    with pytest.raises(WorkerError, match='^original startup failure$'):
        await smoke['exercise'](args, module, capture)
    assert previous == (main.model_manager, ollama.model_manager, openai.model_manager)
    assert any(row['stage'] == 'cleanup-error' and 'cleanup also failed' in row['error'] for row in capture.events)


async def test_cancel_recovery_uses_requested_deadline_and_reports_expiry(monkeypatch):
    api = smoke['module_for']('text')
    calls = []
    async def post(path, body):
        assert path == '/api/chat' and body == {'model': 'a'}
        await asyncio.sleep(0.02)
        return 'response'
    real_wait_for = asyncio.wait_for
    async def timed(awaitable, timeout):
        calls.append(timeout)
        return await real_wait_for(awaitable, timeout)
    monkeypatch.setattr(api.asyncio, 'wait_for', timed)
    monkeypatch.setattr(api, 'report', lambda *a, **k: None)
    assert await api.request_after_cancel(post, {'model': 'a'}, 600) == 'response'
    with pytest.raises(WorkerError, match='--request-timeout'):
        await api.request_after_cancel(post, {'model': 'a'}, 0.001)
    assert calls == [600, 0.001]


@pytest.mark.parametrize('stream', [False, True])
def test_fim_reference_mismatch_retains_both_values_requests_and_final_memory(tmp_path, diagnosed, stream):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, 'fim')
    (root / 'fim-public-mismatch').write_text('stream' if stream else 'nonstream')
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, 'fim',
        '--executable', str(executable), '--transport', 'tcp', '--load-timeout', '5', '--request-timeout', '5')],
        env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture' / 'result.json').read_text())
    rows = [json.loads(line) for line in (tmp_path / 'capture' / 'raw.jsonl').read_text().splitlines()]
    assert run.returncode == 1 and not result['protocol_ok']
    reference = next(row for row in rows if row['stage'] == 'reference')
    mismatch = next(row for row in rows if row['stage'] == 'completion' and not row['reference_matches'])
    assert reference['response']['content'] == 'a + b'
    assert mismatch['content'] == 'different native completion' and mismatch['stream'] is stream
    native = [row['request'] for row in rows if row['stage'] == 'fim-native-request']
    assert native[0]['prompt'] == native[-1]['prompt'] == reference['request']['prompt']
    assert native[0]['cache_prompt'] is False and 'cache_prompt' not in native[-1]
    public = [row for row in rows if row['stage'] == 'public-response'][-1]
    assert public['http_status'] == 200 and public['request']['stream'] is stream
    assert 'different native completion' in public['response']
    assert any(row['stage'] == 'error' and 'full direct FIM reference (add,' in row['error'] for row in rows)
    assert all(check['ok'] for check in result['checks'] if check['stage'].startswith('before-close:'))
    assert any(check['stage'] == 'before-close:http' for check in result['checks'])
    assert any(row['stage'] == 'worker-cleanup' and row['ok'] for row in rows)
    assert (models / 'a.toml').read_text() == definition


@pytest.mark.parametrize('fault', ['nonstream', 'stream', 'semantic-bad'])
def test_fim_cache_intervention_reaches_both_paths_and_retains_full_checks(tmp_path, diagnosed, fault):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, 'fim')
    # The native fixture reproduces a cache-dependent difference. Default
    # policy failure is covered above; this run changes only prompt reuse.
    if fault == 'semantic-bad':
        env['STUB_FIM'] = 'semantic-bad'
    else:
        (root / 'fim-public-mismatch').write_text(fault)
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, 'fim',
        '--fim-no-prompt-cache', '--executable', str(executable), '--transport', 'tcp',
        '--load-timeout', '5', '--request-timeout', '5')],
        env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture' / 'result.json').read_text())
    rows = [json.loads(line) for line in (tmp_path / 'capture' / 'raw.jsonl').read_text().splitlines()]
    assert result['protocol_ok'], (result, run.stderr)
    assert result['ok'] is (fault != 'semantic-bad')
    assert run.returncode == (1 if fault == 'semantic-bad' else 0)
    assert bool(result['observations']) is (fault == 'semantic-bad')
    assert result['fim_no_prompt_cache'] is True
    native = [row['request'] for row in rows if row['stage'] == 'fim-native-request']
    assert native and all(request['cache_prompt'] is False for request in native)
    comparisons = [row for row in rows if row['stage'] == 'completion']
    assert len(comparisons) == 12 and all(row['reference_matches'] for row in comparisons)
    assert {row['stream'] for row in comparisons} == {False, True}
    assert all(check['ok'] for check in result['checks'])
    assert any(check['stage'] == 'before-close:http' for check in result['checks'])
    assert any(row['stage'] == 'worker-cleanup' and row['ok'] for row in rows)
    assert (models / 'a.toml').read_text() == definition


@pytest.mark.parametrize('snapshot_error', [False, True])
async def test_concurrent_qualification_closes_wait_for_snapshot(tmp_path, monkeypatch, snapshot_error):
    args = smoke['parser']().parse_args(arguments(tmp_path, 'fim'))
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def memory(worker, stage):
        calls.append('memory-start')
        started.set()
        await release.wait()
        calls.append('memory-end')
        if snapshot_error:
            raise WorkerError('snapshot failed')
    async def close(worker):
        calls.append('native-close')
        worker._state = 'closed'
    monkeypatch.setattr(smoke['NativeWorker'], 'close', close)
    capture = SimpleNamespace(memory=memory)
    Worker = smoke['manager_type'](args, capture, None).Worker
    worker = object.__new__(Worker)
    worker.process = None
    worker._state = 'ready'
    worker._qualification_close_lock = asyncio.Lock()
    first = asyncio.create_task(worker.close())
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(worker.close())
    await asyncio.sleep(0)
    assert calls == ['memory-start']
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert calls == ['memory-start', 'memory-end', 'native-close', 'native-close']
    assert worker.state == 'closed'
    assert isinstance(results[0], WorkerError) if snapshot_error else results[0] is None
    assert results[1] is None
