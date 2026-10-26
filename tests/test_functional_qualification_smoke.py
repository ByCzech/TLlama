"""Public exact-build qualification over owned native TCP subprocess fixtures."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import gguf
import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.worker import WorkerError
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed
from test_native_fim import write_vocab


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'functional-qualification-smoke.py'
smoke = runpy.run_path(str(SCRIPT))
BUILD = 'b11850-0e9433bfd3'


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
    ['--require-mtp']])
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
    assert RUNTIME_FINGERPRINT != BUILD and BUILD not in SUPPORTED_RUNTIME_FINGERPRINTS
