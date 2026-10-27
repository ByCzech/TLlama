"""Exact-build lifecycle capture uses owned TCP subprocesses, not a GPU.

Fault cases independently break the native fixture's sleep/failure protocol.
The fixture is synthetic; none of these results is hardware qualification.
"""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.worker import WorkerError
from test_native_memory_api import payload, assign
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'lifecycle-smoke.py'
module_spec = importlib.util.spec_from_file_location('lifecycle_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = smoke
module_spec.loader.exec_module(smoke)
BUILD = 'b99999-deadbeef'


def args(tmp_path, *extra):
    return smoke.parser().parse_args(['--model', str(tmp_path / 'm.gguf'), '--device', 'Vulkan0',
        '--expected-fingerprint', BUILD, '--output', str(tmp_path / 'capture'), *extra])


def test_launch_is_strict_exact_build_without_production_sleep_or_policy_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('TLLAMA_RUNTIME_VERSION_POLICY', 'warn')
    spec = smoke.launch_spec(args(tmp_path))
    key, socket = tmp_path / 'key', tmp_path / 'private.sock'
    launch = spec.command(key, socket_file=socket)
    sleeping = replace(spec, sleep_idle_seconds=5).command(key, socket_file=socket)
    assert spec.version_policy == 'strict' and spec.runtime_fingerprint == BUILD
    assert spec.expected_runtime_fingerprint == BUILD
    assert spec.memory_source == 'native' and spec.memory_compare
    assert launch[launch.index('--sleep-idle-seconds') + 1] == '-1'
    assert sleeping[sleeping.index('--sleep-idle-seconds') + 1] == '5'
    assert launch[launch.index('--host') + 1] == str(socket)
    assert '--memory-tensor-diagnostics' in launch
    assert '--fit' in launch and launch[launch.index('--fit') + 1] == 'off'
    assert spec.type_k == spec.type_v == 'q4_0'
    from tllama.worker_config import RUNTIME_FINGERPRINT, SUPPORTED_RUNTIME_FINGERPRINTS, WorkerLaunchSpec
    assert BUILD not in SUPPORTED_RUNTIME_FINGERPRINTS and BUILD != RUNTIME_FINGERPRINT
    production = WorkerLaunchSpec(model=tmp_path / 'm.gguf', device='none').command(key)
    assert production[production.index('--sleep-idle-seconds') + 1] == '-1'


@pytest.mark.parametrize('value', [0, -2, 61, True, 1.5])
def test_sleep_override_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError):
        replace(smoke.launch_spec(args(tmp_path)), sleep_idle_seconds=value)


@pytest.mark.parametrize('extra', [
    ['--expected-fingerprint', ' '], ['--ctx', '0'],
    ['--request-timeout', 'nan'], ['--load-timeout', '-1'], ['--transition-timeout', 'inf'],
    ['--transition-timeout', '5'], ['--memory-tensor-diagnostics', '4097']])
def test_bad_arguments_do_not_create_capture_or_start_workers(tmp_path, extra):
    try:
        result = smoke.main(['--model', str(tmp_path / 'm.gguf'), '--device', 'none',
            '--expected-fingerprint', BUILD, '--output', str(tmp_path / 'capture'), *extra])
    except SystemExit as exc:
        result = exc.code
    assert result != 0 and not (tmp_path / 'capture').exists()


def test_dry_run_preserves_existing_output_and_has_both_commands(tmp_path, capsys):
    capture = tmp_path / 'capture'
    capture.mkdir()
    sentinel = capture / 'keep'
    sentinel.write_text('original')
    assert smoke.main(['--model', str(tmp_path / 'm.gguf'), '--device', 'CUDA0',
        '--expected-fingerprint', BUILD, '--output', str(capture), '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {'initial_argv', 'sleep_argv'}
    assert sentinel.read_text() == 'original' and list(capture.iterdir()) == [sentinel]
    assert '--sleep-idle-seconds' in result['sleep_argv']


@pytest.mark.parametrize('state', ['loaded', 'sleeping'])
def test_evidence_validates_proved_empty_and_reloaded_payload(diagnosed, state):
    diagnosed['runtime_fingerprint'] = BUILD
    if state == 'sleeping':
        diagnosed.update(model_state=state, snapshot_stage='after_unload', owners=[], devices=[], buffers=[])
        diagnosed['placement'].update(status='known', missing_components=[], layer_groups=[])
        diagnosed['weight_storage']['intervals'] = []
        diagnosed['allocation_locality'].update(status='known', missing_components=[], allocations=[], bindings=[])
        diagnosed['weight_storage']['tensor_diagnostics']['records'] = []
    checks = smoke.evidence_checks(NativeMemoryReport(diagnosed), BUILD, state, 4096)
    assert all(checks.values()), checks


@pytest.mark.parametrize('fault', ['stale_weights', 'stale_locality', 'stale_diagnostics',
                                   'wrong_build', 'missing_weights', 'missing_locality', 'bad_diagnostics'])
def test_evidence_rejects_stale_or_invalid_extensions(diagnosed, fault):
    diagnosed['runtime_fingerprint'] = BUILD
    if fault == 'wrong_build':
        diagnosed['runtime_fingerprint'] = 'different'
    elif fault == 'missing_weights':
        del diagnosed['weight_storage']
    elif fault == 'missing_locality':
        del diagnosed['allocation_locality']
    elif fault == 'bad_diagnostics':
        diagnosed['weight_storage']['tensor_diagnostics']['records'][0]['buffer_id'] = 'missing'
    # A claimed sleeping state with any of the old arrays cannot pass even if
    # those arrays are structurally valid. Do not zero them in the checker.
    state = 'sleeping' if fault.startswith('stale_') else 'loaded'
    if state == 'sleeping':
        diagnosed.update(model_state='sleeping', snapshot_stage='after_unload')
    checks = smoke.evidence_checks(NativeMemoryReport(diagnosed), BUILD, state, 4096)
    assert not all(checks.values())


async def test_cancel_requires_real_content_not_only_metadata():
    async def empty():
        yield {'choices': [{'delta': {'role': 'assistant'}}]}
        yield {'usage': {}}
    with pytest.raises(WorkerError, match='cancellation was not exercised'):
        await smoke.first_content(empty())
    async def reasoning():
        yield {'choices': [{'delta': {'role': 'assistant'}}]}
        yield {'choices': [{'delta': {'reasoning_content': 'thinking'}}]}
    assert smoke.content(await smoke.first_content(reasoning()))
    async def terminal():
        yield {'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]}
    with pytest.raises(WorkerError, match='already finished'):
        await smoke.first_content(terminal())


@pytest.mark.parametrize('mode,expected', [
    ('normal', None), ('stale-sleep', 'sleeping:raw'), ('no-sleep', 'TimeoutError'),
    ('wake-no-sequence', 'awakened:raw'), ('accept-invalid', 'failed-replacement'),
    ('invalid-timeout', 'failed-replacement'),
    ('wrong-build', 'Unsupported runtime fingerprint'), ('no-memory', 'initial-ready:http'),
    ('auth-memory', 'initial-ready:http'), ('busy-memory', 'initial-ready:http'),
])
def test_capture_subprocess_owns_all_lifecycle_resources(tmp_path, diagnosed, mode, expected):
    stub = tmp_path / 'stub'
    stub.mkdir()
    diagnosed['runtime_fingerprint'] = BUILD
    # Device=none must genuinely have no accelerator assignment, regardless
    # of provider names retained in the synthetic buffer inventory.
    for item in diagnosed['weight_storage']['intervals']:
        item['assignment']['device_ids'] = ['cpu']
    (stub / 'memory-report').write_text(json.dumps(diagnosed))
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + SCRIPT.parents[1].joinpath('tests/native_worker_stub.py').read_text())
    executable.chmod(0o700)
    model = tmp_path / 'm.gguf'
    model.write_bytes(b'test synthetic model')
    out = tmp_path / 'capture'
    env = {key: value for key, value in os.environ.items() if not key.startswith(('TLLAMA_', 'LLAMA_ARG_'))}
    env.update(STUB_ROOT=str(stub), STUB_LIFECYCLE=mode,
               STUB_BUILD='wrong' if mode == 'wrong-build' else BUILD,
               PYTHONPATH=str(SCRIPT.parents[1] / 'src'))
    if mode in {'auth-memory', 'busy-memory'}:
        (stub / 'memory-status').write_text('401' if mode == 'auth-memory' else '503')
    if mode == 'no-memory':
        (stub / 'memory-report').unlink()
    child = subprocess.run([sys.executable, str(SCRIPT), '--model', str(model), '--device', 'none',
        '--expected-fingerprint', BUILD, '--output', str(out), '--executable', str(executable),
        '--transport', 'tcp', '--sleep-idle-seconds', '1', '--transition-timeout', '2.5',
        '--request-timeout', '3', '--load-timeout', '5'], env=env, capture_output=True, timeout=30)
    result = json.loads((out / 'result.json').read_text())
    assert result['ok'] is (expected is None), (result, child.stderr.decode())
    assert child.returncode == (0 if expected is None else 1)
    if expected:
        assert expected in result['error']
    summaries = [json.loads(line) for line in (out / 'summary.jsonl').read_text().splitlines()]
    assert summaries[-1]['stage'] == 'cleanup' and summaries[-1]['ok']
    workers = summaries[-1]['workers']
    assert all(worker['returncode'] is not None for worker in workers)
    launch = [json.loads(line) for line in (stub / 'launch').read_text().splitlines()] if (stub / 'launch').exists() else []
    assert all(not Path(item['cwd']).exists() for item in launch)
    assert model.read_bytes() == b'test synthetic model'
    assert len(list(out.glob('worker-*.load.log'))) == len(workers)
    if mode in {'auth-memory', 'busy-memory'}:
        raw = json.loads((out / 'raw.jsonl').read_text().splitlines()[0])
        assert raw['telemetry']['source'] is None
        assert raw['telemetry']['native']['support'] != 'unsupported'
        assert raw['telemetry']['native']['reason'] == ('authentication_failed' if mode == 'auth-memory' else 'not_ready')
    if expected is None:
        assert len(workers) == 4
        assert workers[2]['returncode'] == 24
        assert all(workers[i]['returncode'] in (0, -15) for i in (0, 1, 3))
        assert all(check['ok'] for check in result['checks'])
        stages = {row['stage']: row for row in summaries}
        assert stages['recovered-generation']['generation'] > stages['new-generation']['generation']
        assert stages['new-generation']['generation'] > stages['ready']['generation']
        assert stages['sleeping:display']['display']['size'] == 0
        assert stages['awakened:display']['display']['size'] > 0
        raw = [json.loads(line) for line in (out / 'raw.jsonl').read_text().splitlines()]
        assert len(raw) == 9
        assert raw[-2]['raw_native']['body']['model_state'] == 'sleeping'
        assert raw[-1]['raw_native']['body']['model_state'] == 'loaded'
        requests = [json.loads(line) for line in (stub / 'requests').read_text().splitlines()]
        assert any(row['path'] == '/tokenize' for row in requests)
        assert len([row for row in requests if row['body'].get('ignore_eos')]) == 2
        cancelled = [json.loads(line) for line in (stub / 'cancelled').read_text().splitlines()]
        assert cancelled.count('stream') == 2


def test_fresh_worker_check_rejects_a_previous_generations_cached_evidence(diagnosed):
    from tllama.native_memory import NativeMemory
    from tllama.native_telemetry import MemoryTelemetry
    telemetry = MemoryTelemetry(NativeMemory(None), mode='native', compare=True)
    assert smoke.fresh_telemetry(telemetry.snapshot())
    telemetry.accept(NativeMemoryReport(diagnosed))
    assert not smoke.fresh_telemetry(telemetry.snapshot())


@pytest.mark.parametrize('device,expected', [('none', False), ('Vulkan0', True)])
def test_requested_backend_cannot_pass_with_wrong_weight_assignment(diagnosed, device, expected):
    checks = smoke.evidence_checks(NativeMemoryReport(diagnosed), diagnosed['runtime_fingerprint'],
                                  'loaded', 4096, device=device)
    assert checks['requested_offload'] is expected
