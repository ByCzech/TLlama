"""Exact-build qualification: synthetic native TCP children, no hardware claim."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from tllama.native_memory_api import NativeMemoryReport
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'mtp-qualification-smoke.py'
smoke = runpy.run_path(str(SCRIPT))
BUILD = 'b11850-0e9433bfd3'


def arguments(tmp_path, *extra):
    return ['--model', str(tmp_path / 'model.gguf'), '--device', 'Vulkan0',
            '--expected-fingerprint', BUILD, '--output', str(tmp_path / 'capture'), *extra]


def test_dry_run_exact_build_and_explicit_mmap_without_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('TLLAMA_RUNTIME_SPEC_TYPE', 'none')
    monkeypatch.setenv('TLLAMA_RUNTIME_DEVICE', 'ROCm99')
    assert smoke['main'](arguments(tmp_path, '--dry-run', '--no-mmap')) == 0
    rows = json.loads(capsys.readouterr().out)
    assert set(rows) == {'off', 'mtp'}
    for name, command in rows.items():
        assert command[command.index('--device') + 1] == 'Vulkan0'
        assert command.count('--spec-type') == 1
        assert command[command.index('--spec-type') + 1] == ('none' if name == 'off' else 'draft-mtp')
        assert command[command.index('--load-mode') + 1] == 'none'
        assert command[command.index('--sleep-idle-seconds') + 1] == '-1'
    assert not (tmp_path / 'capture').exists()
    specs = smoke['launch_specs'](smoke['parser']().parse_args(arguments(tmp_path)))
    assert all(spec.version_policy == 'strict' and spec.runtime_fingerprint == BUILD for spec in specs.values())


@pytest.mark.parametrize('extra', [['--draft', '0'], ['--draft', '33'], ['--predict', '31'],
    ['--predict', '513'], ['--ctx', '0'], ['--expected-fingerprint', ' '],
    ['--request-timeout', 'nan']])
def test_bad_inputs_do_not_launch_or_create_output(tmp_path, extra):
    try:
        result = smoke['main'](arguments(tmp_path, *extra))
    except SystemExit as exc:
        result = exc.code
    assert result != 0 and not (tmp_path / 'capture').exists()


def test_main_mtp_shared_owner_is_a_union_not_role_sum(diagnosed):
    report = NativeMemoryReport(diagnosed)
    result = smoke['weight_evidence'](report, True)
    assert all(result['checks'].values())
    totals = result['totals']
    assert totals['union']['payload_bytes'] == result['unique_interval_bytes'] == 190
    assert totals['main']['payload_bytes'] + totals['mtp']['payload_bytes'] == 380
    # Tensor diagnostic visits cannot be used as payload totals.
    duplicate = deepcopy(diagnosed['weight_storage']['tensor_diagnostics']['records'][0])
    duplicate['id'] = 'second-visit'
    diagnosed['weight_storage']['tensor_diagnostics']['records'].append(duplicate)
    assert smoke['weight_evidence'](NativeMemoryReport(diagnosed), True) == result


@pytest.mark.parametrize('fault', ['no-role', 'no-head', 'no-storage', 'partial', 'draft-owner'])
def test_missing_or_incomplete_mtp_weight_evidence_fails(diagnosed, fault):
    if fault == 'no-role':
        diagnosed['owners'][0]['roles'] = ['main']
    elif fault == 'no-head':
        for item in diagnosed['weight_storage']['intervals']:
            item['relationships'] = [rel for rel in item['relationships'] if rel['component'] != 'mtp_layers']
    elif fault == 'no-storage':
        del diagnosed['weight_storage']
    elif fault == 'partial':
        diagnosed['weight_storage'].update(status='partial', missing_components=[
            dict(owner_id='model', component='mtp_layers', layer_index=40, reason='unsupported_backend')])
    else:
        diagnosed['owners'][0]['roles'] = ['draft']
    result = smoke['weight_evidence'](NativeMemoryReport(diagnosed), True)
    assert not all(result['checks'].values())


@pytest.mark.parametrize('mode,expected', [
    ('normal', None), ('punctuated-control', None), ('no-drafting', 'mtp:draft-counters'),
    ('no-stream-drafting', 'mtp:draft-counters'), ('no-full-drafting', 'mtp:draft-counters'),
    ('no-acceptance', 'mtp:draft-counters'), ('bad-counts', 'Inconsistent draft counters'),
    ('changed-weights', 'mtp:after-cancel:raw'),
    ('changed-control', 'mtp:cancel-reuse'), ('no-finish', 'Missing or invalid native finish'),
    ('no-done', 'complete terminal event'), ('http500', '500'), ('wrong-build', 'Unsupported runtime fingerprint'),
    ('no-memory', 'off:loaded:http'), ('auth-memory', 'off:loaded:http'),
    ('different-text', 'review'),
])
def test_owned_subprocess_capture_and_negative_evidence(tmp_path, diagnosed, mode, expected):
    root = tmp_path / 'stub'
    root.mkdir()
    diagnosed['runtime_fingerprint'] = BUILD
    for interval in diagnosed['weight_storage']['intervals']:
        interval['assignment']['device_ids'] = ['cpu']
    (root / 'memory-report').write_text(json.dumps(diagnosed))
    if mode == 'no-memory':
        (root / 'memory-report').unlink()
    if mode == 'auth-memory':
        (root / 'memory-status').write_text('401')
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + SCRIPT.parents[1].joinpath('tests/native_worker_stub.py').read_text())
    executable.chmod(0o700)
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic GGUF')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TLLAMA_', 'LLAMA_ARG_'))}
    env.update(STUB_ROOT=str(root), STUB_BUILD='wrong' if mode == 'wrong-build' else BUILD,
               STUB_MTP=mode, PYTHONPATH=str(SCRIPT.parents[1] / 'src'))
    command = [sys.executable, str(SCRIPT), *arguments(tmp_path, '--device', 'none',
        '--executable', str(executable), '--transport', 'tcp', '--request-timeout', '3', '--load-timeout', '5')]
    child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    out = tmp_path / 'capture'
    result = json.loads((out / 'result.json').read_text())
    assert result['ok'] is (expected is None), (result, child.stderr)
    assert child.returncode == (0 if expected is None else 1)
    summaries = [json.loads(line) for line in (out / 'summary.jsonl').read_text().splitlines()]
    cleanups = [row for row in summaries if row['stage'].endswith(':cleanup')]
    assert cleanups and all(row['ok'] and row['directory_removed'] for row in cleanups)
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    assert all(not Path(row['cwd']).exists() for row in launches)
    assert model.read_bytes() == b'synthetic GGUF'
    if expected == 'review':
        assert result['protocol_ok'] and result['review_required'] and len(result['off_on_differences']) == 4
    elif expected:
        assert expected in result['error'], result
    else:
        assert len(launches) == 2
        assert (root / 'cancelled').read_text().splitlines() == ['"mtp"', '"mtp"']
        assert len((out / 'raw.jsonl').read_text().splitlines()) > 6
        assert result['protocol_ok'] and not result['review_required']
        assert len([row for row in summaries if row['stage'] == 'off-on-comparison']) == 4
    before = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
    assert repeat.returncode != 0 and (root / 'launch').read_bytes() == before
