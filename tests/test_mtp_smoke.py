"""Independent native MTP diagnostic: real TCP/children with synthetic counters."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.worker import WorkerError
from test_native_memory_api import payload


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'mtp-smoke.py'
spec = importlib.util.spec_from_file_location('mtp_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_missing_draft_measurements_are_unknown_not_zero():
    values = smoke.metrics({})
    assert values['draft_n'] is values['draft_n_accepted'] is values['acceptance'] is None
    assert values['mtp_observed'] is False
    assert smoke.metrics({'draft_n': 12, 'draft_n_accepted': 6})['acceptance'] == 0.5
    assert smoke.metrics({'draft_n': 12, 'draft_n_accepted': 0})['mtp_observed'] is True


@pytest.mark.parametrize('timings', [None, [], {'draft_n': True}, {'draft_n': -1},
    {'draft_n': 1.5}, {'draft_n': 2, 'draft_n_accepted': 3}, {'draft_n_accepted': 1},
    {'predicted_ms': float('nan')}, {'predicted_per_second': float('inf')}, {'cache_n': '0'}])
def test_invalid_counters_are_not_published_as_measurements(timings):
    with pytest.raises(WorkerError):
        smoke.metrics(timings)


@pytest.mark.parametrize('kind', ['none', 'draft-mtp'])
def test_probe_owns_only_pinned_diagnostic_flags(kind, tmp_path):
    launch = smoke.ProbeLaunchSpec(model=tmp_path/'m.gguf', device='Vulkan0', speculation=kind)
    args = launch.command(tmp_path/'key')
    assert args[-4:] == ('--spec-type', kind, '--spec-draft-n-max', '3')
    assert '--offline' in args and args[args.index('--parallel') + 1] == '1'


@pytest.mark.parametrize('mode,ok,review', [
    ('normal', True, False), ('no-drafting', True, True), ('different-text', True, True),
    ('punctuated-control', True, False), ('changed-control', True, True),
    ('bad-counts', False, True), ('no-done', False, True), ('no-finish', False, True), ('http500', False, True),
])
def test_native_cli_off_on_cleanup_activation_and_cancellation(tmp_path, payload, mode, ok, review):
    root = tmp_path/'stub'; root.mkdir()
    (root/'memory-report').write_text(json.dumps(payload))
    models = tmp_path/'models'/'Local'; models.mkdir(parents=True)
    (models/'a.gguf').write_bytes(b'fake weights')
    definition = '[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_ctx=8192\n'
    (models/'a.toml').write_text(definition)
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n'+Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path/'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output)]
    environment = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1]/'src'), 'STUB_MTP': mode,
                   'STUB_ROOT': str(root), 'STUB_BUILD': payload['runtime_fingerprint'], 'STUB_PUBLIC_CTX': '4096'}
    run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review
    assert run.stdout == (output/'summary.jsonl').read_text()
    assert not any(row['stage'] == 'native-event' for row in map(json.loads, run.stdout.splitlines()))
    assert (models/'a.toml').read_text() == definition
    closed = [row for row in rows if row['stage'] == 'worker-closed']
    assert len(closed) == 2 and all(row['clean'] and row['directory_removed'] for row in closed)
    assert all(row['residency'] == [] for row in rows if row['stage'] == 'profile-finished')
    launches = [json.loads(line) for line in (root/'launch').read_text().splitlines()]
    assert [row['argv'][row['argv'].index('--spec-type') + 1] for row in launches] == ['none', 'draft-mtp']
    assert all(row['argv'][row['argv'].index('--ctx-size') + 1] == '4096' for row in launches)
    if mode in {'normal', 'no-drafting', 'different-text', 'punctuated-control', 'changed-control'}:
        assert (root/'cancelled').read_text().splitlines() == ['"mtp"']
        assert next(r for r in rows if r['stage'] == 'after-cancel')['pid'] == closed[1]['pid']
        comparisons = [row for row in rows if row['stage'] == 'comparison']
        assert len(comparisons) == 4
        assert all(row['exact_text_match'] is (mode != 'different-text') for row in comparisons)
        stream_comparisons = [row for row in rows if row['stage'] == 'stream-comparison']
        assert len(stream_comparisons) == 4 and all(row['exact_text_match'] for row in stream_comparisons)
        assert any(r['mtp_owner_ids'] for r in rows if r['stage'] == 'memory')
    before = (root/'launch').read_bytes()
    repeat = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root/'launch').read_bytes() == before


@pytest.mark.parametrize('mode,review', [('normal', False), ('no-reuse', True),
    ('unknown-cache', True), ('stale-state', True), ('wrong-finish', True),
    ('wrong-count', True), ('unknown-count', True)])
def test_state_cli_cache_controls_branch_recovery_and_configured_mtp(tmp_path, payload, mode, review):
    root = tmp_path/'stub'; root.mkdir()
    (root/'memory-report').write_text(json.dumps(payload))
    models = tmp_path/'models'/'Local'; models.mkdir(parents=True)
    (models/'a.gguf').write_bytes(b'fake weights')
    definition = ('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_ctx=8192\n'
                  'spec_type="draft-mtp"\ndraft_n_max=8\n')
    (models/'a.toml').write_text(definition)
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n'+Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path/'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--state-only', '--model', 'a',
        '--models-dir', str(models.parent), '--executable', str(executable), '--output', str(output)],
        env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1]/'src'), 'STUB_MTP': 'normal',
             'STUB_MTP_STATE': mode, 'STUB_ROOT': str(root), 'STUB_BUILD': payload['runtime_fingerprint'],
             'STUB_PUBLIC_CTX': '4096'}, capture_output=True, text=True, timeout=25)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] and rows[-1]['review_required'] is review
    checks = [r for r in rows if r['stage'] == 'cache-comparison']
    assert len(checks) == 10
    assert all(r['exact_text_match'] for r in checks) is (mode != 'stale-state')
    assert all(r['finish_match'] for r in checks) is (mode != 'wrong-finish')
    assert all(r['output_tokens_match'] for r in checks) is (mode not in {'wrong-count', 'unknown-count'})
    if mode == 'unknown-count':
        assert all(r['output_tokens_match'] is None for r in checks)
    assert (root/'cancelled').read_text().splitlines() == ['"mtp"']
    assert (models/'a.toml').read_text() == definition
    launches = [json.loads(line) for line in (root/'launch').read_text().splitlines()]
    assert len(launches) == 2
    for launch, strategy in zip(launches, ['none', 'draft-mtp']):
        argv = launch['argv']
        assert argv.count('--spec-type') == argv.count('--spec-draft-n-max') == 1
        assert argv[argv.index('--spec-type')+1] == strategy
        assert argv[argv.index('--spec-draft-n-max')+1] == '3'
    requests = [r['body'] for r in map(json.loads, (root/'requests').read_text().splitlines())
                if r.get('body', {}).get('messages')]
    # Each cached comparison has a real independent cache-disabled request,
    # not just an off/on comparison with a potentially shared stale state.
    cold = [r for r in requests if r.get('cache_prompt') is False]
    assert len({json.dumps(r['messages'], sort_keys=True) for r in cold}) >= 4
    assert any(r.get('cache_prompt') and len(r['messages']) == 4 for r in requests)
    assert all(r['clean'] for r in rows if r['stage'] == 'worker-closed')
    assert all(r['residency'] == [] for r in rows if r['stage'] == 'profile-finished')
