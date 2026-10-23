"""The hardware diagnostic must distinguish a missing head from generic failure."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'mtp-smoke.py'


@pytest.mark.parametrize('mode,ok', [('missing-head', True), ('unrelated-error', False),
    ('unexpected-ready', False), ('timeout', False), ('changed-recovery', False), ('fatal', False)])
def test_negative_cli_reason_no_fallback_control_recovery_and_cleanup(tmp_path, mode, ok):
    root = tmp_path / 'stub'
    root.mkdir()
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    (models / 'a.gguf').write_bytes(b'fake weights')
    definition = ('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\n'
                  'spec_type="draft-mtp"\ndraft_n_max=8\n')
    (models / 'a.toml').write_text(definition)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--negative-only', '--model', 'a',
        '--models-dir', str(models.parent), '--executable', str(executable), '--output', str(output),
        '--load-timeout', '0.5' if mode == 'timeout' else '5', '--request-timeout', '5']
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_ROOT': str(root),
           'STUB_MTP': 'normal', 'STUB_MTP_NEGATIVE': mode, 'STUB_PUBLIC_CTX': '4096',
           'STUB_MODE': 'fatal' if mode == 'fatal' else 'normal'}
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    final = rows[-1]
    assert final['stage'] == 'finished' and final['review_required'] is not ok
    assert final['public_api'] == 'not_exercised'
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert (models / 'a.toml').read_text() == definition
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    assert len(launches) == 3  # Exactly one launch per profile, no disabled retry.
    for launch, strategy in zip(launches, ['none', 'draft-mtp', 'none']):
        argv = launch['argv']
        assert argv.count('--spec-type') == argv.count('--spec-draft-n-max') == 1
        assert argv[argv.index('--spec-type') + 1] == strategy
        assert argv[argv.index('--spec-draft-n-max') + 1] == '3'
    closed = [r for r in rows if r['stage'] == 'worker-closed']
    assert len(closed) == 3 and all(r['directory_removed'] for r in closed)
    assert all(r['residency'] == [] for r in rows if r['stage'] == 'profile-finished')
    if ok:
        assert final['protocol_ok'] and final['observations'] == []
        rejection = next(r for r in rows if r['stage'] == 'startup-rejection')
        assert rejection['expected'] and rejection['missing_head_reason']
        assert all(r['clean'] for r in closed)
        recovery = next(r for r in rows if r['stage'] == 'cache-comparison')
        assert recovery['exact_text_match'] and recovery['finish_match'] and recovery['output_tokens_match']
    else:
        assert final['observations'] or any(r['stage'] == 'error' for r in rows)
    before = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root / 'launch').read_bytes() == before


def test_negative_and_state_modes_cannot_be_combined(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), '--negative-only', '--state-only',
        '--output', str(tmp_path / 'unused')], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')}, timeout=5)
    assert run.returncode == 2 and not (tmp_path / 'unused').exists()
