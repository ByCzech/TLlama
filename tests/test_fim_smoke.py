"""Exercise diagnostic CLI/transport with an owned subprocess, not real FIM quality."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'fim-smoke.py'
spec = importlib.util.spec_from_file_location('fim_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize('mode,negative,ok', [
    ('normal', False, True), ('repo', False, True), ('semantic-bad', False, True),
    ('negative', True, True), ('missing-route', True, False), ('bad-terminal', False, False),
    ('normal', True, False), ('negative', False, False),
])
def test_cli_reference_comparison_negative_errors_and_cleanup(tmp_path, mode, negative, ok):
    root = tmp_path / 'stub'
    root.mkdir()
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    (models / 'a.gguf').write_bytes(b'fake weights')
    (models / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_batch=64\nn_ctx=8192\nspm_infill=true\n')
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--negative-model' if negative else '--model', 'a',
               '--models-dir', str(models.parent), '--executable', str(executable), '--output', str(output)]
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_FIM': mode,
           'STUB_ROOT': str(root), 'TLLAMA_MEMORY_SOURCE': 'log'}
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    records = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    summaries = [json.loads(line) for line in (output / 'summary.jsonl').read_text().splitlines()]
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert all('full_tokens' not in record and 'final' not in record for record in summaries)
    assert records[-1]['transport_ok'] is ok and records[-1]['public_suffix'] == 'not_implemented'
    assert records[-1]['review_required'] is (not ok or mode == 'semantic-bad')
    closed = [record for record in records if record['stage'] == 'worker-closed']
    assert len(closed) == (1 if negative else 2)
    assert all(record['clean'] and record['directory_removed']
               and record['returncode'] in (0, -signal.SIGTERM) for record in closed)
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    for index, launch in enumerate(launches):
        argv = launch['argv']
        # Diagnostic overrides win over TOML, without changing the stored TOML.
        assert argv[argv.index('--batch-size') + 1] == '512'
        assert argv[argv.index('--ctx-size') + 1] == '4096'
        assert ('--spm-infill' in argv) is (index == 1)
    if ok and not negative:
        comparisons = [record for record in records if record['stage'] == 'comparison']
        assert len(comparisons) == 4
        assert all(record['infill_matches_native_equivalent'] and record['infill_matches_full']
                   and record['infill_stream_matches'] and record['full_stream_matches'] for record in comparisons)
        assert len([r for r in records if r['stage'] == 'after-cancel']) == 2
        assert len((root / 'cancelled').read_text().splitlines()) == 2
        long = [r for r in records if r['stage'] == 'long-completion']
        assert len(long) == 6 and all(r['token_count_matches'] for r in long)
        prompts = [r for r in records if r['stage'] == 'long-prompts']
        assert all(r['full_counts']['total'] > r['native_equivalent_counts']['total'] for r in prompts)
        assert all(r['native_equivalent_counts']['used_prefix'] == 384
                   and r['native_equivalent_counts']['used_suffix'] == 126 for r in prompts)
        results = [r for r in records if r['stage'] == 'completion']
        assert all(r['semantic_match'] is (mode != 'semantic-bad') for r in results)
        assert all(r['full_counts']['extra'] == 0 for r in prompts)
        assert all((r['native_equivalent_counts']['extra'] > 0) is (mode == 'repo') for r in prompts)
    if ok and negative:
        assert len([r for r in records if r['stage'] == 'negative-rejected']) == 2
        assert any(r['stage'] == 'after-negative' for r in records)
    assert (output / 'worker.log').is_file()
    before = (root / 'launch').read_text()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert repeated.returncode != 0 and (root / 'launch').read_text() == before


@pytest.mark.parametrize('text,case,expected', [('a + b', 'add', True), ('b * a', 'multiply', True),
    ('a - b', 'add', False), ('__import__("os").system("anything")', 'add', False),
    ('a + b\nprint("extra")', 'add', False), ('```python\na+b\n```', 'add', False)])
def test_semantic_probe_only_inspects_expected_expression(text, case, expected):
    assert smoke.semantic_check(text, case) is expected


@pytest.mark.parametrize('extra', [['--ctx', '0'], ['--predict', '0'], ['--batch', '8'],
                                  ['--request-timeout', 'nan']])
def test_invalid_cli_does_not_create_evidence_directory(tmp_path, extra):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), *extra],
                         capture_output=True, text=True, timeout=10)
    assert run.returncode != 0 and not output.exists()
