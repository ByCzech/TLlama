"""Decision comparison harness against independent owned TCP processes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tllama.helpers.model_toml import parse_model_toml
from tllama.worker import WorkerError
from test_decision_smoke import header
from test_native_memory_api import payload


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'decision-api-smoke.py'
SPEC = importlib.util.spec_from_file_location('decision_public_diagnostic', SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_temporary_recipe_uses_registered_weights_placement_and_native_head():
    source = parse_model_toml('''[llm]
model="Local/a.gguf"
[runtime]
device="Vulkan0"
n_ctx=8192
type_k="q8_0"
spec_type="draft-mtp"
draft_n_max=7
spm_infill=true
embedding=true
pooling_type=3
[mmproj]
model="Local/p.gguf"
[sampling]
temperature=0.5
[system]
prompt="generation prompt"
''')
    actual = parse_model_toml(smoke.diagnostic_definition(source, 1024))
    assert actual.llm_model == source.llm_model and actual.runtime['device'] == 'Vulkan0'
    assert actual.runtime['n_ctx'] == actual.runtime['n_batch'] == actual.runtime['n_ubatch'] == 1024
    assert actual.runtime['type_k'] == actual.runtime['type_v'] == 'f16'
    assert actual.runtime['embedding'] is False and actual.runtime['pooling_type'] == -1
    assert actual.runtime['spec_type'] == 'none' and not actual.runtime['flash_attn']
    assert not {'draft_n_max', 'spm_infill'} & actual.runtime.keys()
    assert actual.mmproj_model is None and not actual.sampling and actual.system_prompt is None
    assert source.mmproj_model == 'Local/p.gguf' and source.runtime['n_ctx'] == 8192


@pytest.mark.parametrize('kind,mode,ok,review', [
    ('laya', 'normal', True, False), ('nimble', 'normal', True, False), ('clef', 'normal', True, False),
    ('laya', 'fast', True, False), ('nimble', 'fast', True, False), ('clef', 'fast', True, False),
    ('laya', 'second-worker:drift', True, True), ('clef', 'second-worker:drift', True, True),
    ('nimble', 'second-worker:usage-drift', False, True),
    ('laya', 'second-worker:bad-sum', False, True), ('clef', 'second-worker:wrong-option-order', False, True),
    ('nimble', 'second-worker:no-usage', False, True), ('laya', 'bad-confidence', False, True),
    ('laya', 'no-decisions', False, True),
])
def test_independent_managed_public_comparison_and_owned_lifecycle(tmp_path, payload, kind, mode, ok, review):
    run_comparison(tmp_path, payload, kind, mode, ok, review)


def run_comparison(tmp_path, payload, kind, mode, ok, review, count=1):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(dict(payload, runtime_fingerprint='b11552-3144c167ef')))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    header(models / 'a.gguf', kind=kind)
    definition = '# Original must stay byte-for-byte.\n[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\n'
    (models / 'a.toml').write_text(definition)
    if count == 2:
        header(models / 'b.gguf', kind=kind)
        (models / 'b.toml').write_text(definition.replace('a.gguf', 'b.gguf'))
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output)]
    if count == 2:
        command += ['--model', 'b']
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_ROOT': str(root),
           'STUB_DECISION': mode, 'STUB_DECISION_KIND': kind, 'STUB_BUILD': 'b11552-3144c167ef',
           'STUB_PUBLIC_CTX': '1024' if kind == 'laya' else '2048',
           'TLLAMA_RUNTIME_DRAFT_N_MAX': '3', 'TLLAMA_RUNTIME_SPEC_TYPE': 'draft-mtp'}
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=35)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review, run.stdout
    assert run.stdout == (output / 'summary.jsonl').read_text()
    closed = [r for r in rows if r['stage'] == 'worker-closed']
    assert closed and all(r['clean'] and r['directory_removed'] for r in closed)
    assert all(r['residency'] == [] for r in rows if r['stage'] == 'phase-finished')
    assert (models / 'a.toml').read_text() == definition
    assert not (models.parent / '.tllama').exists()
    assert next(r for r in rows if r['stage'] == 'gguf-fingerprint')['unchanged']
    if ok:
        assert len(closed) == len({r['pid'] for r in closed}) == 2 * count
        ready = [r for r in rows if r['stage'] == 'ready']
        assert [r['launch_path'] for r in ready] == ['independent', 'production'] * count
        assert smoke.stable_command(ready[0]['argv']) == smoke.stable_command(ready[1]['argv'])
        assert ready[1]['runtime']['n_ctx'] == (1024 if kind == 'laya' else 2048)
        assert '--embeddings' not in ready[1]['argv'] and '--mmproj' not in ready[1]['argv']
        checks = [r for r in rows if r['stage'] == 'comparison']
        assert len(checks) == (45 if kind == 'laya' else 57) * count
        assert all(r['within_tolerance'] for r in checks) is (not review)
        cold = [r for r in rows if r['stage'] == 'cold-start-comparison']
        assert len(cold) == count and all(r['within_tolerance'] for r in cold) is (not review)
        internal = [r for r in rows if r['stage'] == 'reference-comparison']
        assert len(internal) == {'laya': 8, 'nimble': 6, 'clef': 7}[kind] * count
        assert all(r['within_tolerance'] for r in internal)
        usage = next(r for r in rows if r['stage'] == 'usage-composition')
        assert (usage['single_input_tokens'] == usage['combined_input_tokens']) is (kind == 'laya')
        assert next(r for r in rows if r['stage'] == 'pre-admission')['workers'] == 0
        errors = [r for r in rows if r['stage'] == 'public-error']
        assert len(errors) == 18 * count and [r['status'] for r in errors] == ([400] * 15 + [404, 400, 400]) * count
        cancels = [r for r in rows if r['stage'] == 'transport-cancel']
        assert [r['source'] for r in cancels] == ['raw-native', 'managed', 'public'] * count
        assert all(r['inference_started_observed'] is r['native_early_stop_observed'] is None for r in cancels)
        if mode != 'fast':
            assert all(r['pending_at_cancel'] for r in cancels)
        reference_responses = [r for r in rows if r['stage'] == 'native-response' and r['probe'] != 'cancel-completed']
        assert len(reference_responses) == (15 if kind == 'laya' else 19) * count
        literal = next(r for r in rows if r['stage'] == 'public-response' and r['probe'] == 'literal-keys-unicode')
        assert list(literal['payload']['answers']) == ['řádek\nliteral\\n 🐈']
        assert literal['payload']['model'] == 'a'
        recovery = [r for r in rows if r['stage'] == 'recovery']
        assert [r['pid'] for r in recovery] == [r['pid'] for r in ready]
    launches = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root / 'launch').read_bytes() == launches
    if count == 2:
        assert (models / 'b.toml').read_text() == definition.replace('a.gguf', 'b.gguf')
        lifecycle = [(r['stage'], r['profile']) for r in rows if r['stage'] in {'ready', 'worker-closed'}]
        assert lifecycle == [(stage, profile) for profile in ('model-0', 'model-1')
                             for stage in ('ready', 'worker-closed', 'ready', 'worker-closed')]


def test_models_run_sequentially_without_overlapping_owned_workers(tmp_path, payload):
    run_comparison(tmp_path, payload, 'laya', 'fast', True, False, count=2)


def test_usage_is_exact_even_with_large_probability_tolerance():
    rows = []
    with pytest.raises(WorkerError, match='usage differs'):
        smoke.compare({'answers': {}, 'usage': {'input_tokens': 1, 'output_tokens': 0}},
                      {'answers': {}, 'usage': {'input_tokens': 2, 'output_tokens': 0}},
                      SimpleNamespace(atol=1, rtol=1), lambda *a, **k: rows.append(k), [], 'test')
    assert rows == []


def test_launch_comparison_ignores_only_port_and_owned_credential_file():
    a = ['native', '--model', 'a.gguf', '--port', '12', '--api-key-file', '/private/a', '--ctx-size', '1024']
    b = ['native', '--model', 'a.gguf', '--port', '34', '--api-key-file', '/private/b', '--ctx-size', '1024']
    assert smoke.stable_command(a) == smoke.stable_command(b)
    b[-1] = '2048'
    assert smoke.stable_command(a) != smoke.stable_command(b)


@pytest.mark.parametrize('option,value', [('--ctx', '1'), ('--ctx', '99999'), ('--atol', 'nan'),
    ('--request-timeout', '-1'), ('--cancel-after-ms', '0'), ('--model', ' ')])
def test_invalid_cli_starts_nothing(tmp_path, option, value):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), option, value],
                         env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')},
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not output.exists(), run.stderr


@pytest.mark.parametrize('kind', [None, 'pplx-decider'])
def test_unintegrated_metadata_starts_no_worker(tmp_path, kind):
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    header(models / 'a.gguf', kind=kind, head=False)
    (models / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n')
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
        '--executable', str(tmp_path / 'missing'), '--output', str(output)], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')}, timeout=10)
    assert run.returncode == 1
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['review_required'] and not any(r['stage'] in {'ready', 'worker-closed'} for r in rows)
