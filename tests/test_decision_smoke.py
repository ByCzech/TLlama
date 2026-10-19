"""Decision reference diagnostic through local GGUF headers and owned TCP stubs."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from tllama.worker import WorkerError
from tllama.worker_config import WorkerLaunchSpec
from test_native_memory_api import payload


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'decision-smoke.py'
SPEC = importlib.util.spec_from_file_location('decision_reference', SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def header(path, *, kind='laya', template=True, head=True):
    arch = {'nimble': 'qwen35', 'clef': 'clef'}.get(kind, 'modern-bert')
    writer = gguf.GGUFWriter(path, arch)
    writer.add_context_length(8192)
    writer.add_embedding_length(4)
    writer.add_block_count(4)
    writer.add_token_list(['<cls>', '<mask>', '<sep>', 'a'])
    writer.add_uint32('tokenizer.ggml.mask_token_id', 1)
    writer.add_uint32('tokenizer.ggml.seperator_token_id', 2)
    if kind:
        writer.add_string(arch + '.decision.type', kind)
    writer.add_uint32('modern-bert.decision.block_count', 2)
    writer.add_uint32('modern-bert.decision.max_head_tokens', 256)
    writer.add_float32('modern-bert.decision.temperature.choice', 1.2)
    if template:
        writer.add_string('tokenizer.chat_template.systemone', '{{ state }} {{ instructions }}')
    if head:
        for name in ('token_types.weight', 'cls.weight', 'cls.bias', 'cls.norm.weight',
                     'cls.norm.bias', 'cls.output.weight', 'cls.output.bias'):
            writer.add_tensor(name, np.ones((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_probe_preserves_placement_and_does_not_force_semantic_embeddings(tmp_path):
    original = WorkerLaunchSpec(model=tmp_path / 'a.gguf', device='Vulkan0', embedding=True,
        pooling_type=3, mmproj=tmp_path / 'p.gguf', mmproj_options={'warmup': False},
        type_k='q4_0', type_v='q4_0', runtime_options={'spec_type': 'none'})
    probe = smoke.probe_launch(original, SimpleNamespace(ctx=1024))
    argv = probe.command(tmp_path / 'key')
    assert probe.model == original.model and probe.device == original.device
    assert probe.n_ctx == probe.runtime_options['n_batch'] == probe.runtime_options['n_ubatch'] == 1024
    assert '--embeddings' not in argv and '--pooling' not in argv and '--mmproj' not in argv
    assert probe.type_k == probe.type_v == 'f16' and probe.flash_attn is False
    assert original.embedding and original.pooling_type == 3 and original.mmproj is not None


@pytest.mark.parametrize('kind,template,head,issue', [(None, True, True, 'decision_type'),
    ('unknown', True, True, 'decision_type'), ('laya', False, True, 'systemone_template'),
    ('laya', True, False, 'scorer'), ('laya', True, True, None)])
def test_inventory_reads_native_keys_and_scorer_descriptors(tmp_path, kind, template, head, issue):
    path = tmp_path / 'a.gguf'
    header(path, kind=kind, template=template, head=head)
    definition = tmp_path / 'a.toml'
    definition.write_text('[llm]\nmodel="a.gguf"\n')
    manager = SimpleNamespace(models_dir=tmp_path, _toml_path_for_reference=lambda _: definition,
        _resolve_virtual_model_spec=lambda _: SimpleNamespace(llm_model='a.gguf', mmproj_model=None))
    item = smoke.inventory(manager, 'a')
    assert item['header_candidate'] is (issue is None)
    assert issue is None or any(issue in s for s in item['issues'])
    assert item['metadata']['modern-bert.decision.max_head_tokens'] == 256
    assert item['vocab_size'] == 4 and (bool(item['tensors']) is head)


@pytest.mark.parametrize('mode,ok,review', [
    ('normal', True, False), ('fast', True, False), ('drift', True, True),
    ('no-decisions', False, True), ('bad-sum', False, True), ('bad-confidence', False, True),
    ('wrong-choice', False, True), ('wrong-score', False, True), ('wrong-legend', False, True),
    ('wrong-order', False, True), ('nan', False, True), ('no-usage', False, True), ('bad-usage', False, True),
])
def test_reference_errors_recovery_and_cleanup(tmp_path, payload, mode, ok, review):
    run_reference(tmp_path, payload, mode, ok, review)


@pytest.mark.parametrize('kind,mode,ok,review', [
    ('nimble', 'normal', True, False), ('clef', 'normal', True, False),
    ('nimble', 'fast', True, False), ('clef', 'fast', True, False),
    ('nimble', 'repeat-drift', True, True), ('clef', 'repeat-drift', True, True),
    ('nimble', 'usage-drift', True, True), ('clef', 'usage-drift', True, True),
    ('clef', 'permutation-drift', True, True),
    ('clef', 'wrong-option-order', False, True),
    ('nimble', 'wrong-many-keys', False, True), ('clef', 'wrong-many-keys', False, True),
])
def test_schema_reference_semantics_and_corruptions(tmp_path, payload, kind, mode, ok, review):
    run_reference(tmp_path, payload, mode, ok, review, kind=kind)


def run_reference(tmp_path, payload, mode, ok, review, *, kind='laya'):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(payload))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    header(models / 'a.gguf', kind=kind)
    definition = b'# Keep original.\n[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\n'
    (models / 'a.toml').write_bytes(definition)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--inventory-model', 'a',
               '--models-dir', str(models.parent), '--executable', str(executable), '--output', str(output)]
    environment = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_ROOT': str(root),
        'STUB_DECISION': mode, 'STUB_DECISION_KIND': kind,
        'STUB_BUILD': 'b11552-3144c167ef', 'STUB_PUBLIC_CTX': '1024'}
    memory = dict(payload, runtime_fingerprint='b11552-3144c167ef')
    (root / 'memory-report').write_text(json.dumps(memory))
    run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(s) for s in (output / 'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review, rows[-1]
    assert rows[-1]['public_api'] == 'not_exercised' and rows[-1]['residency'] == []
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert (models / 'a.toml').read_bytes() == definition
    closed = next(r for r in rows if r['stage'] == 'worker-closed')
    assert closed['clean'] and closed['directory_removed']
    if ok:
        checks = [r for r in rows if r['stage'] == 'comparison']
        assert len(checks) == {'laya': 8, 'nimble': 6, 'clef': 7}[kind]
        assert all(r['within_tolerance'] for r in checks) is (not review)
        assert len([r for r in rows if r['stage'] == 'decision']) == (15 if kind == 'laya' else 19)
        assert [r['status'] for r in rows if r['stage'] == 'native-error'] == [400] * 9 + [501]
        assert next(r for r in rows if r['stage'] == 'state-sensitivity')['distinct']
        cancel = next(r for r in rows if r['stage'] == 'transport-cancel')
        if mode != 'fast':
            assert cancel['pending_at_cancel']
        assert cancel['inference_started_observed'] is cancel['native_early_stop_observed'] is None
        assert next(r for r in rows if r['stage'] == 'recovery')['pid'] == closed['pid']
        contract = next(r for r in rows if r['stage'] == 'decision-contract')
        assert contract['decision_type'] == kind
        usage = next(r for r in rows if r['stage'] == 'usage-composition')
        assert usage['additive_required'] is (kind == 'laya')
        if kind != 'laya':
            assert usage['single_input_tokens'] != usage['combined_input_tokens']
            observations = [r for r in rows if r['stage'] == 'composition-observation']
            assert len(observations) == (5 if kind == 'nimble' else 4)
            assert any(not r['within_tolerance'] for r in observations)
            assert not any(r.get('probe') == 'native-laya-head-truncation' for r in rows)
            sent = {r['probe']: r['body'] for r in rows if r['stage'] == 'request'}
            assert len(sent['cancel']['questions']) == 2
            assert len(sent['twenty-seven-options']['questions']['many']['criteria']) == 27
            literal = next(r for r in rows if r['stage'] == 'native-response' and r['probe'] == 'literal-keys-unicode')
            keys = next(iter(literal['payload']['answers'].values()))['probabilities']
            expected_keys = list(next(iter(sent['literal-keys-unicode']['questions'].values()))['criteria'])
            assert list(keys) == (sorted(expected_keys) if kind == 'clef' else expected_keys)
    launches = (root / 'launch').read_bytes()
    again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert again.returncode == 2 and (root / 'launch').read_bytes() == launches


@pytest.mark.parametrize('kind,inventory_only', [(None, True), (None, False),
                                               ('pplx-decider', False), ('kev', False)])
def test_inventory_and_unsupported_references_start_no_worker(tmp_path, kind, inventory_only):
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    header(models / 'a.gguf', kind=kind, head=False)
    (models / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n')
    output = tmp_path / 'results'
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')}
    run = subprocess.run([sys.executable, str(SCRIPT), *(['--inventory-only'] if inventory_only else []),
        '--model', 'a', '--inventory-model', 'a',
        '--models-dir', str(models.parent), '--executable', str(tmp_path / 'missing'), '--output', str(output)],
        env=env, capture_output=True, text=True, timeout=10)
    assert run.returncode == (0 if inventory_only else 1), run.stdout + run.stderr
    rows = [json.loads(s) for s in (output / 'raw.jsonl').read_text().splitlines()]
    assert next(r for r in rows if r['stage'] == 'inventory')['header_candidate'] is (kind is not None)
    assert not any(r['stage'] in {'diagnostic-overrides', 'ready', 'worker-closed'} for r in rows)
    assert rows[-1]['inventory_only'] is inventory_only and rows[-1]['residency'] == []
    if not inventory_only:
        assert 'Reference requires' in next(r for r in rows if r['stage'] == 'error')['error']


def test_repeat_usage_requires_exact_counts_even_with_probability_tolerance():
    rows, notes = [], []
    smoke.compare({'usage': {'input_tokens': 10000, 'output_tokens': 0}},
                  {'usage': {'input_tokens': 10001, 'output_tokens': 0}},
                  SimpleNamespace(atol=1e-4, rtol=1e-3),
                  lambda stage, **values: rows.append(values), notes, 'repeat')
    assert not rows[0]['within_tolerance'] and notes == ['repeat:difference']


@pytest.mark.parametrize('option,value', [('--ctx','256'), ('--ctx','99999'), ('--atol','nan'),
    ('--rtol','inf'), ('--cancel-after-ms','0')])
def test_bad_cli_is_refused_before_output_or_process(tmp_path, option, value):
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(tmp_path / 'out'), option, value],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not (tmp_path / 'out').exists()
