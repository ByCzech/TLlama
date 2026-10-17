"""Exercise diagnostic CLI/transport with an owned subprocess, not real FIM quality."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import gguf


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'fim-smoke.py'
spec = importlib.util.spec_from_file_location('fim_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize('mode,negative,ok', [
    ('normal', False, True), ('repo', False, True), ('semantic-bad', False, True),
    ('negative', True, True), ('missing-route', True, False), ('bad-terminal', False, False),
    ('normal', True, False), ('negative', False, False),
    ('codegemma', False, True), ('starcoder', False, True), ('ignored-stops', False, False),
    ('codestral', False, True), ('qwen', False, True),
    ('codegemma-split', False, True), ('codegemma-bad-piece', False, False),
])
def test_cli_reference_comparison_negative_errors_and_cleanup(tmp_path, mode, negative, ok):
    root = tmp_path / 'stub'
    root.mkdir()
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    (models / 'a.gguf').write_bytes(b'fake weights')
    (models / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_batch=64\nn_ctx=8192\nspm_infill=true\n')
    if mode == 'codestral':
        (models / 'Codestral-test.toml').write_text((models / 'a.toml').read_text())
    if mode in {'codegemma-split', 'codegemma-bad-piece'}:
        write_vocab(models / 'a.gguf', ['unused'] * 70 + ['<|file_separator|>'])
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--negative-model' if negative else '--model', 'Codestral-test' if mode == 'codestral' else 'a',
               '--models-dir', str(models.parent), '--executable', str(executable), '--output', str(output)]
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_FIM': mode,
           'STUB_ROOT': str(root), 'TLLAMA_MEMORY_SOURCE': 'log'}
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    records = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    summaries = [json.loads(line) for line in (output / 'summary.jsonl').read_text().splitlines()]
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert all('full_tokens' not in record and 'final' not in record for record in summaries)
    assert records[-1]['transport_ok'] is ok and records[-1]['public_suffix'] == 'not_exercised'
    assert records[-1]['review_required'] is (not ok or mode in {'semantic-bad', 'codestral'})
    closed = [record for record in records if record['stage'] == 'worker-closed']
    assert len(closed) == (1 if negative or mode == 'codestral' else 2)
    assert all(record['clean'] and record['directory_removed']
               and record['returncode'] in (0, -signal.SIGTERM) for record in closed)
    launches = [json.loads(line) for line in (root / 'launch').read_text().splitlines()]
    for index, launch in enumerate(launches):
        argv = launch['argv']
        # Diagnostic overrides win over TOML, without changing the stored TOML.
        assert argv[argv.index('--batch-size') + 1] == '512'
        assert argv[argv.index('--ctx-size') + 1] == '4096'
        assert ('--spm-infill' in argv) is (index == 1)
    if ok and not negative and mode != 'codestral':
        comparisons = [record for record in records if record['stage'] == 'comparison']
        assert len(comparisons) == 12
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
        assert all((r['native_equivalent_counts']['extra'] > 0) is (mode in {'repo', 'qwen'}) for r in prompts)
        checks = [r for r in records if r['stage'] == 'stop-check']
        assert len(checks) == (8 if mode in {'codegemma', 'codegemma-split', 'starcoder'} else 4)
        assert all(r['verified'] and r['content'] == '' for r in checks)
        training = [r for r in records if r['stage'] == 'training-spm-prompt']
        assert len(training) == 6
        assert all(r['order'] == 'spm' for r in training)
        policies = [r['tokens'] for r in records if r['stage'] == 'stop-policy']
        if mode == 'codegemma':
            assert all(p['FILE0']['id'] == 70 and p['FILE0']['encoding'] == [1032, 70]
                       for p in policies)
        if mode == 'codegemma-split':
            assert all(p['FILE0']['id'] == 70 and p['FILE0']['resolution'] == 'gguf_and_native_piece'
                       and p['FILE0']['encoding'] != [70] for p in policies)
        if mode == 'qwen':
            assert all('SEP' in p and 'FILE0' not in p for p in policies)
            requests = [json.loads(line) for line in (root / 'requests').read_text().splitlines()]
            assert not any(r['path'] == '/tokenize' and r['body']['content'] == '<|file_separator|>'
                           for r in requests)
        if mode in {'codegemma', 'codegemma-split', 'starcoder'}:
            baselines = [r for r in records if r['stage'] == 'unstopped-baseline']
            assert all('unwanted continuation' in r['content'] for r in baselines)
            assert all('unwanted continuation' not in r['content'] for r in results)
            assert all(r['final']['stop_type'] == 'word' for r in results)
            requests = [json.loads(line) for line in (root / 'requests').read_text().splitlines()]
            controlled = [r['body'] for r in requests if r['path'] in {'/infill', '/completion'} and r['body'].get('stop')]
            assert all(r['stop'] == r['preserved_tokens'] for r in controlled)
    if ok and mode == 'codestral':
        assert len([r for r in records if r['stage'] == 'native-fim-unassigned']) == 2
        results = [r for r in records if r['stage'] == 'completion']
        assert len(results) == 12 and all(r['semantic_match'] for r in results)
        prompts = [r for r in records if r['stage'] == 'prompts']
        assert all(r['reference'] == 'codestral-official' and 1100033 not in r['full_tokens'] for r in prompts)
        assert all(r['full_tokens'][:2] == [7, 1100032] for r in prompts)
        assert all(r['full_tokens'].index(1100032) < r['full_tokens'].index(1100031) for r in prompts)
        assert any(r['stage'] == 'after-cancel' for r in records)

    if ok and negative:
        assert len([r for r in records if r['stage'] == 'negative-rejected']) == 2
        assert any(r['stage'] == 'after-negative' for r in records)
    assert (output / 'worker.log').is_file()
    before = (root / 'launch').read_text()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert repeated.returncode != 0 and (root / 'launch').read_text() == before


@pytest.mark.parametrize('text,case,expected', [('a + b', 'add', True), ('b * a', 'multiply', True),
    ('a - b', 'add', False), ('__import__("os").system("anything")', 'add', False),
    ('a + b\nprint("extra")', 'add', False), ('```python\na+b\n```', 'add', False),
    ('violet', 'python-violet', True), ('amber', 'typescript-amber', True),
    ('amber', 'python-violet', False), ('violet; extra', 'typescript-violet', False)])
def test_semantic_probe_only_inspects_expected_expression(text, case, expected):
    assert smoke.semantic_check(text, case) is expected


@pytest.mark.parametrize('extra', [['--ctx', '0'], ['--predict', '0'], ['--batch', '8'],
                                  ['--request-timeout', 'nan']])
def test_invalid_cli_does_not_create_evidence_directory(tmp_path, extra):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), *extra],
                         capture_output=True, text=True, timeout=10)
    assert run.returncode != 0 and not output.exists()


def test_independent_fim_layouts_do_not_conflate_native_and_training_spm():
    inventory = {key: {'id': value} for key, value in {'PRE': 1, 'SUF': 2, 'MID': 3}.items()}
    assert smoke.layout_tokens(inventory, [10, 11], [20, 21], 'psm') == [1, 10, 11, 2, 20, 21, 3]
    assert smoke.layout_tokens(inventory, [10, 11], [20, 21], 'spm') == [2, 20, 21, 1, 10, 11, 3]
    assert smoke.layout_tokens(inventory, [10, 11], [20, 21], 'training-spm') == [1, 2, 20, 21, 3, 10, 11]


@pytest.mark.parametrize('entries', [
    [{'id': 8, 'piece': ' '}, {'id': 9, 'piece': '<file>'}],
    [{'id': 8, 'piece': 'x'}, {'id': 9, 'piece': '<file>'}],
    [{'id': 8, 'piece': '<fi'}, {'id': 9, 'piece': 'le>'}],
    [{'id': 8, 'piece': '<file>'}, {'id': 9, 'piece': '<file>'}],
    [{'id': 8, 'piece': ' '}, {'id': 77, 'piece': '<file>'}],
    [{'id': 8, 'piece': ''}, {'id': 9, 'piece': '<file>'}],
    [8, 9],
])
def test_delimiter_resolution_requires_exact_piece_and_only_whitespace_framing(entries):
    if entries == [{'id': 8, 'piece': ' '}, {'id': 9, 'piece': '<file>'}]:
        assert smoke.delimiter_id('<file>', [8, 9], entries) == 9
    else:
        with pytest.raises(smoke.WorkerError, match='Cannot resolve delimiter'):
            smoke.delimiter_id('<file>', [8, 9], entries)


def write_vocab(path, tokens):
    writer = gguf.GGUFWriter(path, 'gemma')
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


@pytest.mark.parametrize('tokens,expected', [
    (['x', '<file>', 'y'], 1), (['x', 'y'], None), (['<file>', '<file>'], None),
])
def test_gguf_delimiter_uses_exact_unique_vocabulary_entry_without_mutation(tmp_path, tokens, expected):
    path = tmp_path / 'vocab.gguf'
    write_vocab(path, tokens)
    before = path.read_bytes()
    if expected is None:
        with pytest.raises(smoke.WorkerError, match='Expected one exact GGUF'):
            smoke.gguf_delimiter(path, '<file>')
    else:
        assert smoke.gguf_delimiter(path, '<file>') == {'id': expected, 'vocab_size': len(tokens), 'gguf_token_type': 1}
    assert path.read_bytes() == before
