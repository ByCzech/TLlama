"""Independent embedding diagnostic over owned native-stub subprocesses."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import gguf

from tllama.worker import WorkerError
from tllama.worker_config import WorkerLaunchSpec
from test_native_memory_api import payload


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'embeddings-smoke.py'
SPEC = importlib.util.spec_from_file_location('embeddings_reference', SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def response(vectors):
    return {'object': 'list', 'data': [
        {'object': 'embedding', 'index': index, 'embedding': vector} for index, vector in enumerate(vectors)]}


@pytest.mark.parametrize('vectors', [[[True]], [[float('nan')]], [[float('inf')]], [[0, 0]],
                                     [[], []], [[1], [1, 0]]])
def test_invalid_vector_measurements_are_rejected(vectors):
    with pytest.raises(WorkerError):
        smoke.inspect_vectors(response(vectors), len(vectors))


def test_indices_and_dimensions_are_not_silently_repaired():
    data = response([[1, 0], [0, 1]])
    data['data'][1]['index'] = 0
    with pytest.raises(WorkerError, match='index'):
        smoke.inspect_vectors(data, 2)
    with pytest.raises(WorkerError, match='dimension'):
        smoke.inspect_vectors(response([[1, 0]]), 1, dimension=3)


def test_launch_is_explicitly_text_only_without_changing_production_spec(tmp_path):
    spec = WorkerLaunchSpec(model=tmp_path / 'weights.gguf', device='none', n_ctx=4096,
                            mmproj=tmp_path / 'mmproj.gguf', mmproj_options={'warmup': False},
                            type_k='q4_0', type_v='q4_0', flash_attn=True,
                            runtime_options={'spec_type': 'draft-mtp', 'draft_n_max': 8})
    launch = smoke.probe_launch(spec, SimpleNamespace(ctx=256, pooling='auto'))
    argv = launch.command(tmp_path / 'key')
    assert argv[-1] == '--embeddings' and '--pooling' not in argv
    assert argv.count('--spec-type') == 1 and argv[argv.index('--spec-type') + 1] == 'none'
    assert '--spec-draft-n-max' not in argv and '--mmproj' not in argv
    assert launch.type_k == launch.type_v == 'f16' and launch.flash_attn is False
    assert launch.runtime_options['n_batch'] == launch.runtime_options['n_ubatch'] == 512
    assert spec.mmproj is not None and spec.n_ctx == 4096 and spec.runtime_options['spec_type'] == 'draft-mtp'
    assert smoke.probe_launch(spec, SimpleNamespace(ctx=1024, pooling='last')).command(tmp_path / 'key')[-2:] == ('--pooling', 'last')


@pytest.mark.parametrize('mode,ok,review', [
    ('normal', True, False), ('fast', True, False), ('no-usage', True, True),
    ('drift', True, True), ('bad-norm', True, True), ('error500', True, True),
    ('wrong-dimension', False, True), ('bad-index', False, True),
    ('nan', False, True), ('http500', False, True),
])
def test_native_reference_and_cleanup(tmp_path, payload, mode, ok, review):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(payload))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    # Real metadata is frozen in ModelDefinition; exercise its serialization.
    writer = gguf.GGUFWriter(models / 'a.gguf', 'bert')
    writer.add_name('Embedding fixture')
    writer.add_context_length(256)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    definition = '[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nspec_type="draft-mtp"\n'
    (models / 'a.toml').write_text(definition)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output)]
    environment = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'),
                   'STUB_EMBEDDINGS': mode, 'STUB_ROOT': str(root),
                   'STUB_BUILD': payload['runtime_fingerprint'], 'STUB_PUBLIC_CTX': '256'}
    run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=20)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert next(row for row in rows if row['stage'] == 'ready')['metadata']['general.architecture'] == 'bert'
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review
    assert run.stdout == (output / 'summary.jsonl').read_text()
    closed = [row for row in rows if row['stage'] == 'worker-closed']
    assert len(closed) == 1 and closed[0]['clean'] and closed[0]['directory_removed']
    assert next(row for row in rows if row['stage'] == 'model-finished')['residency'] == []
    assert (models / 'a.toml').read_text() == definition
    if ok:
        assert next(row for row in rows if row['stage'] == 'recovery')['pid'] == closed[0]['pid']
        checks = [row for row in rows if row['stage'] == 'comparison']
        assert len(checks) == 5 and all(row['within_tolerance'] for row in checks) is (mode != 'drift')
        cancel = next(row for row in rows if row['stage'] == 'transport-cancel')
        assert cancel['native_early_stop_observed'] is None
        assert cancel['pending_at_cancel'] is (mode != 'fast')
        if mode != 'fast':
            assert (root / 'cancelled').read_text().splitlines() == ['"embeddings"']
    launches = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root / 'launch').read_bytes() == launches


@pytest.mark.parametrize('option,value', [('--ctx', '1'), ('--ctx', '99999'), ('--atol', 'nan'),
                                         ('--request-timeout', '-1'), ('--cancel-after-ms', '0')])
def test_invalid_cli_does_not_start_a_worker(tmp_path, option, value):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), option, value],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not output.exists()
