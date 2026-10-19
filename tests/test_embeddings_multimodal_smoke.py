"""Independent image-embedding harness through authenticated TCP subprocesses."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import gguf
import pytest

from tllama.worker import WorkerError
from tllama.worker_config import WorkerLaunchSpec
from test_native_memory_api import payload


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'embeddings-multimodal-smoke.py'
SPEC = importlib.util.spec_from_file_location('embeddings_multimodal_reference', SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def test_diagnostic_preserves_resolved_projector_and_production_guards(tmp_path):
    spec = WorkerLaunchSpec(model=tmp_path / 'a.gguf', device='none', n_ctx=262144,
                           mmproj=tmp_path / 'vision.gguf', mmproj_options={'warmup': False},
                           template='configured template', type_k='q4_0', type_v='q4_0',
                           flash_attn=True, runtime_options={'spec_type': 'draft-mtp', 'draft_n_max': 8})
    launch = smoke.probe_launch(spec, SimpleNamespace(ctx=1024))
    argv = launch.command(tmp_path / 'key')
    assert launch.mmproj == spec.mmproj and launch.mmproj_options == spec.mmproj_options
    assert argv.count('--mmproj') == 1 and '--no-mmproj-offload' in argv
    assert argv.count('--embeddings') == 1 and argv[-2:] == ('--pooling', 'last')
    assert argv.count('--spec-type') == 1 and argv[argv.index('--spec-type') + 1] == 'none'
    assert '--spec-draft-n-max' not in argv and '--chat-template-file' not in argv
    assert launch.n_ctx == 1024 and launch.runtime_options['n_ubatch'] == 1024
    assert launch.type_k == launch.type_v == 'f16' and launch.flash_attn is False
    assert spec.n_ctx == 262144 and spec.template == 'configured template'
    assert spec.runtime_options['spec_type'] == 'draft-mtp'
    production = WorkerLaunchSpec(model=spec.model, device='none', embedding=True, mmproj=spec.mmproj)
    assert '--mmproj' in production.command(tmp_path / 'key')
    with pytest.raises(ValueError, match='template'):
        WorkerLaunchSpec(model=spec.model, device='none', embedding=True, mmproj=spec.mmproj,
                         template='generation template')
    with pytest.raises(WorkerError, match='projector'):
        smoke.probe_launch(WorkerLaunchSpec(model=spec.model, device='none'), SimpleNamespace(ctx=1024))


@pytest.mark.parametrize('mode,ok,review', [
    ('normal', True, False), ('fast', True, False), ('ignore-images', True, True),
    ('drift', True, True), ('bad-norm', True, True), ('no-usage', True, True),
    ('wrong-dimension', False, True), ('bad-index', False, True), ('nan', False, True),
    ('no-vision', False, True), ('decoder-error', True, True),
])
@pytest.mark.parametrize('managed', [False, True])
def test_multimodal_reference_and_cleanup(tmp_path, payload, mode, ok, review, managed):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(payload))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    writer = gguf.GGUFWriter(models / 'a.gguf', 'qwen3vl')
    writer.add_name('Embedding fixture')
    writer.add_context_length(262144)
    writer.add_embedding_length(8)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    projector = gguf.GGUFWriter(models / 'vision.gguf', 'clip')
    projector.write_header_to_file()
    projector.write_kv_data_to_file()
    projector.close()
    definition = (b'# Registered model remains unchanged.\n[llm]\nmodel="Local/a.gguf"\n'
                  b'[runtime]\ndevice="none"\n[mmproj]\nmodel="Local/vision.gguf"\n')
    (models / 'a.toml').write_bytes(definition)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output)]
    if managed:
        command.append('--managed')
    environment = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'),
                   'STUB_MULTIMODAL_EMBEDDINGS': mode, 'STUB_ROOT': str(root),
                   'STUB_BUILD': payload['runtime_fingerprint'], 'STUB_PUBLIC_CTX': '1024',
                   'STUB_MODE': 'no-vision' if mode == 'no-vision' else 'normal'}
    run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review, rows[-1]
    assert rows[-1]['public_api'] == 'not_exercised' and rows[-1]['residency'] == []
    assert rows[0]['launch_path'] == ('production' if managed else 'independent_reference')
    if managed:
        probe = smoke.tomlkit.parse((output / 'managed-model.toml').read_text())
        assert probe['runtime']['embedding'] is True and probe['runtime']['pooling_type'] == 3
        assert probe['mmproj']['model'] == 'Local/vision.gguf'
        assert not [row for row in rows if row['stage'] == 'diagnostic-overrides']
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert (models / 'a.toml').read_bytes() == definition
    assert next(row for row in rows if row['stage'] == 'registered-definition' and row['phase'] == 'after')['unchanged']
    closed = next(row for row in rows if row['stage'] == 'worker-closed')
    assert closed['clean'] and closed['directory_removed']
    if ok:
        ready = next(row for row in rows if row['stage'] == 'ready')
        assert ready['dimension'] == 8 and ready['projector_path'] == str(models / 'vision.gguf')
        assert ready['argv'].count('--embeddings') == 1 and ready['argv'].count('--pooling') == 1
        assert next(row for row in rows if row['stage'] == 'recovery')['pid'] == ready['pid'] == closed['pid']
        checks = [row for row in rows if row['stage'] == 'comparison']
        assert len(checks) == 16
        assert all(row['within_tolerance'] for row in checks) is (mode != 'drift')
        sensitivity = [row for row in rows if row['stage'] == 'image-sensitivity']
        assert len(sensitivity) == 3 and all(row['distinct'] for row in sensitivity) is (mode != 'ignore-images')
        errors = [row for row in rows if row['stage'] == 'native-error']
        assert [row['status'] for row in errors] == [400, 400, 400, 500]
        limitations = [row for row in rows if row['stage'] == 'known-native-limitation']
        assert bool(limitations) is (mode != 'decoder-error')
        cancel = next(row for row in rows if row['stage'] == 'transport-cancel')
        assert cancel['pending_at_cancel'] is (mode != 'fast')
        assert cancel['native_early_stop_observed'] is cancel['inference_started_observed'] is None
        assert next(row for row in rows if row['stage'] == 'unloaded')['residency'] == []
        assert len(list(output.glob('image-*.png'))) == 2
    launches = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root / 'launch').read_bytes() == launches


@pytest.mark.parametrize('option,value', [('--ctx', '256'), ('--ctx', '99999'), ('--atol', 'nan'),
                                         ('--request-timeout', '-1'), ('--cancel-after-ms', '0')])
def test_invalid_cli_does_not_start_a_worker(tmp_path, option, value):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), option, value],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not output.exists()
