"""Exercise the actual public/SDK diagnostic over owned TCP native stubs."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import gguf
import pytest

from tllama.helpers.model_toml import parse_model_toml
from tllama.worker import WorkerError
from test_native_memory_api import payload


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'embeddings-api-smoke.py'
SPEC = importlib.util.spec_from_file_location('embeddings_public_diagnostic', SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_temporary_definition_is_reviewable_and_does_not_reuse_generation_settings():
    source = parse_model_toml('''[llm]
model="Local/a.gguf"
[runtime]
device="Vulkan0"
n_ctx=8192
type_k="q8_0"
spec_type="draft-mtp"
draft_n_max=7
spm_infill=true
[sampling]
temperature=0.5
[system]
prompt="generation prompt"
''')
    text = smoke.diagnostic_definition(source, 256)
    actual = parse_model_toml(text)
    assert actual.llm_model == source.llm_model and actual.runtime['device'] == 'Vulkan0'
    assert actual.runtime['embedding'] is True and actual.runtime['pooling_type'] == -1
    assert actual.runtime['n_ctx'] == 256
    assert actual.runtime['n_batch'] == actual.runtime['n_ubatch'] == 512
    assert actual.runtime['type_k'] == actual.runtime['type_v'] == 'f16'
    assert actual.runtime['spec_type'] == 'none' and 'draft_n_max' not in actual.runtime
    assert not actual.sampling and actual.system_prompt is None and actual.template is None
    assert source.runtime['draft_n_max'] == 7 and source.runtime['n_ctx'] == 8192


def encoded(value):
    return {'object': 'list', 'data': [{'object': 'embedding', 'index': 0, 'embedding': value}]}


def test_base64_oracle_decodes_little_endian_float32_without_normalizing():
    value = base64.b64encode(struct.pack('<2f', 3, 4)).decode()
    vectors, norms = smoke.decode_vectors(encoded(value), '/v1/embeddings', 1, 2, encoded=True)
    assert vectors == [[3, 4]] and norms == [5]


@pytest.mark.parametrize('value', ['not base64!', base64.b64encode(b'123').decode(),
                                 base64.b64encode(struct.pack('<2f', float('nan'), 1)).decode()])
def test_encoded_corruption_does_not_pass_the_diagnostic(value):
    with pytest.raises((ValueError, WorkerError)):
        smoke.decode_vectors(encoded(value), '/v1/embeddings', 1, 2, encoded=True)


def test_missing_sdk_stops_before_creating_output_or_loading(monkeypatch, tmp_path):
    def missing():
        raise WorkerError('Missing optional SDK')
    monkeypatch.setattr(smoke, 'load_sdks', missing)
    output = tmp_path / 'results'
    with pytest.raises(SystemExit) as error:
        smoke.main(['--output', str(output)])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize('mode,ok,review,pooling,projected', [('normal', True, False, 1, False), ('fast', True, False, 1, False),
    ('drift', True, True, 1, False), ('no-usage', True, True, 1, False), ('bad-usage', False, True, 1, False),
    ('wrong-dimension', False, True, 1, False), ('normal', True, False, 3, False), ('fast', True, False, 1, True)])
def test_public_native_sdk_comparison_and_owned_cleanup(tmp_path, payload, mode, ok, review, pooling, projected):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(payload))
    models = tmp_path / 'models' / 'Local'
    models.mkdir(parents=True)
    writer = gguf.GGUFWriter(str(models / 'a.gguf'), arch='bert')
    writer.add_uint32('bert.pooling_type', pooling)
    writer.add_uint32('bert.embedding_length', 512 if projected else 8)
    if projected:
        writer.add_uint32('bert.embedding_length_out', 8)
    writer.add_uint32('bert.context_length', 256)
    writer.add_token_list([f'token{i}' for i in range(512)])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    original = '[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nspec_type="draft-mtp"\ndraft_n_max=7\n'
    (models / 'a.toml').write_text(original)
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output)]
    environment = {k: v for k, v in os.environ.items() if k.lower() not in {'http_proxy', 'https_proxy', 'all_proxy'}}
    environment.update(PYTHONPATH=str(SCRIPT.parents[1] / 'src'), STUB_EMBEDDINGS=mode,
                       STUB_ROOT=str(root), STUB_BUILD=payload['runtime_fingerprint'],
                       STUB_PUBLIC_EMBEDDINGS='1', STUB_PUBLIC_CTX='256',
                       STUB_EMBEDDING_POOLING=str(pooling),
                       TLLAMA_RUNTIME_DRAFT_N_MAX='3', TLLAMA_RUNTIME_SPEC_TYPE='draft-mtp')
    # Imports are optional for the project, mandatory for this hardware gate.
    try:
        smoke.load_sdks()
    except WorkerError:
        pytest.skip('Optional Python SDKs are unavailable')
    run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=40)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['protocol_ok'] is ok and rows[-1]['review_required'] is review, run.stdout
    assert run.stdout == (output / 'summary.jsonl').read_text()
    closed = [row for row in rows if row['stage'] == 'worker-closed']
    assert len(closed) == 1 and closed[0]['clean'] and closed[0]['directory_removed']
    assert next(row for row in rows if row['stage'] == 'model-finished')['residency'] == []
    assert (models / 'a.toml').read_text() == original
    assert not (models.parent / '.tllama').exists()  # Cache writes stay private.
    ready = next(row for row in rows if row['stage'] == 'ready')
    assert ready['runtime']['embedding'] is True
    assert ready['runtime']['n_ctx'] == 256 and ready['runtime']['spec_type'] == 'none'
    assert ready['pooling'] == pooling and ready['context_reserve'] == (pooling == 3)
    assert ready['token_limit'] == 256 - (pooling == 3)
    assert ready['dimension'] == 8 and ready['hidden_width'] == (512 if projected else 8)
    assert '--embeddings' in ready['argv'] and '--pooling' not in ready['argv']
    if ok:
        checks = [row for row in rows if row['stage'] == 'comparison']
        assert len(checks) >= 35 and all(r['within_tolerance'] for r in checks) is (mode != 'drift')
        assert next(row for row in rows if row['stage'] == 'unloaded')['residency'] == []
        assert next(row for row in rows if row['stage'] == 'recovery')['pid'] == ready['pid']
        cancellations = [row for row in rows if row['stage'] == 'transport-cancel']
        assert len(cancellations) == 3 and all(r['native_early_stop_observed'] is None for r in cancellations)
        if mode != 'fast':
            assert any(r['pending_at_cancel'] for r in cancellations)
        sdk_requests = [r for r in rows if r['stage'] == 'request' and r.get('source') == 'sdk']
        openai_requests = [r['body'] for r in sdk_requests if r['api'] == 'openai']
        assert openai_requests[0]['encoding_format'] == 'base64'
        assert any(r.get('encoding_format') == 'float' for r in openai_requests)
        assert any(r['path'] == '/api/embeddings' and r['api'] == 'ollama' for r in sdk_requests)
        native_requests = [r['body'] for r in rows if r['stage'] == 'request' and r.get('source') == 'native']
        assert any(r.get('embd_normalize') == -1 for r in native_requests)
        assert any(r.get('embd_normalize') == 2 for r in native_requests)
        truncation = next(row for row in rows if row['stage'] == 'truncation-reference')
        assert truncation['prefix'] == [1] and truncation['suffix'] == [2]
        assert len(truncation['submitted_tokens']) == ready['token_limit']
    launches = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert repeat.returncode == 2 and (root / 'launch').read_bytes() == launches


@pytest.mark.parametrize('option,value', [('--ctx', '1'), ('--ctx', '99999'), ('--atol', 'nan'),
                                         ('--request-timeout', '-1'), ('--cancel-after-ms', '0')])
def test_invalid_cli_has_no_output_or_worker(tmp_path, option, value):
    output = tmp_path / 'results'
    run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), option, value],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not output.exists()
