"""Public suffix uses complete literal code over owned native subprocesses."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import gguf
import httpx
import pytest

from tllama.helpers.gguf_metadata import build_model_metadata_payload, read_gguf_metadata
from tllama.helpers.metadata_cache import SCHEMA_VERSION
from tllama.helpers.model_toml import parse_model_toml, TomlModelError
from tllama.main import app
from tllama.native_fim import prepare_fim
from tllama.native_routes import lease_metadata, serve_native
from tllama.routers import ollama
from tllama.schemas.ollama import OllamaGenerateRequest

from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public


# Match the public stub's compact vocabulary, independent of the GGUF reader.
VOCAB = ['unused', '[PRE]', '[SUF]', '[MID]', '[REP]', '[SEP]', '<bos>',
         '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>', '<|file_separator|>',
         '<fim_prefix>', '<fim_suffix>', '<fim_middle>', '<file_sep>',
         '[PREFIX]', '[SUFFIX]', '[MIDDLE]', '<｜fim▁begin｜>', '<｜fim▁hole｜>', '<｜fim▁end｜>',
         '<|file_sep|>', '<|repo_name|>', '<|fim_pad|>']
FAMILIES = {'codegemma': (7, 8, 9, 10), 'starcoder': (11, 12, 13, 14),
            'deepseek': (18,19,20), 'qwen': (7,8,9,21,22,23),
            'codestral': (15, 16, 17), 'negative': ()}


def write_vocab(path, family='codegemma', duplicate=False):
    tokens = [piece if i in FAMILIES[family] or i == 6 else f'unused-{i}'
              for i, piece in enumerate(VOCAB)]
    if duplicate:
        tokens.append(tokens[7])
    writer = gguf.GGUFWriter(str(path), 'gemma')
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))  # Includes an ordinary file delimiter.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


@pytest.fixture
async def fim(native_manager, monkeypatch):
    manager = native_manager()
    name, toml = manager.test.define(extra='[template]\njinja="SHOULD NOT RENDER {{ Prompt }}"\n'
        '[system]\nprompt="SHOULD NOT APPLY"\n[sampling]\ntemperature=0.4\nstop=["MODEL STOP"]\n')
    weights = manager.local_models_dir / 'a.gguf'
    write_vocab(weights)
    settings = {'family': 'codegemma', 'mode': 'codegemma-split'}
    original = manager._native._factory

    def factory(spec):
        return original(replace(spec, environment={**spec.environment, 'STUB_FIM': settings['mode'],
                                                  'STUB_PUBLIC_FIM': '1'}))
    manager._native._factory = factory
    monkeypatch.setattr(ollama, 'model_manager', manager)
    await manager.start()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://public') as client:
        yield manager, client, name, toml, weights, settings


def body(name, **extra):
    return {'model': name, 'prompt': 'def combine(a, b):\n    return ',
            'suffix': '\n\nassert combine(7, 2) == 14\n', 'stream': False, **extra}


def completions(manager):
    return [call['body'] for root in manager.test.roots for call in records(root, 'requests')
            if call['path'] == '/completion']


def frames(response, stream):
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('family', ['codegemma', 'starcoder'])
async def test_public_suffix_full_tokens_stops_sampling_and_stats(fim, stream, family):
    manager, client, name, _, weights, settings = fim
    write_vocab(weights, family)
    settings['mode'] = 'codegemma-split' if family == 'codegemma' else family
    prefix = '# no synthetic headers\n' + 'p' * 600 + '\nreturn '
    suffix = '\n' + 's' * 300 + '\nassert combine(7, 2) == 14\n'
    response = await client.post('/api/generate', json=body(name, stream=stream,
        prompt=prefix, suffix=suffix, options={'num_ctx': 8192, 'num_predict': 32,
                                             'temperature': 0, 'stop': ['CLIENT STOP']}))
    result = frames(response, stream)
    assert ''.join(frame['response'] for frame in result) == 'a * b'
    assert result[-1]['done_reason'] == 'stop' and result[-1]['model'] == name
    assert sum(frame['done'] for frame in result) == 1 and result[-1]['context'] == []
    pre, suf, mid, file = FAMILIES[family]
    expected = [6, pre] + [ord(c) + 1000 for c in prefix] + [suf] + [ord(c) + 1000 for c in suffix] + [mid]
    [sent] = completions(manager)
    assert sent['prompt'] == expected and len(expected) > 512
    assert result[-1]['prompt_eval_count'] == len(expected)
    assert sent['temperature'] == 0 and sent['n_predict'] == 32
    assert sent['stop'] == ['CLIENT STOP', *(VOCAB[i] for i in (pre, suf, mid, file))]
    assert sent['preserved_tokens'] == [VOCAB[i] for i in (pre, suf, mid, file)]
    assert not any(call['path'] == '/infill' for call in records(manager.test.roots[0], 'requests'))
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('order,expected', [('auto', [6,7,1120,8,1121,9]),
    ('psm', [6,7,1120,8,1121,9]), ('training_spm', [6,7,8,1121,9,1120]),
    ('native_spm', [6,8,1121,7,1120,9])])
async def test_toml_selects_explicit_order_without_native_infill(fim, order, expected):
    manager, client, name, toml, _, _ = fim
    toml.write_text(toml.read_text() + f'\n[fim]\norder="{order}"\n')
    response = await client.post('/api/generate', json=body(name, prompt='x', suffix='y'))
    assert response.status_code == 200
    assert completions(manager)[0]['prompt'] == expected
    assert completions(manager)[0]['stop'][0] == 'MODEL STOP'


async def test_empty_prefix_is_inference_and_keep_alive_zero_closes(fim):
    manager, client, name, _, _, _ = fim
    response = await client.post('/api/generate', json=body(name, prompt='', suffix='y', keep_alive=0))
    assert response.status_code == 200 and response.json()['done_reason'] == 'stop'
    assert len(completions(manager)) == 1
    await wait_until(lambda: not manager.native_snapshot())
    assert not manager.test.ledger[0].runtime_directory.exists()


@pytest.mark.parametrize('extra', [{'raw':True}, {'template':''}, {'system':''}, {'think':True},
    {'think':'high'}, {'images':['not decoded']}, {'context':[1]}, {'options':{'tllama':{'foo':'bar'}}}])
async def test_fim_incompatible_inputs_fail_before_loading(public, extra):
    manager, client, name, _ = public
    response = await client.post('/api/generate', json=body(name, **extra))
    assert response.status_code == 400 and 'error' in response.json()
    assert not manager.test.ledger


@pytest.mark.parametrize('family,disabled', [('negative', False), ('codestral',False), ('codegemma',True)])
@pytest.mark.parametrize('stream', [False, True])
async def test_unsupported_recipe_is_501_before_inference_and_worker_stays_usable(fim, family, disabled, stream):
    manager, client, name, toml, weights, settings = fim
    write_vocab(weights, family)
    settings['mode'] = family
    if disabled:
        toml.write_text(toml.read_text() + '\n[fim]\norder="disabled"\n')
    response = await client.post('/api/generate', json=body(name, stream=stream))
    assert response.status_code == 501 and 'FIM' in response.json()['error']
    assert not completions(manager) and manager.native_snapshot()[0]['leases'] == 0
    response = await client.post('/api/generate', json={'model':name, 'prompt':'hello','raw':True,'stream':False})
    assert response.status_code == 200 and len(completions(manager)) == 1
    assert len(manager.test.ledger) == 1


async def test_exact_effective_context_boundary_never_silently_clips(fim):
    manager, client, name, _, _, _ = fim
    for count, status in [(2042,200), (2043,400), (2100,400)]:
        response = await client.post('/api/generate', json=body(name, prompt='x'*count, suffix='y',
            options={'num_ctx':8192, 'num_predict':8}))
        assert response.status_code == status, response.text
        if status == 400:
            assert 'effective context is 2048' in response.json()['error']
    assert len(completions(manager)) == 1 and len(completions(manager)[0]['prompt']) == 2047
    assert manager.native_snapshot()[0]['leases'] == 0


async def test_literal_control_spellings_and_reasoning_markers_are_code(fim):
    manager, client, name, _, _, _ = fim
    await manager.preload_native_model(name)
    (manager.test.roots[0] / 'fim-content').write_text('<think>literal code</think>')
    prefix = 'x = "<|fim_prefix|>"\n'
    response = await client.post('/api/generate', json=body(name, prompt=prefix))
    assert response.status_code == 200 and response.json()['response'] == '<think>literal code</think>'
    sent = completions(manager)[0]['prompt']
    assert sent[2:2+len(prefix)] == [ord(c)+1000 for c in prefix]


async def test_fim_uses_generation_metadata_and_order_snapshot(fim):
    manager, _, name, toml, weights, _ = fim
    async with manager.acquire_native_model(name) as lease:
        toml.write_text(toml.read_text() + '\n[fim]\norder="disabled"\n')
        weights.write_bytes(b'replaced after generation load')
        prepared = await prepare_fim(OllamaGenerateRequest(**body(name)), lease, lease_metadata(lease))
        assert prepared.body()['prompt'][1] == 7
    assert len(manager.test.ledger) == 1


async def test_live_token_piece_mismatch_fails_without_completion(fim):
    manager, client, name, _, _, settings = fim
    settings['mode'] = 'codegemma-bad-piece'
    response = await client.post('/api/generate', json=body(name))
    assert response.status_code == 500 and 'disagree' in response.json()['error']
    assert not completions(manager) and manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('family', ['qwen', 'deepseek'])
async def test_additional_verified_vocabularies_are_detected_without_model_names(fim, family):
    manager, client, name, _, weights, settings = fim
    write_vocab(weights, family)
    settings['mode'] = family
    response = await client.post('/api/generate', json=body(name))
    assert response.status_code == 200 and response.json()['response'] == 'a * b'
    assert completions(manager)[0]['prompt'][1] == FAMILIES[family][0]


@pytest.mark.parametrize('delimiter', [7,8,9,10])
@pytest.mark.parametrize('stream', [False, True])
async def test_all_fim_boundaries_stop_in_native_engine_including_normal_token(fim, delimiter, stream):
    manager, client, name, toml, _, _ = fim
    toml.write_text(toml.read_text() + f'\n[sampling.logit_bias]\n"{delimiter}"=1000\n')
    response = await client.post('/api/generate', json=body(name, stream=stream))
    result = frames(response, stream)
    assert ''.join(frame['response'] for frame in result) == ''
    assert result[-1]['done_reason'] == 'stop' and result[-1]['eval_count'] == 1


async def test_legacy_native_spm_flag_requires_explicit_public_order(fim):
    manager, client, name, toml, _, _ = fim
    toml.write_text(toml.read_text() + '\n[runtime]\nspm_infill=true\n')
    response = await client.post('/api/generate', json=body(name))
    assert response.status_code == 400 and '[fim].order' in response.json()['error']
    assert not completions(manager)


async def test_json_schema_and_output_limit_preserve_generate_mapping(fim):
    manager, client, name, _, _, _ = fim
    schema = {'type':'string'}
    response = await client.post('/api/generate', json=body(name, format=schema, options={'num_predict':7}))
    assert response.status_code == 200
    assert completions(manager)[0]['json_schema'] == schema and completions(manager)[0]['n_predict'] == 7


async def test_public_stream_disconnect_releases_lease_and_worker_reuse(fim):
    manager, _, name, _, _, _ = fim
    await manager.preload_native_model(name)
    worker = manager.test.ledger[0]
    stream = worker.completion_stream
    worker.completion_stream = lambda payload: stream({**payload, 'stub':'hold'})
    response = await serve_native(manager, OllamaGenerateRequest(**body(name, stream=True)), api='ollama', kind='completion')
    received = asyncio.Event()
    async def send(message):
        if message['type'] == 'http.response.body':
            received.set()
    async def receive():
        await received.wait()
        return {'type':'http.disconnect'}
    await asyncio.wait_for(response({'type':'http','asgi':{'spec_version':'2.4'}}, receive, send), 5)
    await wait_until(lambda: (manager.test.roots[0] / 'cancelled').exists())
    assert manager.native_snapshot()[0]['leases'] == 0 and worker.process.returncode is None
    worker.completion_stream = stream
    result = await serve_native(manager, OllamaGenerateRequest(**body(name)), api='ollama', kind='completion')
    assert result['response'] == 'a * b' and len(manager.test.ledger) == 1


@pytest.mark.parametrize('order', ['wrong', 1, [], {}])
def test_invalid_toml_fim_order_is_not_silently_ignored(order):
    import tomlkit
    doc = {'llm':{'model':'Local/a.gguf'}, 'fim':{'order':order}}
    with pytest.raises(TomlModelError, match='order'):
        parse_model_toml(tomlkit.dumps(doc))


def test_fim_metadata_is_small_exact_and_rejects_duplicate_vocabulary(tmp_path):
    path = tmp_path/'m.gguf'
    write_vocab(path)
    metadata = build_model_metadata_payload(read_gguf_metadata(str(path)))
    assert metadata['fim']['tokens']['<|file_separator|>'] == 10
    assert set(metadata['fim']['tokens'].values()) == {7,8,9,10}
    assert 'tokenizer.ggml.tokens' not in metadata['metadata_raw'] and 'fim' not in metadata['metadata_raw']
    assert SCHEMA_VERSION >= 3  # FIM inventory requires at least version 3.
    write_vocab(path, duplicate=True)
    assert build_model_metadata_payload(read_gguf_metadata(str(path)))['fim'] is None


@pytest.mark.parametrize('family', ['codegemma','negative'])
def test_public_fim_smoke_cli_uses_owned_tcp_server_and_preserves_definition(tmp_path, family):
    import os
    import subprocess
    import sys
    models = tmp_path/'models'/'Local'
    models.mkdir(parents=True)
    weights = models/'a.gguf'
    write_vocab(weights, family)
    definition = '[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\n'
    (models/'a.toml').write_text(definition)
    stub = tmp_path/'native-stub'
    stub.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    stub.chmod(0o700)
    root = tmp_path/'worker'
    root.mkdir()
    output = tmp_path/'output'
    script = Path(__file__).parents[1]/'scripts'/'fim-api-smoke.py'
    result = subprocess.run([sys.executable,str(script),
        '--unsupported-model' if family == 'negative' else '--model','a',
        '--models-dir',str(models.parent),'--executable',str(stub),'--output',str(output),
        '--load-timeout','5','--request-timeout','5'], capture_output=True, text=True,timeout=40,
        env={**os.environ,'PYTHONPATH':str(script.parents[1]/'src'),'STUB_FIM':family,
             'STUB_PUBLIC_FIM':'1','STUB_PUBLIC_CTX':'4096','STUB_ROOT':str(root)})
    assert result.returncode == 0, result.stdout + result.stderr
    values = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert values[-1] == {'stage':'manager-closed','residency':[]}
    assert next(value for value in values if value['stage'] == 'finished')['transport_ok']
    assert next(value for value in values if value['stage'] == 'worker-closed')['clean']
    assert (output/'results.jsonl').read_text() == result.stdout
    assert (models/'a.toml').read_text() == definition
    assert len(records(root,'launch')) == 1
