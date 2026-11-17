"""Explicit per-request strategy selection across REPL, HTTP and worker ownership."""

from pathlib import Path

import httpx
import pytest

from tllama.cli_repl import command
from tllama.cli_run import RunSession
from tllama.client import TLlamaClient
from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.main import app
from tllama.native_config import resolve_worker_config
from tllama.ollama_options import ollama_runtime_overrides
from test_helper_definitions import publish, reference
from test_native_manager import native_manager, records
from test_native_public_api import public, request_body


@pytest.mark.parametrize('configured', [None, 'none', 'draft-mtp', 'draft-dflash'])
@pytest.mark.parametrize('strategy', ['none', 'draft-mtp', 'draft-dflash'])
@pytest.mark.parametrize('length', [None, 0, 2])
def test_explicit_method_beats_configuration_and_never_falls_back(configured, strategy, length):
    text = '[llm]\nmodel="main.gguf"\n[runtime]\ndraft_n_max=5\n'
    if configured is not None:
        text += f'spec_type="{configured}"\n'
    spec = parse_model_toml(text)
    options = {'spec_type': strategy}
    if length is not None:
        options['draft_num_predict'] = length
    original = dict(options)
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}),
        Path('/models/main.gguf'), spec, draft=Path('/models/draft.gguf'),
        runtime_overrides=ollama_runtime_overrides(options), draft_metadata={'arch': 'dflash'})
    args = resolved.launch.command(Path('/private/key'))
    expected = 'none' if length == 0 else strategy
    assert args[args.index('--spec-type') + 1] == expected
    assert ('--spec-draft-model' in args) == (expected == 'draft-dflash')
    assert ('--spec-draft-n-max' in args) == (expected != 'none')
    if expected != 'none':
        assert args[args.index('--spec-draft-n-max') + 1] == str(5 if length is None else length)
    assert resolved.sources['spec_type'] == 'request'
    assert options == original and spec.document.as_string() == text


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_bad_methods_and_unapproved_load_options_preserve_resident(public, path):
    manager, client, name, _ = public
    first = await manager.preload_native_model(name)
    invalid = [None, True, 1, [], {}, '', 'DFLASH', 'draft-dflash2', 'draft-simple',
               'draft-mtp,draft-dflash']
    options = [{'spec_type': value} for value in invalid]
    options += [{'draft_n_max': 3}, {'draft_device': 'none'}, {'draft_n_gpu_layers': 0},
                {'draft_type_k': 'f16'}, {'draft_type_v': 'f16'}, {'draft_n_ubatch': 0}]
    for value in options:
        response = await client.post(path, json={'model': name, 'messages': [],
                                               'options': value, 'keep_alive': 0})
        assert response.status_code == 400, (value, response.text)
        assert manager.native_snapshot()[0]['generation'] == first
        assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode is None


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_method_switch_reloads_once_inherits_on_omission_and_keeps_definition(public, tmp_path, path, stream):
    manager, client, name, toml = public
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    original = toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="none"\n'
    toml.write_text(original)
    digest = manager._resolve_native_definition_sync(name, None).file_info['model_digest']
    steps = [({}, 'none', 1),
             ({'spec_type': 'draft-dflash'}, 'draft-dflash', 2),
             ({'spec_type': 'draft-dflash', 'draft_num_predict': 3}, 'draft-dflash', 2),
             ({'spec_type': 'draft-dflash', 'draft_num_predict': 0}, 'none', 3),
             ({'spec_type': 'draft-dflash', 'draft_num_predict': 2}, 'draft-dflash', 4),
             ({'spec_type': 'none', 'draft_num_predict': 2}, 'none', 5),
             ({'spec_type': 'none', 'draft_num_predict': 7}, 'none', 5),
             ({'spec_type': 'draft-mtp', 'draft_num_predict': 2}, 'draft-mtp', 6),
             ({}, 'none', 7)]
    for options, expected, count in steps:
        response = await client.post(path, json={**request_body(name, path, stream), 'options': options})
        assert response.status_code == 200, response.text
        assert len(manager.test.ledger) == count
        args = manager.test.ledger[-1].command
        assert args[args.index('--spec-type') + 1] == expected
        assert ('--spec-draft-model' in args) == (expected == 'draft-dflash')
        sent = [r['body'] for r in records(manager.test.roots[-1], 'requests')
                if r['path'] in {'/completion', '/v1/chat/completions'}][-1]
        assert not {'spec_type', 'draft_num_predict', 'draft_n_max'} & sent.keys()
    assert toml.read_text() == original
    assert manager._resolve_native_definition_sync(name, None).file_info['model_digest'] == digest
    assert all(w.process.returncode is not None for w in manager.test.ledger[:-1])


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_missing_draft_rejects_explicit_dflash_without_replacing_working_worker(public, path):
    manager, client, name, _ = public
    first = await manager.preload_native_model(name)
    response = await client.post(path, json={**request_body(name, path),
                                           'options': {'spec_type': 'draft-dflash'}})
    assert response.status_code == 400 and 'draft' in response.text
    assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode is None
    assert manager.native_snapshot()[0]['generation'] == first


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_empty_preload_applies_method_without_inference(public, path):
    manager, client, name, _ = public
    for _ in range(2):
        response = await client.post(path, json={'model': name, 'messages': [],
                                               'options': {'spec_type': 'draft-mtp'}})
        assert response.status_code == 200, response.text
    assert len(manager.test.ledger) == 1
    args = manager.test.ledger[0].command
    assert args[args.index('--spec-type') + 1] == 'draft-mtp'
    assert not any(r['path'] in {'/completion', '/v1/chat/completions'}
                   for r in records(manager.test.roots[0], 'requests'))


async def test_repl_selects_dflash_from_inactive_binding_and_keeps_zero_until_changed(public, tmp_path, capsys):
    manager, _, name, toml = public
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    original = toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="none"\n'
    toml.write_text(original)
    session = RunSession(name, 'completion')
    async with TLlamaClient('http://public', transport=httpx.ASGITransport(app=app)) as client:
        await command(client, session, '/set parameter draft_num_predict 0')
        await command(client, session, '/set parameter spec_type "draft-dflash"')
        assert not manager.test.ledger  # /set itself is local and does not load.
        await session.submit(client, 'off')
        assert '--spec-draft-model' not in manager.test.ledger[-1].command
        await command(client, session, '/set parameter draft_num_predict 3')
        for _ in range(2):
            await session.submit(client, 'on')
        assert len(manager.test.ledger) == 2
        assert '--spec-draft-model' in manager.test.ledger[-1].command
        await command(client, session, '/set parameter spec_type "none"')
        await session.submit(client, 'off again')
        assert len(manager.test.ledger) == 3
        assert '--spec-draft-model' not in manager.test.ledger[-1].command
        assert session.controls['options'] == {'spec_type': 'none', 'draft_num_predict': 3}
    assert toml.read_text() == original and session.messages
