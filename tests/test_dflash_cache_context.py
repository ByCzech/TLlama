"""DFlash cache evidence and existing context controls; no native GPU oracle."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_dflash_qualification_smoke import SCRIPT, BUILD, arguments, draft_inventory
from test_model_create import gguf_file
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_dflash_tools_structured import dflash, measured, launch
from test_dflash_vision_fim import dflash_vision, dflash_fim
from test_native_vision import vision, body as image_body
from test_native_fim import fim, body as fim_body, completions
from test_native_manager import native_manager, records
from test_native_public_api import public, request_body
from test_repository_worker_pins import protected_groups
from vision_fixtures import RED_BLUE


@pytest.mark.parametrize('mode,expected', [('normal', None), ('stale-state', 'review'),
    ('wrong-finish', 'review'), ('wrong-count', 'review'), ('no-reuse', 'review'),
    ('unknown-cached', 'cache-known'), ('reused-cold', 'cache-disabled')])
def test_state_capture_checks_each_cached_path_and_keeps_failure_evidence(tmp_path, draft_inventory, mode, expected):
    root = tmp_path / 'stub'
    root.mkdir()
    (root / 'memory-report').write_text(json.dumps(draft_inventory))
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + SCRIPT.parents[1].joinpath('tests/native_worker_stub.py').read_text())
    executable.chmod(0o700)
    gguf_file(tmp_path / 'model.gguf')
    draft = gguf_file(tmp_path / 'draft.gguf', arch='dflash')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('TLLAMA_', 'LLAMA_ARG_', 'STUB_'))}
    env.update(STUB_ROOT=str(root), STUB_BUILD=BUILD, STUB_DFLASH_QUAL='normal',
               STUB_MTP_STATE=mode, PYTHONPATH=str(SCRIPT.parents[1] / 'src'))
    reply = subprocess.run([sys.executable, str(SCRIPT), *arguments(tmp_path, '--state-only',
        '--draft-model', str(draft), '--executable', str(executable), '--transport', 'tcp',
        '--load-timeout', '15', '--request-timeout', '5')], env=env, capture_output=True, text=True, timeout=60)
    output = tmp_path / 'capture'
    result = json.loads((output / 'result.json').read_text())
    valid = expected in (None, 'review')
    assert result['ok'] is valid and result['protocol_ok'] is valid, (result, reply.stderr)
    assert reply.returncode == (0 if valid else 1)
    assert result['mode'] == 'state' and result['review_required']
    assert 'native_cache_rollback' in result['manual_review'] and result['public_api'] == 'not_exercised'
    assert bool(result['observations']) is (expected == 'review')
    if not valid:
        assert expected in result['error'] and not all(c['ok'] for c in result['checks'])
    summary = [json.loads(l) for l in (output / 'summary.jsonl').read_text().splitlines()]
    raw = [json.loads(l) for l in (output / 'raw.jsonl').read_text().splitlines()]
    requests = [r for r in raw if r['stage'] == 'request']
    assert any(r['probe'] == 'base-cold' and r['body']['cache_prompt'] is False for r in requests)
    assert any(r['probe'] == 'branch-cached' and r['body']['cache_prompt'] is True for r in requests)
    assert any(r['probe'] == 'extension-cached' and len(r['body']['messages']) == 4 for r in requests)
    assert (output / 'off-load.log').exists()
    assert all(c['ok'] for c in result['checks'] if c['stage'].endswith(':cleanup'))
    if valid:
        assert (output / 'dflash-load.log').exists()
        assert len([c for c in result['checks'] if c['stage'].endswith(':cleanup')]) == 2
        assert len([r for r in summary if r['stage'] == 'cancel-after-content']) == 2
        for profile in ('off', 'dflash'):
            p = [r for r in requests if r['profile'] == profile]
            assert any(r['probe'] == 'cancel' and r['body']['cache_prompt'] is True for r in p)
            assert any(r['probe'] == 'after-cancel' and r['body']['cache_prompt'] for r in p)
            assert any(r['probe'] == 'after-cancel-cold' and r['body']['cache_prompt'] is False for r in p)
        assert all('mtp_observed' not in r.get('metrics', {}) for r in summary)
        evidence = [r for r in summary if r['stage'] == 'state-evidence']
        assert len(evidence) == 2 and all(r['measured_prefix_reuse'] is (mode != 'no-reuse') for r in evidence)


@pytest.mark.parametrize('path,raw', [('/api/chat', False), ('/api/generate', False),
                                    ('/api/generate', True), ('/v1/chat/completions', False)])
@pytest.mark.parametrize('stream', [False, True])
async def test_public_repeat_preserves_native_cache_defaults_and_request_counters(dflash, path, raw, stream):
    manager, client, name, _ = dflash
    body = request_body(name, path, stream)
    if raw:
        body['raw'] = True
    for _ in range(2):
        values = measured(await client.post(path, json=body), path, stream)
        terminal = next(v for v in values if v.get('tllama'))
        if not path.startswith('/v1/'):
            assert terminal['prompt_eval_cached_count'] == 19
        else:
            assert terminal['choices'][0]['finish_reason'] == 'stop'
    assert len(manager.test.ledger) == 1
    sent = [r['body'] for r in records(manager.test.roots[0], 'requests')
            if r['path'] in {'/completion', '/v1/chat/completions'}]
    assert len(sent) == 2 and sent[0] == sent[1]
    assert all('cache_prompt' not in b for b in sent)  # Public API inherits native cache policy.
    launch(manager)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_context_shift_changes_reload_only_effective_launch(dflash, path, stream):
    manager, client, name, _ = dflash
    body = request_body(name, path, stream)
    steps = [({}, 1, False), ({'num_keep': 0}, 2, True), ({'num_keep': 4}, 2, True),
             ({'num_keep': -1}, 2, True), ({'num_keep': 4, 'num_predict': -2}, 3, False),
             ({'num_keep': 1, 'num_predict': -2}, 3, False), ({'num_keep': 1}, 4, True)]
    for options, count, shift in steps:
        body['options'] = options
        measured(await client.post(path, json=body), path, stream)
        assert len(manager.test.ledger) == count
        args = manager.test.ledger[-1].command
        assert ('--context-shift' in args) is shift and ('--no-context-shift' in args) is not shift
        sent = records(manager.test.roots[-1], 'requests')[-1]['body']
        if 'num_keep' in options:
            assert sent['n_keep'] == options['num_keep']
        if options.get('num_predict') == -2:
            assert sent['n_predict'] == -1
        launch(manager)


async def test_long_chat_removes_the_whole_old_tool_exchange_with_dflash(dflash):
    manager, client, name, _ = dflash
    messages = [{'role': 'system', 'content': 'SYSTEM'}, {'role': 'user', 'content': 'x' * 2500},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'q': 'old'}}}]},
        {'role': 'tool', 'content': 'old result', 'tool_name': 'lookup'},
        {'role': 'assistant', 'content': 'old answer'}, {'role': 'user', 'content': 'current question'}]
    measured(await client.post('/api/chat', json={'model': name, 'messages': messages,
        'options': {'num_keep': 4}, 'stream': False}), '/api/chat', False)
    sent = records(manager.test.roots[0], 'requests')[-1]['body']
    assert sent['messages'] == [messages[0], messages[-1]] and sent['n_keep'] == 4
    launch(manager)


async def test_oversized_latest_turn_keeps_worker_and_releases_lease(dflash):
    manager, client, name, _ = dflash
    first = await manager.preload_native_model(name, runtime_overrides={'context_shift': True})
    response = await client.post('/api/chat', json={'model': name,
        'messages': [{'role': 'user', 'content': 'x' * 2500}], 'options': {'num_keep': -1}, 'stream': False})
    assert response.status_code == 400 and 'latest chat turn' in response.text
    assert '"speculative"' not in response.text
    assert manager.native_snapshot()[0]['generation'] == first and len(manager.test.ledger) == 1
    assert not any(r['path'] == '/v1/chat/completions' for r in records(manager.test.roots[0], 'requests'))
    launch(manager)


@pytest.mark.parametrize('stream', [False, True])
async def test_long_raw_completion_retains_bos_prefix_and_recent_tail(dflash, stream):
    manager, client, name, _ = dflash
    prompt = 'PREFIX' + 'x' * 2500 + 'SUFFIX'
    measured(await client.post('/api/generate', json={'model': name, 'prompt': prompt, 'raw': True,
        'options': {'num_keep': 6}, 'stream': stream}), '/api/generate', stream)
    sent = records(manager.test.roots[0], 'requests')[-1]['body']
    assert sent['prompt'][0] == 1 and len(sent['prompt']) < 2048
    text = ''.join(chr(t - 10) for t in sent['prompt'][1:])
    assert text.startswith('PREFIX') and text.endswith('SUFFIX')
    launch(manager)


@pytest.mark.parametrize('stream', [False, True])
async def test_fim_context_limit_is_not_silently_truncated_by_num_keep(dflash_fim, stream):
    manager, client, name, _, _, _, group = dflash_fim
    response = await client.post('/api/generate', json=fim_body(name, prompt='p' * 2500,
        options={'num_ctx': 8192, 'num_keep': 0}, stream=stream))
    assert response.status_code == 400 and 'effective context is 2048' in response.text and 'not truncated' in response.text
    assert not completions(manager) and protected_groups(manager) == {group}
    launch(manager)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_multimodal_shift_refusal_does_not_retire_resident_helpers(dflash_vision, path):
    manager, client, name, _, groups = dflash_vision
    first = await manager.preload_native_model(name)
    body = (image_body(name, path) if path == '/api/chat' else
            {'model': name, 'prompt': 'describe', 'images': [RED_BLUE], 'stream': False})
    body['options'] = {'num_keep': 1}
    response = await client.post(path, json=body)
    assert response.status_code == 400 and 'multimodal' in response.text
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['generation'] == first
    assert protected_groups(manager) == groups and manager.native_snapshot()[0]['leases'] == 0
