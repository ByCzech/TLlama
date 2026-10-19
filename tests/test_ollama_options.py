"""Ollama options reach native requests or immutable worker generations."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import build_native_sampling, resolve_worker_config
from tllama.ollama_options import ollama_runtime_overrides

from test_native_public_api import public, request_body
from test_native_manager import native_manager, records


# The 20 JSON fields in Ollama api.Options (including embedded Runner).
CLIENT_OPTIONS = {
    'num_ctx': 1024, 'num_batch': 128, 'num_gpu': 0, 'main_gpu': 0,
    'use_mmap': False, 'num_thread': 2, 'draft_num_predict': 3,
    'num_keep': 4, 'seed': 7, 'num_predict': 8, 'top_k': 30, 'top_p': .8,
    'min_p': .1, 'typical_p': .9, 'repeat_last_n': 24, 'temperature': .4,
    'repeat_penalty': 1.2, 'presence_penalty': .3, 'frequency_penalty': .2, 'stop': ['END'],
}


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_all_twenty_client_fields_are_applied(public, path, stream):
    manager, client, name, _ = public
    body = request_body(name, path, stream)
    body['options'] = CLIENT_OPTIONS
    response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    worker = manager.test.ledger[0]
    args = worker.command
    for flag, value in {'--ctx-size': '1024', '--batch-size': '128', '--ubatch-size': '128',
                        '--n-gpu-layers': '0', '--device': 'none', '--main-gpu': '0',
                        '--threads': '2', '--threads-batch': '2', '--load-mode': 'none',
                        '--spec-type': 'draft-mtp', '--spec-draft-n-max': '3'}.items():
        assert args[args.index(flag) + 1] == value
    assert '--context-shift' in args and '--no-context-shift' not in args
    sent = [r['body'] for r in records(manager.test.roots[0], 'requests')
            if r['path'] in {'/completion', '/v1/chat/completions'}][-1]
    for key, value in CLIENT_OPTIONS.items():
        if key not in {'num_ctx', 'num_batch', 'num_gpu', 'main_gpu', 'use_mmap', 'num_thread', 'draft_num_predict'}:
            native = {'num_predict': 'n_predict', 'num_keep': 'n_keep'}.get(key, key)
            if native == 'stop':
                assert sent[native][:len(value)] == value
            else:
                assert sent[native] == value


@pytest.mark.parametrize('options,flag,value', [
    ({'num_ctx': 1024}, '--ctx-size', '1024'),
    ({'num_batch': 128}, '--batch-size', '128'),
    ({'num_gpu': 3}, '--n-gpu-layers', '3'),
    ({'main_gpu': 1}, '--main-gpu', '1'),
    ({'use_mmap': False}, '--load-mode', 'none'),
    ({'num_thread': 2}, '--threads', '2'),
    ({'draft_num_predict': 4}, '--spec-draft-n-max', '4'),
])
async def test_runner_option_change_reloads_and_repeated_effective_value_reuses(public, options, flag, value):
    manager, client, name, toml = public
    manager.config = replace(manager.config, runtime_overrides={'device': 'Vulkan0'})
    original = toml.read_bytes()
    first = await manager.preload_native_model(name)
    body = request_body(name, '/api/chat')
    body['options'] = options
    for _ in range(2):
        response = await client.post('/api/chat', json=body)
        assert response.status_code == 200, response.text
    assert len(manager.test.ledger) == 2
    assert manager.test.ledger[0].process.returncode is not None
    args = manager.test.ledger[-1].command
    assert args[args.index(flag) + 1] == value
    await manager.wait_native_closed(first)
    del body['options']
    assert (await client.post('/api/chat', json=body)).status_code == 200
    assert len(manager.test.ledger) == 3 and toml.read_bytes() == original


async def test_sampling_options_do_not_reload_and_zero_is_not_unlimited(public):
    manager, client, name, _ = public
    first = await manager.preload_native_model(name)
    for options in [{'repeat_last_n': -1, 'num_keep': 0}, {'repeat_last_n': 0, 'num_keep': 8}]:
        # The first explicit num_keep enables shifting; changing its value is per request.
        response = await client.post('/api/chat', json={**request_body(name, '/api/chat'), 'options': options})
        assert response.status_code == 200
    assert len(manager.test.ledger) == 2
    for options in [{'temperature': .1}, {'seed': 0}, {'stop': []}, {'num_predict': 0}]:
        options['num_keep'] = 8
        assert (await client.post('/api/chat', json={**request_body(name, '/api/chat'), 'options': options})).status_code == 200
    assert len(manager.test.ledger) == 2
    sent = records(manager.test.roots[-1], 'requests')[-1]['body']
    assert sent['n_predict'] == 0
    await manager.wait_native_closed(first)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_empty_preload_uses_runner_options(public, path):
    manager, client, name, _ = public
    body = {'model': name, 'messages': [], 'stream': False, 'options': {'num_thread': 3}}
    assert (await client.post(path, json=body)).status_code == 200
    assert '--threads' in manager.test.ledger[0].command
    assert not any(r['path'] in {'/completion', '/v1/chat/completions'} for r in records(manager.test.roots[0], 'requests'))


@pytest.mark.parametrize('name,value', [
    ('num_ctx', -1), ('num_ctx', True), ('num_batch', 0), ('num_batch', '32'),
    ('num_thread', 2.5), ('num_gpu', -2), ('main_gpu', -1), ('use_mmap', 'false'),
    ('draft_num_predict', -1), ('draft_num_predict', True), ('draft_num_predict', 3.5),
    ('draft_num_predict', 2**31), ('repeat_last_n', -2), ('repeat_last_n', False),
    ('num_keep', -2), ('num_keep', 1.5), ('num_keep', True), ('num_keep', 2**31),
    ('stop', ['ok', 42]), ('stop', 42), ('num_predict', True), ('num_predict', 3.5),
])
@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
async def test_invalid_options_fail_before_preload_or_unload(public, path, name, value):
    manager, client, model, _ = public
    first = await manager.preload_native_model(model)
    response = await client.post(path, json={'model': model, 'options': {name: value}, 'keep_alive': 0})
    assert response.status_code == 400, response.text
    assert len(manager.test.ledger) == 1 and manager.test.ledger[0].process.returncode is None
    assert manager.native_snapshot()[0]['generation'] == first


def test_request_beats_toml_and_globals_without_mutation_and_same_value_does_not_reload():
    config = BackendConfig(runtime_overrides={'device': 'Vulkan0', 'n_threads': '1'})
    spec = parse_model_toml('[llm]\nmodel="Local/a.gguf"\n[runtime]\nn_threads=2\nspec_type="draft-mtp"\ndraft_n_max=5')
    plain = resolve_worker_config(config, Path('/models/a.gguf'), spec)
    same = resolve_worker_config(config, Path('/models/a.gguf'), spec, runtime_overrides={'n_threads': 2})
    assert plain == same
    requested = resolve_worker_config(config, Path('/models/a.gguf'), spec,
                                      runtime_overrides=ollama_runtime_overrides({'num_thread': 3, 'draft_num_predict': 0}))
    args = requested.launch.command(Path('/private/key'))
    assert args[args.index('--spec-type') + 1] == 'none' and '--spec-draft-n-max' not in args
    assert requested.runtime['n_threads'] == 3 and requested.sources['n_threads'] == 'request'
    assert spec.runtime['n_threads'] == 2 and config.runtime_overrides['n_threads'] == '1'


@pytest.mark.parametrize('value', [0, 1, -1])
def test_repeat_window_is_a_request_setting(value):
    assert build_native_sampling({'repeat_last_n': value})['repeat_last_n'] == value


def test_explicit_empty_stop_clears_model_defaults():
    assert build_native_sampling({'stop': []}, {'stop_defaults': ['END']})['stop'] == []


def test_fill_context_disables_shift_without_sending_native_minus_two():
    options = {'num_keep': 8, 'num_predict': -2}
    assert ollama_runtime_overrides(options)['context_shift'] is False
    assert build_native_sampling(options)['n_predict'] == -1


async def test_concurrent_preloads_share_request_runner_configuration(native_manager):
    manager = native_manager()
    name, _ = manager.test.define()
    overrides = ollama_runtime_overrides({'num_batch': 96, 'draft_num_predict': 2})
    generations = await asyncio.gather(*(manager.preload_native_model(name, runtime_overrides=overrides) for _ in range(5)))
    assert len(set(generations)) == len(manager.test.ledger) == 1


def test_ollama_num_gpu_minus_one_means_auto_fit_even_with_an_explicit_device():
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'Vulkan0'}), Path('/models/a.gguf'),
                                     runtime_overrides=ollama_runtime_overrides({'num_gpu': -1}))
    args = resolved.launch.command(Path('/private/key'))
    assert args[args.index('--fit') + 1] == 'on'
    assert args[args.index('--n-gpu-layers') + 1] == 'auto'


async def test_request_conflicting_with_cpu_does_not_unload_a_working_worker(public):
    manager, client, name, _ = public
    first = await manager.preload_native_model(name)
    response = await client.post('/api/chat', json={**request_body(name, '/api/chat'), 'options': {'num_gpu': 3}})
    assert response.status_code == 400 and 'conflicts' in response.text
    assert manager.native_snapshot()[0]['generation'] == first
    assert manager.test.ledger[0].process.returncode is None


@pytest.mark.parametrize('keep,expected', [(0, list(range(14, 20))), (2, [0, 1, *range(15, 20)]),
                                           (-1, [*range(8), 18, 19])])
def test_long_completion_keeps_prefix_and_recent_tail(keep, expected):
    from tllama.ollama_context import truncate_completion
    assert truncate_completion(list(range(20)), 12, keep) == expected


async def test_long_chat_discards_whole_old_tool_exchange(public):
    manager, client, name, _ = public
    messages = [{'role': 'system', 'content': 'SYSTEM'},
                {'role': 'user', 'content': 'x' * 2500},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'q': 'old'}}}]},
                {'role': 'tool', 'content': 'old tool result', 'tool_name': 'lookup'},
                {'role': 'assistant', 'content': 'old answer'},
                {'role': 'user', 'content': 'current question'}]
    response = await client.post('/api/chat', json={'model': name, 'messages': messages,
                                                   'options': {'num_keep': 4}, 'stream': False})
    assert response.status_code == 200, response.text
    sent = records(manager.test.roots[0], 'requests')[-1]['body']
    assert sent['messages'] == [messages[0], messages[-1]] and sent['n_keep'] == 4


async def test_oversized_latest_turn_is_a_request_error_and_releases_lease(public):
    manager, client, name, _ = public
    response = await client.post('/api/chat', json={'model': name,
        'messages': [{'role': 'user', 'content': 'x' * 2500}], 'options': {'num_keep': -1}, 'stream': False})
    assert response.status_code == 400 and 'latest chat turn' in response.text
    assert manager.native_snapshot()[0]['leases'] == 0
    assert not any(r['path'] == '/v1/chat/completions' for r in records(manager.test.roots[0], 'requests'))


async def test_long_generate_preserves_requested_prefix_and_tail(public):
    manager, client, name, _ = public
    prompt = 'PREFIX' + 'x' * 2500 + 'SUFFIX'
    response = await client.post('/api/generate', json={'model': name, 'prompt': prompt, 'raw': True,
        'options': {'num_keep': 6}, 'stream': False})
    assert response.status_code == 200, response.text
    sent = records(manager.test.roots[0], 'requests')[-1]['body']
    assert len(sent['prompt']) < 2048
    text = ''.join(chr(token - 10) for token in sent['prompt'][1:])
    assert text.startswith('PREFIX') and text.endswith('SUFFIX')


async def test_request_reload_keeps_existing_lease_immutable(native_manager):
    manager = native_manager(limit=2)
    name, _ = manager.test.define()
    async with manager.acquire_native_model(name, runtime_overrides={'n_threads': 2}) as first:
        async with manager.acquire_native_model(name, runtime_overrides={'n_threads': 3}) as second:
            assert first.generation != second.generation
            assert first.resolved.runtime['n_threads'] == 2
            assert second.resolved.runtime['n_threads'] == 3
            assert first.worker.process.returncode is None
        assert first.worker.process.returncode is None
    await manager.wait_native_closed(first.generation)


def test_num_keep_rejects_native_projector_limitation_instead_of_ignoring():
    from tllama.config import ConfigError
    with pytest.raises(ConfigError, match='multimodal'):
        resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}), Path('/models/m.gguf'),
                              mmproj=Path('/models/mmproj.gguf'),
                              runtime_overrides=ollama_runtime_overrides({'num_keep': 1}))


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
def test_explicit_empty_stop_clears_configured_defaults_on_both_routes(client, native_reply, path):
    state = native_reply(metadata={'stop_defaults': ['MODEL_STOP']})
    body = request_body('m', path)
    body['options'] = {'stop': []}
    assert client.post(path, json=body).status_code == 200
    assert 'MODEL_STOP' not in state.worker.calls[-1][1].get('stop', [])


@pytest.mark.parametrize('threads', [0, -1])
def test_nonpositive_ollama_threads_restore_native_automatic_policy(threads):
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none', 'n_threads': '2', 'n_threads_batch': '4'}),
        Path('/models/a.gguf'), runtime_overrides=ollama_runtime_overrides({'num_thread': threads}))
    args = resolved.launch.command(Path('/private/key'))
    assert '--threads' not in args and '--threads-batch' not in args


async def test_fill_context_reuses_worker_when_shifting_is_already_disabled(public):
    manager, client, name, _ = public
    first = await manager.preload_native_model(name)
    response = await client.post('/api/chat', json={**request_body(name, '/api/chat'), 'options': {'num_predict': -2}})
    assert response.status_code == 200
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['generation'] == first
