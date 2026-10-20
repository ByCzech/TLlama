"""Embedding schemas and worker admission agree on supported load controls."""

import pytest

from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public
from test_embedding_api import embeddings, body, calls


@pytest.mark.parametrize('path', ['/api/embed', '/api/embeddings'])
async def test_embedding_runner_options_reach_launch_and_reload_without_changing_definition(embeddings, path):
    manager, client, name, toml, _ = embeddings
    original = toml.read_bytes()
    options = {'num_ctx': 16, 'num_batch': 32, 'num_gpu': 0, 'main_gpu': 0,
               'use_mmap': False, 'num_thread': 2}
    first_vector = None
    for threads in [2, 2, 3]:
        options['num_thread'] = threads
        response = await client.post(path, json=body(name, path, options=options))
        assert response.status_code == 200, response.text
        payload = response.json()
        vector = payload.get('embedding') or payload['embeddings'][0]
        if first_vector is None:
            first_vector = vector
        assert vector == first_vector
        assert len(manager.test.ledger) == (1 if threads == 2 else 2)
        command = manager.test.ledger[-1].command
        for flag, value in {'--ctx-size': '16', '--batch-size': '32', '--ubatch-size': '32',
                            '--n-gpu-layers': '0', '--device': 'none', '--main-gpu': '0',
                            '--load-mode': 'none', '--threads': str(threads),
                            '--threads-batch': str(threads)}.items():
            assert command[command.index(flag) + 1] == value
    assert toml.read_bytes() == original and len(calls(manager)) == 3
    assert manager.test.ledger[0].process.returncode is not None
    assert all('num_thread' not in c and 'num_gpu' not in c for c in calls(manager))


@pytest.mark.parametrize('path', ['/api/embed', '/api/embeddings'])
@pytest.mark.parametrize('options', [
    {'num_thread': True}, {'num_thread': '2'}, {'num_batch': 0}, {'num_gpu': -2},
    {'main_gpu': -1}, {'use_mmap': 1}, {'num_ctx': 2**31}, {'unknown': 1},
    {'temperature': 0.5}, {'draft_num_predict': 0}, {'num_keep': 1},
])
async def test_invalid_or_generation_only_options_never_load_even_for_empty_unload(embeddings, path, options):
    manager, client, name, _, _ = embeddings
    response = await client.post(path, json=body(name, path, '', options=options, keep_alive=0))
    assert response.status_code == 400, response.text
    assert not manager.test.ledger


@pytest.mark.parametrize('path', ['/api/embed', '/api/embeddings'])
async def test_empty_embedding_preload_applies_options_and_invalid_request_preserves_resident(embeddings, path):
    manager, client, name, _, _ = embeddings
    response = await client.post(path, json=body(name, path, '', options={'num_thread': 2}, keep_alive=-1))
    assert response.status_code == 200 and not calls(manager)
    generation = manager.native_snapshot()[0]['generation']
    bad = await client.post(path, json=body(name, path, '', options={'num_thread': True}, keep_alive=0))
    assert bad.status_code == 400 and manager.native_snapshot()[0]['generation'] == generation
    assert manager.test.ledger[0].process.returncode is None
    good = await client.post(path, json=body(name, path, keep_alive=0, options={'num_thread': 2}))
    assert good.status_code == 200
    await wait_until(lambda: not manager.native_snapshot())
    assert len(manager.test.ledger) == len(calls(manager)) == 1
