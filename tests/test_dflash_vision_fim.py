"""Composite DFlash adapters with synthetic vision/FIM producers, not GPU quality."""

import asyncio
import json
from pathlib import Path

import pytest

from tllama.helpers.model_toml import parse_model_toml
from test_helper_definitions import publish, reference
from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, stream_mode
from test_native_vision import vision, body as chat_image
from test_native_generate_vision import body as generate_image
from test_native_fim import fim, body as fim_body, write_vocab, FAMILIES, VOCAB, completions
from test_dflash_tools_structured import measured
from test_repository_worker_pins import protected_groups
from vision_fixtures import RED_BLUE, BLUE_RED, RED_BLUE_URL, BLUE_RED_URL


def attach(manager, toml, tmp_path, monkeypatch):
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    toml.write_text(toml.read_text() + '\n' + reference('draft', draft)
                    + '[runtime]\nspec_type="draft-dflash"\n')
    monkeypatch.setenv('STUB_PUBLIC_DFLASH', '1')
    return Path(draft).parts[1]


@pytest.fixture
async def dflash_vision(vision, tmp_path, monkeypatch):
    manager, client, name, toml = vision
    _, projector = publish(manager, tmp_path, 'clip', count=2)
    document = parse_model_toml(toml.read_text()).document
    document['mmproj']['model'] = projector
    toml.write_text(document.as_string())
    draft_group = attach(manager, toml, tmp_path, monkeypatch)
    original = toml.read_bytes()
    yield manager, client, name, toml, {Path(projector).parts[1], draft_group}
    assert toml.read_bytes() == original


@pytest.fixture
async def dflash_fim(fim, tmp_path, monkeypatch):
    manager, _, _, toml, _, _ = fim
    group = attach(manager, toml, tmp_path, monkeypatch)
    yield (*fim, group)


def image_body(name, path, stream=False, image=RED_BLUE):
    return (generate_image(name, stream=stream, images=[image]) if path == '/api/generate'
            else chat_image(name, path, stream, image))


def content(values, path, stream):
    if path == '/api/generate':
        return ''.join(v.get('response', '') for v in values)
    if path == '/api/chat':
        return ''.join(v['message'].get('content', '') for v in values)
    return ''.join(c.get('delta' if stream else 'message', {}).get('content', '')
                   for v in values for c in v['choices'])


def launch(manager, groups, active=True, projector=False):
    args = manager.test.ledger[-1].command
    assert args[args.index('--spec-type') + 1] == ('draft-dflash' if active else 'none')
    assert ('--spec-draft-model' in args) == active
    assert ('--mmproj' in args) == projector
    for flag in ('--mmproj', '--spec-draft-model'):
        if flag in args:
            assert len(list(Path(args[args.index(flag) + 1]).parent.glob('*.gguf'))) == 2
    assert protected_groups(manager) == groups
    assert manager.native_snapshot()[0]['leases'] == 0


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
async def test_vision_keeps_projector_draft_image_changes_and_schema(dflash_vision, path, stream):
    manager, client, name, _, groups = dflash_vision
    schema = {'type': 'object', 'properties': {'left': {'type': 'string'}, 'right': {'type': 'string'}}}
    for image, url, expected in [(RED_BLUE, RED_BLUE_URL, {'left': 'red', 'right': 'blue'}),
                                 (BLUE_RED, BLUE_RED_URL, {'left': 'blue', 'right': 'red'})]:
        body = image_body(name, path, stream, image)
        native_format = {'type': 'json_object', 'schema': schema}
        if path.startswith('/v1/'):
            native_format = {'type': 'json_schema', 'json_schema': {'name': 'panels', 'strict': True, 'schema': schema}}
            body['response_format'] = native_format
        else:
            body['format'] = schema
        response = await client.post(path, json=body)
        values = measured(response, path, stream)
        assert json.loads(content(values, path, stream)) == expected
        calls = records(manager.test.roots[0], 'requests')
        assert not any(c['path'] in {'/tokenize', '/completion', '/infill'} for c in calls)
        sent = [c['body'] for c in calls if c['path'] == '/v1/chat/completions'][-1]
        assert sent['messages'][-1]['content'][1]['image_url']['url'] == url
        assert sent['messages'][0] == {'role': 'system', 'content': 'SYSTEM'}
        assert sent['response_format'] == native_format and sent['temperature'] == 0.4
        assert sent['stop'] == ['TOML STOP'] and sent['reasoning_effort'] == 'none'
        assert len(manager.test.ledger) == 1
        launch(manager, groups, projector=True)


@pytest.mark.parametrize('stream', [False, True])
async def test_vision_off_unpins_only_draft_and_retains_both_stored_references(dflash_vision, stream):
    manager, client, name, toml, groups = dflash_vision
    original = toml.read_bytes()
    doc = parse_model_toml(toml.read_text())
    projector_group = Path(doc.mmproj_model).parts[1]
    body = image_body(name, '/api/chat', stream)
    for index, active in enumerate((True, False, True)):
        body['options'] = {'spec_type': 'draft-dflash' if active else 'none', 'draft_num_predict': 3}
        values = measured(await client.post('/api/chat', json=body), '/api/chat', stream, active)
        assert json.loads(content(values, '/api/chat', stream)) == {'left': 'red', 'right': 'blue'}
        launch(manager, groups if active else {projector_group}, active, projector=True)
        assert len(manager.test.ledger) == index + 1
        assert all(w.process.returncode is not None for w in manager.test.ledger[:-1])
    assert toml.read_bytes() == original
    generation = manager.native_snapshot()[0]['generation']
    await manager.unload_native_model(name)
    await manager.wait_native_closed(generation)
    assert not protected_groups(manager)
    assert all(len(list((manager.models_dir / 'data' / group).glob('*.gguf'))) == 2 for group in groups)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mode', ['no-vision', 'bad-image'])
async def test_vision_failure_does_not_publish_speculative_success(dflash_vision, path, stream, mode):
    manager, client, name, _, groups = dflash_vision
    manager.test.options['mode'] = mode
    response = await client.post(path, json=image_body(name, path, stream))
    assert response.status_code == (200 if mode == 'bad-image' and stream else 400)
    assert 'error' in response.text and '"speculative"' not in response.text
    assert '[DONE]' not in response.text and '"done": true' not in response.text
    if mode == 'no-vision':
        assert not any(c['path'] == '/v1/chat/completions' for c in records(manager.test.roots[0], 'requests'))
    launch(manager, groups, projector=True)


async def test_draining_vision_stream_keeps_both_helpers_until_child_reap(dflash_vision):
    manager, client, name, _, groups = dflash_vision
    stream_mode(manager, 'controlled')
    task = asyncio.create_task(client.post('/api/generate', json=image_body(name, '/api/generate', True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
        generation = manager.native_snapshot()[0]['generation']
        await manager.unload_native_model(name)
        assert manager.native_snapshot()[0]['state'] == 'draining'
        assert manager.native_snapshot()[0]['leases'] == 1 and protected_groups(manager) == groups
        assert manager.test.ledger[0].process.returncode is None
        (manager.test.roots[0] / 'release').touch()
        response = await task
    finally:
        if manager.test.roots:
            (manager.test.roots[0] / 'release').touch()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    measured(response, '/api/generate', True)
    await manager.wait_native_closed(generation)
    assert not protected_groups(manager) and not manager.test.ledger[0].runtime_directory.exists()


@pytest.mark.parametrize('family', ['codegemma', 'starcoder'])
@pytest.mark.parametrize('stream', [False, True])
async def test_fim_preserves_full_literal_tokens_stops_and_draft_counts(dflash_fim, family, stream):
    manager, client, name, toml, weights, settings, group = dflash_fim
    original = toml.read_bytes()
    write_vocab(weights, family)
    settings['mode'] = 'codegemma-split' if family == 'codegemma' else family
    prefix, suffix = '# literal code\n' + 'p' * 600 + '\nreturn ', '\n' + 's' * 300 + '\nassert combine(7, 2) == 14\n'
    options = {'num_ctx': 8192, 'num_predict': 32, 'temperature': 0, 'stop': ['CLIENT STOP']}
    values = measured(await client.post('/api/generate', json=fim_body(name, stream=stream,
        prompt=prefix, suffix=suffix, options=options)), '/api/generate', stream)
    assert content(values, '/api/generate', stream) == 'a * b'
    pre, suf, mid, file = FAMILIES[family]
    expected = [6, pre] + [ord(c) + 1000 for c in prefix] + [suf] + [ord(c) + 1000 for c in suffix] + [mid]
    [sent] = completions(manager)
    assert sent['prompt'] == expected and len(expected) > 512
    assert sent['stop'] == ['CLIENT STOP', *(VOCAB[i] for i in (pre, suf, mid, file))]
    assert sent['preserved_tokens'] == [VOCAB[i] for i in (pre, suf, mid, file)]
    assert sent['temperature'] == 0 and sent['n_predict'] == 32
    assert values[-1]['prompt_eval_count'] == len(expected) and values[-1]['done_reason'] == 'stop'
    assert not any(c['path'] in {'/infill', '/v1/chat/completions'} for c in records(manager.test.roots[0], 'requests'))
    assert toml.read_bytes() == original
    launch(manager, {group})


@pytest.mark.parametrize('order,expected', [('auto', [6,7,1120,8,1121,9]),
    ('training_spm', [6,7,8,1121,9,1120]), ('native_spm', [6,8,1121,7,1120,9])])
async def test_fim_order_is_independent_of_dflash_selection(dflash_fim, order, expected):
    manager, client, name, toml, _, _, group = dflash_fim
    toml.write_text(toml.read_text() + f'\n[fim]\norder="{order}"\n')
    measured(await client.post('/api/generate', json=fim_body(name, prompt='x', suffix='y')), '/api/generate', False)
    assert completions(manager)[0]['prompt'] == expected
    launch(manager, {group})


@pytest.mark.parametrize('stream', [False, True])
async def test_fim_off_reloads_without_changing_suffix_or_reusing_counters(dflash_fim, stream):
    manager, client, name, toml, _, _, group = dflash_fim
    original = toml.read_bytes()
    for index, active in enumerate((True, False)):
        response = await client.post('/api/generate', json=fim_body(name, stream=stream,
            options={'spec_type': 'draft-dflash' if active else 'none', 'draft_num_predict': 3}))
        values = measured(response, '/api/generate', stream, active)
        assert content(values, '/api/generate', stream) == 'a * b'
        launch(manager, {group} if active else set(), active)
        assert len(manager.test.ledger) == index + 1
    assert completions(manager)[0]['prompt'] == completions(manager)[1]['prompt']
    assert manager.test.ledger[0].process.returncode is not None and toml.read_bytes() == original


@pytest.mark.parametrize('family', ['negative', 'codestral'])
@pytest.mark.parametrize('stream', [False, True])
async def test_unknown_fim_recipe_is_not_enabled_by_draft_presence(dflash_fim, family, stream):
    manager, client, name, _, weights, settings, group = dflash_fim
    write_vocab(weights, family)
    settings['mode'] = family
    response = await client.post('/api/generate', json=fim_body(name, stream=stream))
    assert response.status_code == 501 and '"speculative"' not in response.text
    assert not completions(manager)
    launch(manager, {group})


@pytest.mark.parametrize('extra', [{'raw': True}, {'think': True}, {'images': [RED_BLUE]}, {'template': ''}])
async def test_invalid_fim_input_never_loads_target_or_draft(dflash_fim, extra):
    manager, client, name, _, _, _, _ = dflash_fim
    response = await client.post('/api/generate', json=fim_body(name, **extra))
    assert response.status_code == 400 and not manager.test.ledger
    assert not protected_groups(manager)


async def test_cancelled_draining_fim_releases_draft_only_after_worker_exit(dflash_fim):
    manager, client, name, _, _, _, group = dflash_fim
    stream_mode(manager, 'hold', 'completion_stream')
    task = asyncio.create_task(client.post('/api/generate', json=fim_body(name, stream=True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
        generation = manager.native_snapshot()[0]['generation']
        await manager.unload_native_model(name)
        assert manager.native_snapshot()[0]['state'] == 'draining' and protected_groups(manager) == {group}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await manager.wait_native_closed(generation)
        assert not protected_groups(manager) and not manager.test.ledger[0].runtime_directory.exists()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
