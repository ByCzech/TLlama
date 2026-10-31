"""Image bytes over the real client transport and transactional REPL/session state."""

import asyncio
import base64
from copy import deepcopy
import json
import os
import shlex
from types import SimpleNamespace

import pytest
import httpx

from tllama import cli
from tllama.cli_images import read_image
from tllama.cli_repl import command, repl
from tllama.cli_run import RunSession
from tllama.cli_sessions import SessionStore
from tllama.client import ClientError, TLlamaClient
from test_client_cli import cli_server, isolated_proxy_environment
from test_client_repl import Reader
from test_client_run import Input
from test_native_manager import native_manager, records
from test_native_public_api import public
from test_native_vision import vision


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7l8AAAAASUVORK5CYII=')
GIF = base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')


@pytest.fixture
def pictures(tmp_path):
    paths = [tmp_path / 'first image.not-a-png-extension', tmp_path / 'second.gif']
    for path, data in zip(paths, [PNG, GIF]):
        path.write_bytes(data)
    return paths


def finished(text='answer'):
    return {'message': {'role': 'assistant', 'content': text}, 'done': True}


def test_cli_sends_ordered_local_bytes_with_piped_text_and_clean_json(cli_server, monkeypatch, pictures, capsys):
    monkeypatch.setattr('sys.stdin', Input('piped'))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion', 'vision']})
    cli_server['responses']['/api/chat'] = (200, (json.dumps(finished()) + '\n').encode())
    assert cli.main(['run', 'vision', 'compare', '--image', str(pictures[0]), '--image', str(pictures[1]),
                     '--json', '--host', cli_server['url']]) == 0
    assert [call[1] for call in cli_server['calls']] == ['/api/show', '/api/chat']
    user = cli_server['calls'][-1][2]['messages'][0]
    assert user['content'] == 'piped\n\ncompare'
    assert [base64.b64decode(image) for image in user['images']] == [PNG, GIF]
    output = capsys.readouterr()
    assert json.loads(output.out) == finished() and output.err == ''
    assert all(str(path) not in json.dumps(user) for path in pictures)


async def test_image_client_reaches_the_public_api_and_owned_native_worker(vision, tmp_path, capsys):
    from tllama.main import app
    from vision_fixtures import RED_BLUE, RED_BLUE_URL
    manager, _, model, _ = vision
    image = tmp_path / 'panels.png'
    image.write_bytes(base64.b64decode(RED_BLUE))
    args = cli.build_parser().parse_args(['run', model, '--image', str(image), '--think', 'false'])
    async with TLlamaClient('http://public', transport=httpx.ASGITransport(app=app)) as client:
        session = await RunSession.create(client, args)
        await session.submit(client, 'Describe the two colored panels.')
        image.unlink()
        await session.submit(client, 'What was on the left?')
    assert len(manager.test.ledger) == 1 and manager.native_snapshot()[0]['leases'] == 0
    calls = [row['body'] for row in records(manager.test.roots[0], 'requests')
             if row['path'] == '/v1/chat/completions']
    assert len(calls) == 2
    assert calls[0]['messages'][-1]['content'][1]['image_url']['url'] == RED_BLUE_URL
    assert calls[1]['messages'][1]['content'][1]['image_url']['url'] == RED_BLUE_URL
    assert calls[1]['messages'][-1]['content'] == 'What was on the left?'
    assert len(session.messages) == 4 and not session.pending_images


@pytest.mark.parametrize('arguments', [
    ['run', 'm', '--image', 'first', 'prompt', '--image', 'second'],
    ['run', '--image', 'first', 'm', 'prompt', '--image', 'second'],
    ['run', 'm', 'prompt', '--image', 'first', '--image', 'second'],
])
def test_image_options_preserve_order_around_model_and_prompt(arguments):
    parsed = cli.build_parser().parse_args(arguments)
    assert parsed.model == 'm' and parsed.prompt == ['prompt']
    assert parsed.image == ['first', 'second']


def test_bad_file_is_reported_before_preload_and_without_stdout(cli_server, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion', 'vision']})
    assert cli.main(['run', 'm', '--image', str(tmp_path / 'missing'), '--host', cli_server['url']]) == 1
    assert [call[1] for call in cli_server['calls']] == ['/api/show']
    output = capsys.readouterr()
    assert not output.out and 'Cannot attach image' in output.err


@pytest.mark.parametrize('limit,success', [(1, False), (2, True)])
def test_configured_image_limit_reaches_cli_file_reader(cli_server, monkeypatch, tmp_path, capsys, limit, success):
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    image = tmp_path / 'picture.png'
    image.write_bytes(PNG + bytes(1024 * 1024 + 1 - len(PNG)))
    config = tmp_path / 'client.toml'
    config.write_text(f'[images]\nmax_file_mib={limit}\n')
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion', 'vision']})
    cli_server['responses']['/api/chat'] = (200, (json.dumps(finished()) + '\n').encode())
    code = cli.main(['--config', str(config), 'run', 'm', 'look', '--image', str(image), '--host', cli_server['url']])
    assert code == (0 if success else 1)
    assert [call[1] for call in cli_server['calls']] == (['/api/show', '/api/chat'] if success else ['/api/show'])
    output = capsys.readouterr()
    if success:
        assert base64.b64decode(cli_server['calls'][-1][2]['messages'][0]['images'][0]) == image.read_bytes()
        assert output.out == 'answer\n'
    else:
        assert not output.out and '1 MiB' in output.err


@pytest.mark.parametrize('capabilities', [['completion'], ['embedding'], ['decision']])
def test_nonvision_modes_refuse_images_before_preload_or_inference(cli_server, monkeypatch, pictures, capabilities, capsys):
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': capabilities})
    assert cli.main(['run', 'm', '--image', str(pictures[0]), '--host', cli_server['url']]) == 1
    assert [call[1] for call in cli_server['calls']] == ['/api/show']
    assert 'advertising vision' in capsys.readouterr().err


def test_cli_images_without_prompt_queue_for_first_repl_input(cli_server, monkeypatch, pictures):
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion', 'vision']})
    cli_server['responses']['/api/generate'] = (200, {'done': True, 'done_reason': 'load'})
    seen = []
    async def shell(client, session):
        seen.extend(session.pending_images)
        return 0
    monkeypatch.setattr('tllama.cli_repl.repl', shell)
    assert cli.main(['run', 'm', '--image', str(pictures[0]), '--host', cli_server['url']]) == 0
    assert base64.b64decode(seen[0].data) == PNG


@pytest.mark.parametrize('kind', ['missing', 'directory', 'empty', 'text', 'oversize', 'denied'])
def test_bad_local_files_leave_existing_queue_unchanged(kind, pictures, tmp_path, monkeypatch):
    session = RunSession('m', 'completion', vision=True)
    session.attach_image(pictures[0])
    before = deepcopy(session.pending_images)
    path = tmp_path / kind
    if kind == 'directory':
        path.mkdir()
    elif kind in {'empty', 'text'}:
        path.write_bytes(b'' if kind == 'empty' else b'not an image')
    elif kind == 'oversize':
        path.write_bytes(PNG)
        with path.open('ab') as file:
            file.truncate(1024 * 1024 + 1)
        session.image_max_file_mib = 1
    elif kind == 'denied':
        def denied(*args, **kwargs):
            raise PermissionError('denied')
        monkeypatch.setattr('tllama.cli_images.os.open', denied)
    with pytest.raises(ClientError, match='Cannot attach image'):
        session.attach_image(path)
    assert session.pending_images == before


@pytest.mark.skipif(os.name != 'posix', reason='POSIX FIFO input')
def test_fifo_is_refused_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / 'pipe'
    os.mkfifo(path)
    with pytest.raises(ClientError, match='regular file'):
        read_image(path)


async def test_repl_queue_remove_image_only_followup_and_clear(pictures, capsys):
    calls = []
    async def chat(model, messages, **controls):
        calls.append(deepcopy(messages))
        yield finished()
    session = RunSession('m', 'completion', vision=True)
    assert await repl(SimpleNamespace(chat=chat), session, reader=Reader(
        '/image ' + shlex.quote(str(pictures[0])), '/image ' + shlex.quote(str(pictures[1])),
        '/images', '/unimage 1', '', 'follow up',
        '/image ' + shlex.quote(str(pictures[0])), '/unimage all',
        '/image ' + shlex.quote(str(pictures[0])), '/clear', '/images', '/bye')) == 0
    image = base64.b64encode(GIF).decode()
    assert calls[0] == [{'role': 'user', 'content': '', 'images': [image]}]
    assert calls[1][0] == calls[0][0] and calls[1][-1] == {'role': 'user', 'content': 'follow up'}
    assert session.messages == [] and session.pending_images == []
    output = capsys.readouterr()
    assert output.out == 'answer\nanswer\n' and 'No queued images.' in output.err
    assert image not in output.out + output.err


@pytest.mark.parametrize('failure', ['error', 'cancel'])
async def test_failed_image_turn_preserves_queue_for_retry(failure, pictures, capsys):
    session = RunSession('m', 'completion', vision=True)
    session.attach_image(pictures[0])
    queued = deepcopy(session.pending_images)
    async def broken(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'content': 'partial'}, 'done': False}
        raise ClientError('failed') if failure == 'error' else asyncio.CancelledError()
    with pytest.raises(ClientError if failure == 'error' else asyncio.CancelledError):
        await session.submit(SimpleNamespace(chat=broken), 'prompt')
    assert session.messages == [] and session.pending_images == queued
    calls = []
    async def ok(model, messages, **kwargs):
        calls.extend(deepcopy(messages))
        yield finished()
    pictures[0].unlink()
    await session.submit(SimpleNamespace(chat=ok), 'retry')
    assert base64.b64decode(calls[0]['images'][0]) == PNG
    assert session.pending_images == [] and session.messages == calls + [finished()['message']]


async def test_save_restore_contains_image_bytes_and_replays_no_request(pictures, capsys):
    session = RunSession('m', 'completion', vision=True)
    session.attach_image(pictures[0])
    async def chat(*args, **kwargs):
        yield finished()
    await session.submit(SimpleNamespace(chat=chat), 'picture')
    store = SessionStore(None, session.model, session.mode)
    session.attach_image(pictures[1])
    store.save('vision', session)
    saved = json.loads(store.path('vision').read_text())
    assert saved['version'] == 1 and 'pending_images' not in saved
    assert base64.b64decode(saved['messages'][0]['images'][0]) == PNG
    for path in pictures:
        path.unlink()
    restored = RunSession('m', 'completion', vision=True)
    store.load('vision', restored)
    assert restored.messages == session.messages and restored.pending_images == []
    calls = []
    async def next_turn(model, messages, **kwargs):
        calls.extend(deepcopy(messages))
        yield finished()
    await restored.submit(SimpleNamespace(chat=next_turn), 'what colour?')
    assert calls[0]['images'] == saved['messages'][0]['images']


@pytest.mark.parametrize('change', [
    lambda data: data['messages'][0].update(images=['not base64']),
    lambda data: data['messages'][0].update(images=[]),
    lambda data: data['messages'][0].update(images='string'),
    lambda data: data['messages'][1].update(images=[base64.b64encode(PNG).decode()]),
])
def test_invalid_saved_images_leave_conversation_and_queue_unchanged(pictures, change):
    session = RunSession('m', 'completion', vision=True, messages=[
        {'role': 'user', 'content': '', 'images': [base64.b64encode(PNG).decode()]},
        {'role': 'assistant', 'content': 'answer'}])
    session.attach_image(pictures[0])
    before = deepcopy(session)
    store = SessionStore(None, session.model, session.mode)
    store.save('good', session)
    data = json.loads(store.path('good').read_text())
    change(data)
    store.path('good').write_text(json.dumps(data))
    with pytest.raises(ClientError, match='Invalid session'):
        store.load('good', session)
    assert session == before


def test_large_image_save_does_not_replace_existing_save(pictures):
    store = SessionStore(None, 'm', 'completion', max_file_mib=1)
    session = RunSession('m', 'completion', vision=True)
    store.save('demo', session)
    original = store.path('demo').read_bytes()
    session.messages = [{'role': 'user', 'content': 'x' * (1024 * 1024), 'images': [base64.b64encode(PNG).decode()]}]
    with pytest.raises(ClientError, match='1 MiB'):
        store.save('demo', session)
    assert store.path('demo').read_bytes() == original


async def test_load_and_system_reset_clear_only_pending_images_after_success(pictures):
    session = RunSession('m', 'completion', vision=True)
    store = SessionStore(None, 'm', 'completion')
    store.save('empty', session)
    session.attach_image(pictures[0])
    store.load('empty', session)
    assert session.pending_images == []
    session.attach_image(pictures[0])
    await command(None, session, '/set system reset')
    assert session.pending_images == []


async def test_history_contains_typed_paths_and_prompts_without_encoded_bytes(pictures, capsys):
    from tllama.cli_history import input_history
    history = input_history(None, 'm')
    image_command = '/image ' + shlex.quote(str(pictures[0]))
    class RecordingReader(Reader):
        async def prompt_async(self, prompt):
            text = await super().prompt_async(prompt)
            history.append_string(text)
            return text
    async def chat(*args, **kwargs):
        yield finished()
    session = RunSession('m', 'completion', vision=True)
    await repl(SimpleNamespace(chat=chat), session, reader=RecordingReader(image_command, 'look', '/bye'))
    recorded = list(input_history(None, 'm').load_history_strings())
    assert image_command in recorded and 'look' in recorded
    assert base64.b64encode(PNG).decode() not in '\n'.join(recorded)


def test_image_saves_refuse_a_model_without_vision(pictures):
    store = SessionStore(None, 'm', 'completion')
    source = RunSession('m', 'completion', vision=True, messages=[
        {'role': 'user', 'content': '', 'images': [base64.b64encode(PNG).decode()]}])
    store.save('demo', source)
    destination = RunSession('m', 'completion', system='keep')
    with pytest.raises(ClientError, match='advertising vision'):
        store.load('demo', destination)
    assert destination.system == 'keep' and destination.messages == []


@pytest.mark.parametrize('text', ['/image', '/image a b', '/image "unfinished',
                                 '/unimage', '/unimage 0', '/unimage 2', '/unimage -1', '/images extra'])
async def test_bad_image_commands_keep_the_queue(text, pictures):
    session = RunSession('m', 'completion', vision=True)
    session.attach_image(pictures[0])
    before = deepcopy(session.pending_images)
    with pytest.raises((ClientError, ValueError)):
        await command(None, session, text)
    assert session.pending_images == before
