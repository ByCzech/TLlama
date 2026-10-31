from copy import deepcopy
import json
import base64
from types import SimpleNamespace

import pytest

from tllama.cli_repl import command
from tllama.cli_run import RunSession
from tllama.cli_sessions import SessionStore
from tllama.client import ClientError


def conversation():
    return RunSession('model', 'completion', {'options': {'num_ctx': 4096}, 'think': True},
                      system='system', messages=[{'role': 'user', 'content': 'hello'},
                      {'role': 'assistant', 'content': 'answer', 'thinking': 'reasoning'}], verbose=True)


async def test_save_load_and_forget_are_local_scoped_and_restore_controls(capsys):
    client = SimpleNamespace(url='http://localhost:54800/')
    session = conversation()
    expected = deepcopy(session)
    assert await command(client, session, '/save demo') is True
    store = SessionStore(client.url, session.model, session.mode)
    assert store.path('demo').stat().st_mode & 0o777 == 0o600
    assert store.directory.stat().st_mode & 0o777 == 0o700
    session.messages.clear()
    session.controls.clear()
    session.system = None
    await command(client, session, '/load demo')
    assert session == expected
    assert SessionStore('other:54800', 'model', 'completion').names() == []
    assert SessionStore(client.url, 'other', 'completion').names() == []
    await command(client, session, '/sessions')
    await command(client, session, '/forget demo')
    assert store.names() == [] and session == expected
    output = capsys.readouterr()
    assert not output.out and 'demo' in output.err


@pytest.mark.parametrize('change', [
    lambda d: d.update(server='http://other:54800/'),
    lambda d: d.update(model='other'),
    lambda d: d.update(version=True),
    lambda d: d.update(mode='embedding'),
    lambda d: d.update(messages=[{'role': 'user', 'content': 1}]),
    lambda d: d.update(display={'verbose': 1, 'hide_thinking': False}),
    lambda d: d.update(controls={'unsupported': True}),
])
def test_bad_saves_do_not_partially_mutate_live_session(change):
    session = conversation()
    before = deepcopy(session)
    store = SessionStore(None, 'model', 'completion')
    store.save('demo', session)
    data = json.loads(store.path('demo').read_text())
    change(data)
    store.path('demo').write_text(json.dumps(data))
    with pytest.raises(ClientError):
        store.load('demo', session)
    assert session == before


@pytest.mark.parametrize('name', ['', '../secret', '/absolute', 'a/b', '.', 'a b', 'x' * 65])
def test_session_names_cannot_escape_the_scope(name):
    with pytest.raises(ClientError):
        SessionStore(None, 'model', 'completion').save(name, conversation())


def test_duplicate_json_and_symlinks_are_rejected(tmp_path):
    session = conversation()
    store = SessionStore(None, 'model', 'completion')
    store.save('demo', session)
    store.path('demo').write_text('{"version":1,"version":1}')
    with pytest.raises(ClientError, match='Duplicate'):
        store.load('demo', session)
    target = tmp_path / 'outside'
    target.write_text('{}')
    store.path('demo').unlink()
    store.path('demo').symlink_to(target)
    with pytest.raises(ClientError):
        store.load('demo', session)


def test_default_limit_saves_full_size_image_and_additional_image_without_source_paths():
    from test_client_images import PNG, GIF
    from tllama.cli_images import MAX_IMAGE_BYTES
    # Include one image at the client limit: base64 alone exceeds the old
    # 16 MiB session limit. Trailing bytes do not affect header sniffing.
    image = base64.b64encode(PNG + bytes(MAX_IMAGE_BYTES - len(PNG))).decode('ascii')
    session = RunSession('model', 'completion', vision=True, messages=[
        {'role': 'user', 'content': 'two pictures', 'images': [image, base64.b64encode(GIF).decode('ascii')]},
        {'role': 'assistant', 'content': 'answer'}])
    store = SessionStore(None, session.model, session.mode)
    store.save('images', session)
    assert 16 * 1024 * 1024 < store.path('images').stat().st_size < 256 * 1024 * 1024
    restored = RunSession('model', 'completion', vision=True, session_max_file_mib=128)
    store.load('images', restored)
    assert restored.messages == session.messages
    # This client preference is not part of the saved conversation.
    assert restored.session_max_file_mib == 128


def test_smaller_load_limit_keeps_live_messages_controls_and_pending_images():
    from tllama.cli_images import PendingImage
    writer = SessionStore(None, 'model', 'completion', max_file_mib=2)
    saved = RunSession('model', 'completion', messages=[{'role': 'user', 'content': 'x' * (1024 * 1024)}])
    writer.save('large', saved)
    live = conversation()
    live.pending_images = [PendingImage('local', 'pending')]
    before = deepcopy(live)
    with pytest.raises(ClientError, match='1 MiB'):
        SessionStore(None, 'model', 'completion', max_file_mib=1).load('large', live)
    assert live == before


async def test_repl_uses_configured_limit_for_save_and_load():
    live = RunSession('model', 'completion', session_max_file_mib=1)
    await command(None, live, '/save demo')
    store = SessionStore(None, 'model', 'completion')
    original = store.path('demo').read_bytes()
    live.messages = [{'role': 'user', 'content': 'x' * (1024 * 1024)}]
    with pytest.raises(ClientError, match='1 MiB'):
        await command(None, live, '/save demo')
    assert store.path('demo').read_bytes() == original
    store.save('large', live)
    before = deepcopy(live)
    with pytest.raises(ClientError, match='1 MiB'):
        await command(None, live, '/load large')
    assert live == before
