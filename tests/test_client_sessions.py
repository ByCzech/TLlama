from copy import deepcopy
import json
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
