import asyncio
import io
from types import SimpleNamespace

import pytest

from tllama.client import ClientError
from tllama.client_config import load_client_config
from tllama.client_notifications import BellNotifier
from tllama.cli_run import RunSession
from test_client_cli import cli_server, isolated_proxy_environment


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_bell_is_opt_in_tty_only_and_never_modifies_stdout(monkeypatch):
    err, out = Terminal(), io.StringIO()
    monkeypatch.setattr('sys.stderr', err)
    monkeypatch.setattr('sys.stdout', out)
    BellNotifier().notify('response_finished')
    assert err.getvalue() == ''
    notifier = BellNotifier(enabled=True)
    notifier.notify('pull_finished')
    notifier.notify('response_finished')
    assert err.getvalue() == '\a' and out.getvalue() == ''
    pipe = io.StringIO()
    monkeypatch.setattr('sys.stderr', pipe)
    notifier.notify('response_finished')
    assert pipe.getvalue() == ''


def test_failed_bell_does_not_fail_the_operation(monkeypatch):
    class Broken(Terminal):
        def write(self, text):
            raise OSError('closed terminal')
    monkeypatch.setattr('sys.stderr', Broken())
    BellNotifier(enabled=True).notify('response_finished')


async def test_preload_and_completed_response_notify_once_and_after_output(capsys):
    events = []
    class Notifier:
        def notify(self, event):
            events.append((event, capsys.readouterr().out))
    async def preload(*args, **kw):
        return True
    async def chat(*args, **kw):
        yield {'message': {'role': 'assistant', 'content': 'answer'}, 'done': True}
    session = RunSession('m', 'completion', notifier=Notifier())
    client = SimpleNamespace(preload=preload, chat=chat)
    await session.preload(client)
    await session.submit(client, 'question')
    assert events == [('model_ready', ''), ('response_finished', 'answer\n')]


@pytest.mark.parametrize('failure', [ClientError('failed'), asyncio.CancelledError()])
async def test_failed_or_cancelled_response_does_not_notify_completion(failure):
    events = []
    async def chat(*args, **kw):
        raise failure
        yield
    session = RunSession('m', 'completion', notifier=SimpleNamespace(notify=events.append))
    with pytest.raises(type(failure)):
        await session.submit(SimpleNamespace(chat=chat), 'question')
    assert events == [] and session.messages == []


async def test_unretained_preload_does_not_notify():
    events = []
    async def preload(*args, **kw):
        return False
    session = RunSession('m', 'completion', notifier=SimpleNamespace(notify=events.append))
    await session.preload(SimpleNamespace(preload=preload))
    assert events == []


@pytest.mark.parametrize('events', ['["unknown"]', '["error", "error"]', '[1]', 'true'])
def test_event_lists_are_validated(tmp_path, events):
    file = tmp_path / 'client.toml'
    file.write_text(f'[notifications]\nbell=true\nbell_events={events}\n')
    with pytest.raises(ValueError, match='supported events'):
        load_client_config(file)


def test_bell_config_and_cli_precedence(tmp_path, monkeypatch):
    from tllama import cli
    file = tmp_path / 'client.toml'
    file.write_text('[notifications]\nbell=true\nbell_events=["error"]\n')
    seen = []
    async def execute(args):
        seen.append((args.bell, args.bell_events))
        return 0
    monkeypatch.setattr(cli, '_execute', execute)
    assert cli.main(['--config', str(file), 'ps']) == 0
    assert cli.main(['--config', str(file), '--no-bell', 'ps']) == 0
    assert cli.main(['--config', str(file), 'ps', '--no-bell']) == 0
    assert seen == [(True, ('error',)), (False, ('error',)), (False, ('error',))]


def test_pull_success_and_error_are_distinct_events(cli_server, tmp_path, monkeypatch, capsys):
    from tllama.cli import main
    events = []
    monkeypatch.setattr(BellNotifier, 'notify', lambda self, event: events.append(event))
    file = tmp_path / 'client.toml'
    file.write_text('[notifications]\nbell=true\nbell_events=["pull_finished","error"]\n')
    args = ['--config', str(file), '--host', cli_server['url'], 'pull', 'model', '--json']
    cli_server['responses']['/api/pull'] = (200, b'{"status":"success"}\n')
    assert main(args) == 0 and events == ['pull_finished']
    assert capsys.readouterr().out == '{"status": "success"}\n'
    events.clear()
    cli_server['responses']['/api/pull'] = (500, {'error': 'failed'})
    assert main(args) == 1 and events == ['error']


def test_create_success_notifies_after_import(tmp_path, monkeypatch):
    from tllama.cli import main
    events = []
    monkeypatch.setattr(BellNotifier, 'notify', lambda self, event: events.append(event))
    async def create(*args, **kw):
        assert events == []
        return {'status': 'success'}
    monkeypatch.setattr('tllama.client_import.create_from_file', create)
    assert main(['create', 'model', '-f', str(tmp_path / 'unused'), '--bell']) == 0
    assert events == ['create_finished']
