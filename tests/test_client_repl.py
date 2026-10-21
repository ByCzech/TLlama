"""Real terminal editing plus request interruption and session recovery."""

import asyncio
import io
import signal
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tllama import cli
from tllama.cli_repl import command, repl, submit, terminal
from tllama.cli_run import RunSession
from tllama.client import ClientError
from test_client_cli import cli_server, isolated_proxy_environment


class Reader:
    def __init__(self, *values):
        self.values = iter(values)

    async def prompt_async(self, prompt):
        value = next(self.values, EOFError())
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.parametrize('sequence,expected', [
    ('first\x1b\rsecond\r', 'first\nsecond'),
    ('\x1b[200~  one\n/two "three"\x1b[201~\r', '  one\n/two "three"'),
])
async def test_real_prompt_toolkit_preserves_multiline_input(monkeypatch, sequence, expected):
    monkeypatch.setattr('prompt_toolkit.output.defaults.create_output', lambda **kw: DummyOutput())
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        reader = terminal()
        pipe.send_text(sequence)
        assert await asyncio.wait_for(reader.prompt_async('>>> '), 2) == expected


async def test_repl_commands_history_and_literal_prompt_are_separate(capsys):
    calls = []
    async def chat(model, messages, **controls):
        calls.append((messages, {**controls}))
        yield {'message': {'role': 'assistant', 'content': 'ok'}, 'done': True}
    client = SimpleNamespace(chat=chat)
    session = RunSession('model', 'completion')
    reader = Reader(KeyboardInterrupt(), '/help', '/set parameter temperature 0',
                    '  literal "prompt"\n', '/clear', '//slash', '/bye')
    assert await repl(client, session, reader=reader) == 0
    assert calls[0][0] == [{'role': 'user', 'content': '  literal "prompt"\n'}]
    assert calls[1][0] == [{'role': 'user', 'content': '/slash'}]
    assert calls[1][1] == {'options': {'temperature': 0}}
    assert session.messages[0]['content'] == '/slash'
    output = capsys.readouterr()
    assert output.out == 'ok\nok\n' and '/help' in output.err


async def test_actual_sigint_closes_request_and_allows_next_turn(capsys):
    original = signal.getsignal(signal.SIGINT)
    closed = []
    async def chat(model, messages, **controls):
        try:
            if messages[-1]['content'] == 'cancel me':
                yield {'message': {'role': 'assistant', 'content': 'partial'}, 'done': False}
                signal.raise_signal(signal.SIGINT)
                await asyncio.Event().wait()
            yield {'message': {'role': 'assistant', 'content': 'complete'}, 'done': True}
        finally:
            closed.append(messages[-1]['content'])
    session = RunSession('model', 'completion')
    assert await repl(SimpleNamespace(chat=chat), session,
                      reader=Reader('cancel me', 'next', EOFError())) == 0
    assert signal.getsignal(signal.SIGINT) == original
    assert closed == ['cancel me', 'next']
    assert [m['content'] for m in session.messages] == ['next', 'complete']
    assert 'Request cancelled.' in capsys.readouterr().err


async def test_external_cancellation_is_not_swallowed():
    entered = asyncio.Event()
    closed = asyncio.Event()
    async def request(client, text):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()
    original = signal.getsignal(signal.SIGINT)
    task = asyncio.create_task(submit(None, SimpleNamespace(submit=request), 'prompt'))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and signal.getsignal(signal.SIGINT) == original


async def test_repeated_sigint_waits_for_cleanup(capsys):
    cleaned = asyncio.Event()
    async def request(client, text):
        try:
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        finally:
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0)
            cleaned.set()
    await submit(None, SimpleNamespace(submit=request), 'prompt')
    assert cleaned.is_set()


async def test_sigint_in_show_returns_to_prompt(capsys):
    closed = asyncio.Event()
    async def show(model):
        try:
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        finally:
            closed.set()
    assert await repl(SimpleNamespace(show=show), RunSession('model', 'completion'),
                      reader=Reader('/show model', '/bye')) == 0
    assert closed.is_set() and 'Request cancelled.' in capsys.readouterr().err


@pytest.mark.parametrize('text', ['/unknown', '/set', '/set parameter', '/set parameter x',
                                 '/set parameter x NaN', '/set format []', '/exit extra'])
async def test_bad_command_recovers_without_a_request(text, capsys):
    session = RunSession('model', 'completion')
    assert await repl(None, session, reader=Reader(text, '/bye')) == 0
    assert 'tllama:' in capsys.readouterr().err
    assert session.messages == []


async def test_failed_request_recovers_and_does_not_commit(capsys):
    calls = []
    async def chat(model, messages, **controls):
        calls.append(messages)
        if len(calls) == 1:
            raise ClientError('worker failed')
        yield {'message': {'role': 'assistant', 'content': 'answer'}, 'done': True}
    session = RunSession('model', 'completion')
    assert await repl(SimpleNamespace(chat=chat), session, reader=Reader('bad', 'good', '/exit')) == 0
    assert calls[1] == [{'role': 'user', 'content': 'good'}]
    assert 'worker failed' in capsys.readouterr().err


async def test_session_settings_are_local_and_system_change_clears_messages(capsys):
    session = RunSession('model', 'completion', messages=[{'role': 'user', 'content': 'old'}])
    for text in ('/set think false', '/set keep_alive 0', '/set format json',
                 '/set parameter stop ["END"]', '/set system New system'):
        assert await command(None, session, text)
    assert session.controls == {'think': False, 'keep_alive': '0', 'format': 'json',
                                'options': {'stop': ['END']}}
    assert session.system == 'New system' and session.messages == []
    await command(None, session, '/set format off')
    assert 'format' not in session.controls


@pytest.mark.parametrize('mode,text', [('embedding', '/set think false'),
                                      ('decision', '/set parameter temperature 0')])
async def test_repl_rejects_mode_inappropriate_settings(mode, text):
    with pytest.raises(ClientError):
        await command(None, RunSession('model', mode), text)


async def test_json_session_keeps_control_messages_off_stdout(capsys):
    async def chat(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'content': 'answer'}, 'done': True}
    session = RunSession('model', 'completion', as_json=True)
    await repl(SimpleNamespace(chat=chat), session, reader=Reader('/help', '/show parameters', 'prompt', '/exit'))
    output = capsys.readouterr()
    assert output.out.count('\n') == 1 and '"done": true' in output.out
    assert 'Session' not in output.out and '/help' in output.err


async def test_run_selects_repl_only_without_arguments_on_tty(monkeypatch):
    from tllama.cli_run import run
    class TTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr('sys.stdin', TTY())
    async def show(model):
        return {'capabilities': ['embedding']}
    async def preload(model, **kwargs):
        return True
    called = []
    async def shell(client, session):
        called.append(session.mode)
        return 0
    monkeypatch.setattr('tllama.cli_repl.repl', shell)
    args = cli.build_parser().parse_args(['run', 'model'])
    assert await run(SimpleNamespace(show=show, preload=preload), args) == 0
    assert called == ['embedding']


@pytest.mark.skipif(os.name != 'posix', reason='owned POSIX terminal smoke')
def test_real_cli_terminal_recovers_after_sigint(cli_server):
    import pty
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    cli_server['responses']['/api/generate'] = (200, {'done': True, 'done_reason': 'load'})
    def response(*args):
        # Keep the first request pending long enough for the parent to
        # observe admission and send a real SIGINT to the client process.
        if len([c for c in cli_server['calls'] if c[1] == '/api/chat']) == 1:
            time.sleep(0.5)
        return 200, b'{"message":{"role":"assistant","content":"FINAL_ANSWER"},"done":true}\n'
    cli_server['responses']['/api/chat'] = response
    master, slave = pty.openpty()
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
           'TERM': 'xterm', 'PROMPT_TOOLKIT_NO_CPR': '1'}
    process = None
    captured = bytearray()
    try:
        process = subprocess.Popen([sys.executable, '-m', 'tllama', 'run', 'model',
                                    '--host', cli_server['url']], stdin=slave, stderr=slave,
                                   stdout=subprocess.PIPE, env=env, start_new_session=True)
        os.close(slave)
        slave = None
        def until(phrase):
            deadline = time.monotonic() + 10
            while phrase not in captured:
                assert process.poll() is None, captured.decode(errors='replace')
                assert time.monotonic() < deadline, captured.decode(errors='replace')
                ready, _, _ = select.select([master, process.stdout.fileno()], [], [], 0.05)
                for descriptor in ready:
                    captured.extend(os.read(descriptor, 65536))
        until(b'>>>')
        os.write(master, b'cancel this\r')
        deadline = time.monotonic() + 5
        while not any(c[1] == '/api/chat' for c in cli_server['calls']):
            assert time.monotonic() < deadline
            # Poll through the terminal without blocking the server thread.
            select.select([], [], [], 0.01)
        os.kill(process.pid, signal.SIGINT)
        until(b'Request cancelled.')
        os.write(master, b'complete this\r')
        until(b'FINAL_ANSWER')
        os.write(master, b'/bye\r')
        assert process.wait(timeout=5) == 0
        requests = [c[2] for c in cli_server['calls'] if c[1] == '/api/chat']
        assert requests[-1]['messages'] == [{'role': 'user', 'content': 'complete this'}]
        assert b'Traceback' not in captured
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
        os.close(master)
        if slave is not None:
            os.close(slave)
