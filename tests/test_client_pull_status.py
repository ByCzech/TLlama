"""TTY pull rendering, wire compatibility and owned indicator/stream cleanup."""

import asyncio
import io
import json
import os

import httpx
import pytest

from tllama.cli import main
from tllama.cli_pull import pull_progress
from tllama.cli_status import FRAMES, WaitStatus
from tllama.client import ClientError, TLlamaClient
from test_client_cli import cli_server, isolated_proxy_environment


class Terminal(io.StringIO):
    def __init__(self, *, tty=True):
        super().__init__()
        self.tty = tty
        self.changed = asyncio.Event()

    def isatty(self):
        return self.tty

    def write(self, text):
        result = super().write(text)
        self.changed.set()
        return result

    async def until(self, text):
        async def wait():
            while text not in self.getvalue():
                self.changed.clear()
                await self.changed.wait()
        await asyncio.wait_for(wait(), 1)

    def history(self):
        # A carriage return moves the cursor; it does not add a history line.
        lines, line, cursor = [], [], 0
        for char in self.getvalue():
            if char == '\r':
                cursor = 0
            elif char == '\n':
                lines.append(''.join(line).rstrip())
                line, cursor = [], 0
            else:
                if cursor == len(line):
                    line.append(char)
                else:
                    line[cursor] = char
                cursor += 1
        return lines


@pytest.fixture
def terminal(monkeypatch):
    def make():
        output = Terminal()
        monkeypatch.setattr('sys.stderr', output)
        return output
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr('tllama.cli_status.INTERVAL', 0.001)
    monkeypatch.setattr('tllama.cli_pull.shutil.get_terminal_size', lambda: os.terminal_size((100, 24)))
    return make


def _events(cli_server, events):
    cli_server['responses']['/api/pull'] = (200, ('\n'.join(json.dumps(e) for e in events) + '\n').encode())


@pytest.mark.parametrize('ollama', [False, True])
def test_terminal_pull_keeps_only_phase_final_lines_and_same_request(cli_server, terminal, capsys, ollama):
    terminal = terminal()
    phase = 'pulling abcdef123456' if ollama else 'downloading model'
    extra = {'digest': 'sha256:abcdef123456'} if ollama else {}
    events = [{'status': 'resolving repository'}, *[
        {'status': phase, 'completed': n, 'total': 1000, **extra} for n in range(0, 1001, 10)],
        {'status': 'verifying sha256 digest'}, {'status': 'success'}]
    _events(cli_server, events)
    assert main(['pull', 'org/repo/file', '--host', cli_server['url']]) == 0
    assert capsys.readouterr().out == 'pulled: org/repo/file\n'
    history = terminal.history()
    assert history == ['resolving repository', f'{phase}: 1000 B / 1000 B',
                       'verifying sha256 digest', 'success']
    assert cli_server['calls'] == [('POST', '/api/pull', {'model': 'org/repo/file', 'stream': True})]


@pytest.mark.parametrize('mode', ['redirected', 'dumb', 'encoding', 'json'])
def test_nonanimated_modes_preserve_existing_output(cli_server, terminal, capsys, monkeypatch, mode):
    terminal = terminal()
    if mode == 'redirected':
        terminal.tty = False
    elif mode == 'dumb':
        monkeypatch.setenv('TERM', 'dumb')
    elif mode == 'encoding':
        class AsciiTerminal(Terminal):
            encoding = 'ascii'
        terminal = AsciiTerminal()
        monkeypatch.setattr('sys.stderr', terminal)
    events = [{'status': 'downloading model', 'completed': n, 'total': 3} for n in (0, 1, 1, 3)]
    events.append({'status': 'success'})
    _events(cli_server, events)
    args = ['pull', 'model', '--host', cli_server['url']] + (['--json'] if mode == 'json' else [])
    assert main(args) == 0
    output = capsys.readouterr()
    assert '\r' not in terminal.getvalue() and '\x1b' not in terminal.getvalue()
    if mode == 'json':
        assert [json.loads(line) for line in output.out.splitlines()] == events
        assert terminal.getvalue() == ''
    else:
        assert output.out == 'pulled: model\n'
        assert terminal.getvalue().splitlines() == [
            'downloading model: 0 B / 3 B', 'downloading model: 1 B / 3 B',
            'downloading model: 3 B / 3 B', 'success']


async def test_spinner_waits_before_first_event_and_tracks_unknown_and_changing_shard_totals(terminal):
    terminal = terminal()
    async def progress(model):
        await terminal.until('resolving repository')
        yield {'status': 'downloading model', 'completed': 10}
        await terminal.until('10 B / unknown')
        assert '#' not in terminal.getvalue()
        yield {'status': 'downloading model', 'completed': 10, 'total': 100}
        await terminal.until('[##------------------] 10%')
        # A newly discovered shard changes the total; do not sum cumulative events.
        yield {'status': 'downloading model', 'completed': 50, 'total': 200}
        await terminal.until('50 B / 200 B [#####---------------] 25%')
        yield {'status': 'downloading model', 'completed': 200, 'total': 200}
        await terminal.until('[####################] 100%')
        yield {'status': 'success'}
    class Client:
        pull = staticmethod(progress)
    await pull_progress(Client(), 'model', as_json=False, decimal_units=False)
    assert terminal.getvalue().endswith('downloading model: 200 B / 200 B\nsuccess\n')
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                and t.get_coro().__qualname__ == 'WaitStatus._animate']


@pytest.mark.parametrize('total', [None, 0, -1, True, '100', float('nan'), float('inf')])
async def test_unknown_or_invalid_total_does_not_divide_or_fabricate_percentage(terminal, total):
    terminal = terminal()
    async def progress(model):
        yield {'status': 'downloading model', 'completed': 0, 'total': total}
        await terminal.until('downloading model: 0 B / ')
        assert '%' not in terminal.getvalue() and '[' not in terminal.getvalue()
        yield {'status': 'success'}
    class Client:
        pull = staticmethod(progress)
    await pull_progress(Client(), 'model', as_json=False, decimal_units=False)


@pytest.mark.parametrize('columns', [1, 20, 45, 100])
async def test_narrow_terminal_omits_bar_and_bounds_each_animated_line(terminal, monkeypatch, columns):
    terminal = terminal()
    monkeypatch.setattr('tllama.cli_pull.shutil.get_terminal_size', lambda: os.terminal_size((columns, 24)))
    async def progress(model):
        yield {'status': 'downloading model', 'completed': 50, 'total': 100}
        if columns >= 3:
            await terminal.until('\r')
        yield {'status': 'success'}
    class Client:
        pull = staticmethod(progress)
    await pull_progress(Client(), 'model', as_json=False, decimal_units=False)
    frames = [part for part in terminal.getvalue().split('\r') if part and part[0] in FRAMES]
    if columns < 3:
        assert not frames and '\r' not in terminal.getvalue()
        return
    assert frames
    assert all(len(frame) <= max(columns - 1, 2) for frame in frames)
    if columns <= 45:
        assert '%' not in ''.join(frames)


@pytest.mark.parametrize('decimal_units,expected', [(False, '1.5 KiB / 2.9 KiB'), (True, '1.5 kB / 3.0 kB')])
async def test_terminal_sizes_respect_unit_setting(terminal, decimal_units, expected):
    terminal = terminal()
    async def progress(model):
        yield {'status': 'downloading model', 'completed': 1500, 'total': 3000}
        await terminal.until(expected)
        yield {'status': 'success'}
    class Client:
        pull = staticmethod(progress)
    await pull_progress(Client(), 'model', as_json=False, decimal_units=decimal_units)


@pytest.mark.parametrize('ending', ['error', 'malformed', 'eof', 'cancel'])
async def test_failure_or_cancel_clears_indicator_and_closes_http_response(terminal, ending):
    terminal = terminal()
    class Stream(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            yield b'{"status":"downloading model","completed":10,"total":100}\n'
            await terminal.until('10 B / 100 B')
            if ending == 'error':
                yield b'{"error":"download failed"}\n'
            elif ending == 'malformed':
                yield b'not json\n'
            elif ending == 'cancel':
                raise asyncio.CancelledError
        async def aclose(self):
            self.closed = True
    stream = Stream()
    async with TLlamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))) as client:
        with pytest.raises(asyncio.CancelledError if ending == 'cancel' else ClientError):
            await pull_progress(client, 'model', as_json=False, decimal_units=False)
    assert stream.closed
    assert 'success' not in terminal.getvalue()
    assert terminal.getvalue().endswith('\r')
    snapshot = terminal.getvalue()
    await asyncio.sleep(0.01)
    assert terminal.getvalue() == snapshot


async def test_cancelling_pending_pull_joins_spinner_and_closes_transport(terminal):
    terminal = terminal()
    class Stream(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            yield b'{"status":"downloading model"}\n'
            await asyncio.Event().wait()
        async def aclose(self):
            self.closed = True
    stream = Stream()
    async with TLlamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))) as client:
        task = asyncio.create_task(pull_progress(client, 'model', as_json=False, decimal_units=False))
        try:
            await terminal.until('downloading model')
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert stream.closed and terminal.getvalue().endswith('\r')
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                and t.get_coro().__qualname__ == 'WaitStatus._animate']


async def test_shorter_label_and_clear_remove_the_actually_drawn_width(terminal):
    terminal = terminal()
    async with WaitStatus('a long wait label') as status:
        await terminal.until('a long wait label')
        status.label = 'short'
        await terminal.until('short' + ' ' * 12)
        status.label = 'not yet drawn and much longer'
        status.clear()
        assert terminal.getvalue().endswith('\r' + ' ' * 7 + '\r')
    assert status.task.done()


def test_already_downloaded_weights_require_no_intermediate_progress(cli_server, terminal, capsys):
    terminal = terminal()
    _events(cli_server, [{'status': 'success'}])
    assert main(['pull', 'model', '--host', cli_server['url']]) == 0
    assert terminal.history() == ['success']
    assert capsys.readouterr().out == 'pulled: model\n'
