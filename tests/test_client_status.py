"""Exact spinner frames, bounded lifetime and preload before interactive input."""

import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from tllama.cli import build_parser
from tllama.cli_run import RunSession, run
from tllama.cli_status import FRAMES, WaitStatus
from tllama.client import ClientError, TLlamaClient


class Terminal(io.StringIO):
    def __init__(self, *, tty=True):
        super().__init__()
        self.tty = tty
        self.frames = []
        self.eight = asyncio.Event()

    def isatty(self):
        return self.tty

    def write(self, text):
        if text.startswith('\r') and len(text) > 1 and text[1] in FRAMES:
            self.frames.append(text[1])
            if len(self.frames) >= 8:
                self.eight.set()
        return super().write(text)


async def test_selected_spinner_frames_are_ordered_and_joined(monkeypatch):
    output = Terminal()
    monkeypatch.setattr('sys.stderr', output)
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr('tllama.cli_status.INTERVAL', 0)
    async with WaitStatus('Waiting') as status:
        await asyncio.wait_for(output.eight.wait(), 1)
        status.stop()
        after_stop = output.getvalue()
        await asyncio.sleep(0)
        assert output.getvalue() == after_stop
    assert ''.join(output.frames[:8]) == '⣷⣯⣟⡿⢿⣻⣽⣾'
    assert output.getvalue().endswith('\r' + ' ' * 9 + '\r')
    assert status.task.done()


@pytest.mark.parametrize('tty,enabled,term', [(False, True, 'xterm'), (True, False, 'xterm'),
                                            (True, True, 'dumb')])
async def test_noninteractive_or_json_status_is_quiet(monkeypatch, tty, enabled, term):
    output = Terminal(tty=tty)
    monkeypatch.setattr('sys.stderr', output)
    monkeypatch.setenv('TERM', term)
    async with WaitStatus('Waiting', enabled=enabled) as status:
        await asyncio.sleep(0)
    assert output.getvalue() == '' and status.task is None


@pytest.mark.parametrize('outcome', ['error', 'cancel'])
async def test_spinner_cleanup_precedes_error_or_cancellation(monkeypatch, outcome):
    output = Terminal()
    monkeypatch.setattr('sys.stderr', output)
    monkeypatch.setenv('TERM', 'xterm')
    entered = asyncio.Event()
    status = WaitStatus('Waiting')
    async def operation():
        async with status:
            await asyncio.sleep(0)
            entered.set()
            if outcome == 'error':
                raise ClientError('failed')
            await asyncio.Event().wait()
    task = asyncio.create_task(operation())
    await entered.wait()
    if outcome == 'cancel':
        task.cancel()
    with pytest.raises(ClientError if outcome == 'error' else asyncio.CancelledError):
        await task
    assert status.task.done() and status.stopped and not status.drawn


@pytest.mark.parametrize('reason,loaded', [('load', True), ('unload', False)])
@pytest.mark.parametrize('timeout', [None, 2.5])
async def test_preload_preserves_runtime_and_residence_controls(reason, loaded, timeout):
    def handler(request):
        assert request.url.path == '/prefix/api/generate'
        assert json.loads(request.content) == {'model': 'model', 'prompt': '', 'stream': False,
                                               'options': {'num_ctx': 4096}, 'keep_alive': '0'}
        assert request.extensions['timeout']['read'] == timeout
        return httpx.Response(200, json={'done': True, 'done_reason': reason})
    async with TLlamaClient('http://localhost/prefix', timeout=timeout,
                            transport=httpx.MockTransport(handler)) as client:
        assert await client.preload('model', options={'num_ctx': 4096}, keep_alive='0') is loaded


@pytest.mark.parametrize('reply', [{'done': False, 'done_reason': 'load'}, {'done': True},
                                 {'done': True, 'done_reason': 'stop'}])
async def test_preload_requires_explicit_confirmation(reply):
    async with TLlamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply))) as client:
        with pytest.raises(ClientError, match='confirm model preparation'):
            await client.preload('model')


@pytest.mark.parametrize('mode', ['completion', 'embedding', 'decision'])
async def test_preload_finishes_before_repl_prompt(monkeypatch, mode):
    events = []
    class Input(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr('sys.stdin', Input())
    async def show(model):
        events.append('show')
        return {'capabilities': [mode]}
    async def preload(model, **kwargs):
        events.append('preload')
        return True
    async def repl(client, session):
        assert events == ['show', 'preload']
        events.append('prompt')
        return 0
    monkeypatch.setattr('tllama.cli_repl.repl', repl)
    assert await run(SimpleNamespace(show=show, preload=preload), build_parser().parse_args(['run', 'model'])) == 0
    assert events[-1] == 'prompt'


async def test_preload_failure_does_not_open_repl(monkeypatch):
    class Input(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr('sys.stdin', Input())
    async def show(model):
        return {'capabilities': ['completion']}
    async def preload(model, **kwargs):
        raise ClientError('load failed')
    with pytest.raises(ClientError, match='load failed'):
        await run(SimpleNamespace(show=show, preload=preload), build_parser().parse_args(['run', 'model']))


async def test_hidden_reasoning_keeps_wait_indicator_until_visible_content(monkeypatch):
    statuses = []
    class Status:
        def __init__(self, *args, **kwargs):
            self.stopped = False
            statuses.append(self)
        async def __aenter__(self):
            return self
        def stop(self):
            self.stopped = True
        async def __aexit__(self, *args):
            self.stop()
    monkeypatch.setattr('tllama.cli_run.WaitStatus', Status)
    async def chat(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'thinking': 'hidden'}, 'done': False}
        assert not statuses[0].stopped
        yield {'message': {'role': 'assistant', 'content': 'answer'}, 'done': False}
        assert statuses[0].stopped
        yield {'message': {'role': 'assistant', 'content': ''}, 'done': True}
    await RunSession('model', 'completion', hide_thinking=True).submit(SimpleNamespace(chat=chat), 'question')
