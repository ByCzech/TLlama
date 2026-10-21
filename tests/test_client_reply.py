"""Reasoning boundaries and terminal styles never leak into answers or history."""

import asyncio
from types import SimpleNamespace

import pytest

from tllama.cli_reply import ReplyText
from tllama.cli_run import RunSession
from tllama.client import ClientError
from test_client_status import Terminal


@pytest.mark.parametrize('tty,no_color,term,colored', [
    (True, False, 'xterm', True), (False, False, 'xterm', False),
    (True, True, 'xterm', False), (True, False, 'dumb', False),
])
def test_thinking_has_boundaries_and_optional_gray(monkeypatch, tty, no_color, term, colored):
    out, err = Terminal(tty=tty), Terminal(tty=tty)
    monkeypatch.setattr('sys.stdout', out)
    monkeypatch.setattr('sys.stderr', err)
    monkeypatch.setenv('TERM', term)
    if no_color:
        monkeypatch.setenv('NO_COLOR', '')
    else:
        monkeypatch.delenv('NO_COLOR', raising=False)
    renderer = ReplyText()
    renderer.thinking('reason ')
    renderer.thinking('more')
    renderer.answer('answer')
    renderer.close()
    assert out.getvalue() == 'answer\n'
    text = err.getvalue()
    assert text.count('Thinking:') == 1 and text.count('Answer:') == 1
    assert text.replace('\x1b[90m', '').replace('\x1b[0m', '') == 'Thinking:\nreason more\n\nAnswer:\n'
    assert ('\x1b[90m' in text) is colored
    assert ('\x1b[0m' in text) is colored


def test_plain_answer_has_no_extra_label(capsys):
    renderer = ReplyText()
    renderer.thinking('')
    renderer.answer('plain')
    renderer.answer(' text')
    renderer.close()
    renderer.close()
    output = capsys.readouterr()
    assert output.out == 'plain text\n' and output.err == ''


@pytest.mark.parametrize('outcome', ['error', 'cancel'])
async def test_interrupted_thinking_resets_color_and_does_not_enter_history(monkeypatch, outcome):
    err = Terminal()
    monkeypatch.setattr('sys.stderr', err)
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    async def chat(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'thinking': 'partial'}, 'done': False}
        raise (ClientError('failed') if outcome == 'error' else asyncio.CancelledError())
    session = RunSession('model', 'completion')
    with pytest.raises(ClientError if outcome == 'error' else asyncio.CancelledError):
        await session.submit(SimpleNamespace(chat=chat), 'prompt')
    assert err.getvalue().endswith('\x1b[0m\n\n')
    assert 'Answer:' not in err.getvalue() and session.messages == []


@pytest.mark.parametrize('hidden,json_mode', [(False, False), (True, False), (False, True)])
async def test_rendering_does_not_change_conversation_or_json(capsys, hidden, json_mode):
    async def chat(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'thinking': 'raw thinking', 'content': 'raw answer'}, 'done': True}
    session = RunSession('model', 'completion', hide_thinking=hidden, as_json=json_mode)
    await session.submit(SimpleNamespace(chat=chat), 'prompt')
    assert session.messages[-1] == {'role': 'assistant', 'content': 'raw answer', 'thinking': 'raw thinking'}
    output = capsys.readouterr()
    if json_mode:
        assert output.err == '' and 'Thinking:' not in output.out and '\u001b' not in output.out
    elif hidden:
        assert output.err == '' and output.out == 'raw answer\n'
    else:
        assert 'Thinking:\nraw thinking\n\nAnswer:\n' == output.err
        assert output.out == 'raw answer\n'


def test_tool_calls_close_thinking_and_restore_normal_color(monkeypatch):
    err = Terminal()
    monkeypatch.setattr('sys.stderr', err)
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    renderer = ReplyText()
    renderer.thinking('reason\n')
    renderer.tools([{'function': {'name': 'f'}}])
    renderer.close()
    assert '\x1b[0m\nTool calls:' in err.getvalue()
