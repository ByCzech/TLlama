"""Run dispatch, literal prompts and transactional conversation state."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from tllama import cli
from tllama.cli_run import RunSession, load_json
from tllama.client import ClientError
from test_client_cli import cli_server, isolated_proxy_environment


class Input(io.StringIO):
    def __init__(self, text='', *, tty=False):
        super().__init__(text)
        self.tty = tty

    def isatty(self):
        return self.tty


@pytest.mark.parametrize('mode,path,reply,input_text', [
    ('completion', '/api/chat', b'{"message":{"role":"assistant","content":"ahoj"},"done":false}\n'
     b'{"message":{"role":"assistant","content":""},"done":true}\n', 'prompt'),
    ('embedding', '/api/embed', {'embeddings': [[0.1, 0.2]]}, 'embedding text'),
    ('decision', '/v1/systemone', {'answers': {'q': {}}, 'usage': {}},
     '{"state":{"x":1},"questions":{"q":{"type":"score"}}}'),
])
def test_run_dispatches_capabilities(cli_server, monkeypatch, capsys, mode, path, reply, input_text):
    monkeypatch.setattr('sys.stdin', Input(input_text))
    cli_server['responses']['/api/show'] = (200, {'capabilities': [mode]})
    cli_server['responses'][path] = (200, reply)
    assert cli.main(['run', 'model', '--host', cli_server['url']]) == 0
    calls = cli_server['calls']
    assert [row[1] for row in calls] == ['/api/show', path]
    body = calls[-1][2]
    assert body['model'] == 'model'
    if mode == 'completion':
        assert body == {'model': 'model', 'stream': True,
                        'messages': [{'role': 'user', 'content': input_text}]}
        assert capsys.readouterr().out == 'ahoj\n'
    elif mode == 'embedding':
        assert body == {'model': 'model', 'input': input_text}
        assert json.loads(capsys.readouterr().out) == reply
    else:
        assert body == {'model': 'model', **json.loads(input_text)}
        assert json.loads(capsys.readouterr().out) == reply


def test_run_keeps_prompt_literal_and_emits_ndjson(cli_server, monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', Input('piped\n'))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    reply = {'message': {'role': 'assistant', 'content': 'česky', 'thinking': 'reason'}, 'done': True}
    cli_server['responses']['/api/chat'] = (200, (json.dumps(reply) + '\n').encode())
    assert cli.main(['--json', 'run', 'model', '  /a "b"\n', '--host', cli_server['url'],
                     '--option', 'temperature=0', '--option', 'num_predict=0', '--think', 'false',
                     '--format', 'json', '--keep-alive', '0', '--system', 'system']) == 0
    assert cli_server['calls'][-1][2] == {
        'model': 'model', 'stream': True, 'messages': [{'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'piped\n\n\n  /a "b"\n'}],
        'options': {'temperature': 0, 'num_predict': 0}, 'think': False, 'format': 'json', 'keep_alive': '0'}
    output = capsys.readouterr()
    assert json.loads(output.out) == reply and output.err == ''


@pytest.mark.parametrize('args', [
    ['--option', 'missing'], ['--option', 'x=NaN'], ['--option', 'x=1e999'],
    ['--option', 'x={"a":1,"a":2}'], ['--format', '[]'],
])
def test_run_rejects_invalid_controls_before_network(args):
    with pytest.raises(SystemExit) as exc:
        cli.main(['run', 'model', 'prompt', *args])
    assert exc.value.code == 2


@pytest.mark.parametrize('mode,args', [
    ('embedding', ['--system', 's']), ('embedding', ['--think', 'false']),
    ('embedding', ['--format', 'json']), ('decision', ['--option', 'temperature=1']),
    ('decision', ['--keep-alive', '0']),
])
def test_run_rejects_mode_inappropriate_controls(cli_server, monkeypatch, mode, args):
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': [mode]})
    assert cli.main(['run', 'model', 'input', '--host', cli_server['url'], *args]) == 1
    assert [row[1] for row in cli_server['calls']] == ['/api/show']


@pytest.mark.parametrize('failure', ['error', 'cancel'])
async def test_incomplete_turn_never_enters_conversation(failure, capsys):
    calls = []
    closed = []
    async def chat(model, messages, **controls):
        calls.append(messages)
        try:
            yield {'message': {'role': 'assistant', 'content': 'partial', 'thinking': 'private'}, 'done': False}
            raise (ClientError('failed') if failure == 'error' else asyncio.CancelledError())
        finally:
            closed.append(True)
    session = RunSession('model', 'completion', system='s')
    with pytest.raises(ClientError if failure == 'error' else asyncio.CancelledError):
        await session.submit(SimpleNamespace(chat=chat), 'prompt')
    assert session.messages == [] and closed == [True]


async def test_completed_turn_includes_thinking_and_calls_only_once(capsys):
    calls = []
    async def chat(model, messages, **controls):
        calls.append(messages)
        yield {'message': {'role': 'assistant', 'thinking': 'hidden'}, 'done': False}
        yield {'message': {'role': 'assistant', 'content': 'answer',
                          'tool_calls': [{'function': {'name': 'f'}}]}, 'done': True}
    session = RunSession('model', 'completion', hide_thinking=True)
    client = SimpleNamespace(chat=chat)
    await session.submit(client, 'first')
    await session.submit(client, 'second')
    assert len(calls) == 2 and len(calls[1]) == 3
    assert calls[1][1] == {'role': 'assistant', 'content': 'answer', 'thinking': 'hidden',
                            'tool_calls': [{'function': {'name': 'f'}}]}
    output = capsys.readouterr()
    assert 'hidden' not in output.out + output.err


@pytest.mark.parametrize('text', ['NaN', '1e999', '{"x":1,"x":2}'])
def test_input_json_does_not_silently_rewrite_decision_contract(text):
    with pytest.raises(ValueError):
        load_json(text)
