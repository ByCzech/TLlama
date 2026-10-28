"""Display defaults stay local and can be overridden per run or REPL session."""

import json

import pytest

from tllama.cli import main
from tllama.client import ClientConfigError
from tllama.client_config import load_client_config
from test_client_cli import cli_server, isolated_proxy_environment
from test_client_config import files
from test_client_run import Input


def test_display_preferences_merge_by_key_and_explicit_false_wins(files):
    files[0].write_text('[client]\nhost="system:1234"\n[display]\nverbose=true\nhide_thinking=true\n')
    files[1].write_text('[display]\nverbose=false\n')
    config = load_client_config(environ={})
    assert config.host == 'system:1234' and config.hide_thinking and not config.verbose
    assert not config.decimal_units


def test_user_can_replace_system_server_without_repeating_display_defaults(files):
    files[0].write_text('[client]\nhost="system:1234"\n[display]\nverbose=true\n')
    files[1].write_text('[client]\nhost="user:2345"\n')
    config = load_client_config(environ={})
    assert config.host == 'user:2345' and config.verbose


@pytest.mark.parametrize('key', ['verbose', 'hide_thinking'])
@pytest.mark.parametrize('value', ['"false"', '0', '1', '[]'])
def test_display_settings_require_real_booleans(files, key, value):
    files[1].write_text(f'[display]\n{key}={value}\n')
    with pytest.raises(ClientConfigError, match=f'display.{key}'):
        load_client_config(environ={})


@pytest.mark.parametrize('flags,hidden,verbose', [
    ([], True, True), (['--show-thinking', '--no-verbose'], False, False),
    (['--show-thinking', '--hide-thinking', '--no-verbose', '-v'], True, True),
])
def test_completion_preferences_only_change_rendering(files, cli_server, monkeypatch, capsys,
                                                     flags, hidden, verbose):
    files[1].write_text('[display]\nverbose=true\nhide_thinking=true\n')
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    reply = {'message': {'role': 'assistant', 'content': 'answer', 'thinking': 'private reason'},
             'done': True, 'eval_count': 3, 'eval_duration': 1_000_000_000}
    cli_server['responses']['/api/chat'] = (200, (json.dumps(reply) + '\n').encode())
    assert main(['run', 'm', 'question', '--host', cli_server['url'], *flags]) == 0
    output = capsys.readouterr()
    assert output.out == 'answer\n'
    assert ('private reason' not in output.err) is hidden
    assert ('Statistics:' in output.err) is verbose
    assert cli_server['calls'][-1][2] == {'model': 'm', 'stream': True,
                                        'messages': [{'role': 'user', 'content': 'question'}]}


def test_json_preserves_hidden_reasoning_and_sends_statistics_only_to_stderr(files, cli_server,
                                                                         monkeypatch, capsys):
    files[1].write_text('[display]\nverbose=true\nhide_thinking=true\n')
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    event = {'message': {'role': 'assistant', 'thinking': 'reason', 'content': 'answer'}, 'done': True}
    cli_server['responses']['/api/chat'] = (200, (json.dumps(event) + '\n').encode())
    assert main(['run', 'm', 'question', '--json', '--host', cli_server['url']]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == event and 'Statistics:' in output.err


@pytest.mark.parametrize('mode,path,payload,text', [
    ('embedding', '/api/embed', {'embeddings': [[1.0]]}, 'question'),
    ('decision', '/v1/systemone', {'answers': {}, 'usage': {'input_tokens': 2}},
     '{"state":{},"questions":{}}'),
])
def test_completion_display_default_does_not_block_other_model_modes(files, cli_server,
                                                                   monkeypatch, capsys, mode, path, payload, text):
    files[1].write_text('[display]\nhide_thinking=true\nverbose=true\n')
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': [mode]})
    cli_server['responses'][path] = (200, payload)
    assert main(['run', 'm', text, '--host', cli_server['url']]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == payload and 'Statistics:' in output.err
    assert set(cli_server['calls'][-1][2]) == ({'model', 'input'} if mode == 'embedding'
                                             else {'model', 'state', 'questions'})


def test_repl_starts_with_config_then_changes_only_current_session(files, cli_server,
                                                                 monkeypatch, capsys):
    from tllama import cli_repl
    files[1].write_text('[display]\nverbose=true\nhide_thinking=true\n')
    original = files[1].read_text()
    # The fake REPL exercises the real session controls after CLI/preload.
    async def inspect(client, session):
        assert session.verbose and session.hide_thinking
        await cli_repl.command(client, session, '/set quiet')
        assert not session.verbose
        await cli_repl.command(client, session, '/set verbose')
        assert session.verbose
        return 0
    monkeypatch.setattr(cli_repl, 'repl', inspect)
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    cli_server['responses']['/api/generate'] = (200, {'done': True, 'done_reason': 'load'})
    assert main(['run', 'm', '--host', cli_server['url']]) == 0
    assert files[1].read_text() == original
