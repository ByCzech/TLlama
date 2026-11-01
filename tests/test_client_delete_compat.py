"""Ollama-style empty DELETE success must remain visible and interoperable."""

import json

import httpx
import pytest

from tllama.cli import main
from tllama.client import ClientError, TLlamaClient
from test_client_cli import cli_server, isolated_proxy_environment


@pytest.mark.parametrize('as_json', [False, True])
@pytest.mark.parametrize('status', [200, 204])
def test_empty_delete_succeeds_warns_once_and_preserves_stdout(cli_server, capsys, status, as_json):
    cli_server['responses']['/api/delete'] = (status, b'')
    arguments = ['rm', 'first', 'second', '--host', cli_server['url']]
    assert main(arguments + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    assert output.err.count('Upozornění:') == 1
    assert 'Pravděpodobně používáte Ollama daemon.' in output.err
    if as_json:
        assert [json.loads(line) for line in output.out.splitlines()] == [
            {'model': model, 'response': {'status': 'success'}} for model in ('first', 'second')]
    else:
        assert output.out == 'deleted: first\ndeleted: second\n'
    assert cli_server['calls'] == [('DELETE', '/api/delete', {'model': model})
                                  for model in ('first', 'second')]


def test_a_new_cli_command_warns_again(cli_server, capsys):
    cli_server['responses']['/api/delete'] = (200, b'')
    for model in ('first', 'second'):
        assert main(['rm', model, '--host', cli_server['url']]) == 0
        assert capsys.readouterr().err.count('Upozornění:') == 1


@pytest.mark.parametrize('status,body', [
    (404, b''), (500, b''), (200, b'{}'), (200, b' '), (200, b'not json'),
    (200, b'[]'), (200, b'{"status":"pending"}'), (200, b'{"error":"failed"}'),
])
def test_failed_or_nonempty_unconfirmed_delete_is_not_accepted(cli_server, capsys, status, body):
    cli_server['responses']['/api/delete'] = (status, body)
    assert main(['rm', 'a', '--host', cli_server['url']]) == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert 'Upozornění:' not in output.err and 'Ollama daemon' not in output.err


def test_empty_success_does_not_hide_failure_for_another_model(cli_server, capsys):
    def reply(method, body):
        return (404, b'') if body['model'] == 'missing' else (200, b'')
    cli_server['responses']['/api/delete'] = reply
    assert main(['rm', 'first', 'missing', 'last', '--json', '--host', cli_server['url']]) == 1
    output = capsys.readouterr()
    records = [json.loads(line) for line in output.out.splitlines()]
    assert [record['model'] for record in records] == ['first', 'missing', 'last']
    assert records[0]['response'] == records[2]['response'] == {'status': 'success'}
    assert 'error' in records[1]
    assert output.err.count('Upozornění:') == 1


async def test_json_success_has_no_warning_and_retains_server_details(capsys):
    reply = {'status': 'success', 'deleted': 'Local/a.toml', 'kept_model_file': 'data/a/weights.gguf'}
    async with TLlamaClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply))) as client:
        assert await client.remove('a') == reply
    assert capsys.readouterr().err == ''


async def test_empty_other_endpoints_still_fail_json_validation(capsys):
    async with TLlamaClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        for operation in (client.version, client.list_models, client.ps):
            with pytest.raises(ClientError, match='invalid JSON'):
                await operation()
    assert capsys.readouterr().err == ''
