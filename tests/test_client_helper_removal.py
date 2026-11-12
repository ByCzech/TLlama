"""Ordinary Ollama delete stays compatible; extended delete needs confirmation."""

import json

import httpx
import pytest

from tllama.cli import main
from tllama.client import TLlamaClient, ClientError, ClientConfigError
from test_client_cli import cli_server


async def test_extended_delete_probes_capability_and_sends_control():
    calls = []
    reply = {'status': 'success', 'cleanup': {'removed_helpers': ['projector'], 'unused_helpers': []}}
    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json={'models': [], 'server_features': ['repository_helpers']}
                              if request.method == 'GET' else reply)
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        assert await client.remove('main', remove_unused_helpers=True) == reply
    assert calls == [('GET', '/api/tags', None), ('DELETE', '/api/delete',
                     {'model': 'main', 'remove_unused_helpers': True})]


async def test_unknown_server_capability_stops_before_delete():
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={'models': []})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match='repository_helpers'):
            await client.remove('main', remove_unused_helpers=True)
    assert calls == ['GET']


@pytest.mark.parametrize('body', [b'', b'{"status":"success"}',
    b'{"status":"success","cleanup":{"removed_helpers":[],"unused_helpers":1}}'])
async def test_extended_delete_requires_nonempty_structured_confirmation(body, capsys):
    def handler(request):
        return (httpx.Response(200, json={'models': [], 'server_features': ['repository_helpers']})
                if request.method == 'GET' else httpx.Response(200, content=body))
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError):
            await client.remove('main', remove_unused_helpers=True)
    assert 'Ollama daemon' not in capsys.readouterr().err


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
async def test_bad_flag_fails_without_http(value):
    def no_http(request):
        raise AssertionError('Invalid controls must not send a request')
    async with TLlamaClient(transport=httpx.MockTransport(no_http)) as client:
        with pytest.raises(ClientConfigError):
            await client.remove('main', remove_unused_helpers=value)


@pytest.mark.parametrize('as_json', [False, True])
def test_cli_reports_removed_and_preserved_helpers_without_polluting_stdout(cli_server, capsys, as_json):
    response = {'status': 'success', 'cleanup': {'removed_helpers': ['projector'], 'unused_helpers': ['draft'],
                                               'warnings': [], 'deferred_groups': ['group']}}
    cli_server['responses']['/api/tags'] = (200, {'models': [], 'server_features': ['repository_helpers']})
    cli_server['responses']['/api/delete'] = (200, response)
    args = ['rm', 'main', '--remove-unused-helpers', '--host', cli_server['url']]
    assert main(args + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    assert 'removed helper: projector' in output.err and 'unused helper kept: draft' in output.err
    assert 'physical cleanup deferred' in output.err
    assert (json.loads(output.out) == {'model': 'main', 'response': response}) if as_json else output.out == 'deleted: main\n'
    assert cli_server['calls'] == [('GET', '/api/tags', None),
                                  ('DELETE', '/api/delete', {'model': 'main', 'remove_unused_helpers': True})]


def test_default_cli_delete_reports_kept_helpers_without_capability_probe(cli_server, capsys):
    cli_server['responses']['/api/delete'] = (200, {'status': 'success',
        'cleanup': {'removed_helpers': [], 'unused_helpers': ['projector']}})
    assert main(['rm', 'main', '--host', cli_server['url']]) == 0
    output = capsys.readouterr()
    assert output.out == 'deleted: main\n' and 'unused helper kept: projector' in output.err
    assert cli_server['calls'] == [('DELETE', '/api/delete', {'model': 'main'})]
