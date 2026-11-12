"""Helper copy requires an explicit server capability before any mutation."""

import json

import httpx
import pytest

from tllama.cli import main
from tllama.client import TLlamaClient, ClientError, ClientConfigError
from test_client_cli import cli_server


@pytest.mark.parametrize('controls', [{'mmproj': 'projector'}, {'draft': 'draft'},
                                     {'mmproj': 'projector', 'draft': 'draft'}])
async def test_copy_checks_support_and_sends_only_explicit_bindings(controls):
    calls = []
    def reply(request):
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json=({'models': [], 'server_features': ['repository_helpers']}
                                        if request.method == 'GET' else {'status': 'success'}))
    async with TLlamaClient(transport=httpx.MockTransport(reply)) as client:
        assert await client.copy('source', 'destination', force=True, **controls) == {'status': 'success'}
    assert calls == [('GET', '/api/tags', None), ('POST', '/api/copy',
                     {'source': 'source', 'destination': 'destination', 'force': True, **controls})]


@pytest.mark.parametrize('features', [None, [], 'repository_helpers', {'repository_helpers': True},
                                     ['other'], ['repository_helpers', 1]])
async def test_unknown_or_malformed_server_support_prevents_post(features):
    calls = []
    def reply(request):
        calls.append(request.method)
        # A permissive server might otherwise ignore unknown copy fields and
        # report success. Never send the mutation when support is absent.
        return httpx.Response(200, json={'models': [], 'server_features': features, 'status': 'success'})
    async with TLlamaClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(ClientError, match='does not advertise repository_helpers'):
            await client.copy('source', 'destination', mmproj='projector')
    assert calls == ['GET']


async def test_plain_copy_keeps_existing_wire_contract_without_capability_probe():
    calls = []
    def reply(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'status': 'success'})
    async with TLlamaClient(transport=httpx.MockTransport(reply)) as client:
        await client.copy('source', 'destination')
    assert calls == [('POST', '/api/copy', {'source': 'source', 'destination': 'destination', 'force': False})]


@pytest.mark.parametrize('name', ['', '../outside', '/absolute', 'a/b/c/d', 'bad\\path'])
async def test_bad_helper_name_fails_before_http(name):
    def no_http(request):
        raise AssertionError('Invalid helper names must not cause any request')
    async with TLlamaClient(transport=httpx.MockTransport(no_http)) as client:
        with pytest.raises(ClientConfigError):
            await client.copy('source', 'destination', draft=name)


def test_cli_passes_named_attachment_controls_after_probe(cli_server, capsys):
    cli_server['responses']['/api/tags'] = (200, {'models': [], 'server_features': ['repository_helpers']})
    cli_server['responses']['/api/copy'] = (200, {'status': 'success'})
    assert main(['cp', 'source', 'destination', '--mmproj', 'ByCzech/Projector:Q8%',
                 '--draft', 'draft', '--force', '--json', '--host', cli_server['url']]) == 0
    assert cli_server['calls'] == [('GET', '/api/tags', None), ('POST', '/api/copy',
        {'source': 'source', 'destination': 'destination', 'force': True,
         'mmproj': 'ByCzech/Projector:Q8%', 'draft': 'draft'})]
    assert json.loads(capsys.readouterr().out) == {'status': 'success'}


def test_cli_against_unsupported_server_does_not_report_a_copy(cli_server, capsys):
    cli_server['responses']['/api/tags'] = (200, {'models': []})
    cli_server['responses']['/api/copy'] = (200, {'status': 'success'})
    assert main(['cp', 'source', 'destination', '--draft', 'draft', '--host', cli_server['url']]) == 1
    output = capsys.readouterr()
    assert output.out == '' and 'repository_helpers' in output.err
    assert cli_server['calls'] == [('GET', '/api/tags', None)]
