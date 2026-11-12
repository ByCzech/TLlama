"""Explicit helper listing needs an advertised, correctly typed response."""

import json

import httpx
import pytest

from tllama.cli import main
from tllama.client import TLlamaClient, ClientError, ClientConfigError
from test_client_cli import cli_server


def row(kind):
    return {'name': kind + '-model', 'digest': 'a' * 64, 'size': 1024 ** 3,
            'model_definition': {'type': kind}}


@pytest.mark.parametrize('kind', ['main', 'mmproj', 'draft', 'all'])
@pytest.mark.parametrize('empty', [False, True])
async def test_filtered_list_probes_then_preserves_typed_server_response(kind, empty):
    calls = []
    models = [] if empty else [row(value) for value in ('main', 'mmproj', 'draft') if kind in (value, 'all')]
    data = {'models': models, 'server_features': ['repository_helpers'], 'extra': 'preserve'}
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=data if request.url.query else
                              {'models': [], 'server_features': ['repository_helpers']})
    async with TLlamaClient('http://example/proxy', transport=httpx.MockTransport(handler)) as client:
        assert await client.list_models(model_type=kind) == data
    assert calls == ['http://example:54800/proxy/api/tags', 'http://example:54800/proxy/api/tags?type=' + kind]


@pytest.mark.parametrize('features', [None, [], 'repository_helpers', ['other'], ['repository_helpers', 1]])
async def test_unknown_support_does_not_send_a_filter_even_when_server_would_ignore_it(features):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={'models': [], 'server_features': features})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match='repository_helpers'):
            await client.list_models(model_type='all')
    assert len(calls) == 1 and '?' not in calls[0]


@pytest.mark.parametrize('kind, models', [
    ('mmproj', [row('main')]), ('main', [row('draft')]),
    ('all', [{'name': 'unknown'}]), ('draft', [{'model_definition': {'type': 'unknown'}}]),
    ('all', [{'model_definition': None}]), ('all', [{'model_definition': {'type': []}}]),
])
async def test_ignored_filter_or_unknown_row_type_is_not_presented_as_success(kind, models):
    def handler(request):
        return httpx.Response(200, json={'models': models if request.url.query else [],
                                        'server_features': ['repository_helpers']})
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match='model type filter'):
            await client.list_models(model_type=kind)


@pytest.mark.parametrize('kind', ['', 'MAIN', 'other', True, 1, [], {}])
async def test_invalid_filter_fails_without_http(kind):
    def forbidden(request):
        raise AssertionError('Invalid model type must not cause HTTP requests')
    async with TLlamaClient(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ClientConfigError):
            await client.list_models(model_type=kind)


@pytest.mark.parametrize('command', ['list', 'ls'])
@pytest.mark.parametrize('flags, kind', [(['--all'], 'all'), (['--type', 'main'], 'main'),
                                      (['--type', 'mmproj'], 'mmproj'), (['--type', 'draft'], 'draft')])
@pytest.mark.parametrize('as_json', [False, True])
def test_cli_filters_and_type_column_preserve_json_and_units(cli_server, capsys, command, flags, kind, as_json):
    models = [row(value) for value in ('main', 'mmproj', 'draft') if kind in (value, 'all')]
    data = {'models': models, 'server_features': ['repository_helpers']}
    cli_server['responses']['/api/tags'] = (200, {'models': [], 'server_features': ['repository_helpers']})
    cli_server['responses']['/api/tags?type=' + kind] = (200, data)
    args = [command, *flags, '--decimal-units', '--host', cli_server['url']]
    assert main(args + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr()
    if as_json:
        assert json.loads(output.out) == data
    else:
        assert output.out.splitlines()[0].split() == ['NAME', 'TYPE', 'ID', 'SIZE', 'MODIFIED']
        for model in models:
            assert model['name'] in output.out and model['model_definition']['type'] in output.out
        assert '1.1 GB' in output.out and 'aaaaaaaaaaaa' in output.out
    assert cli_server['calls'] == [('GET', '/api/tags', None), ('GET', '/api/tags?type=' + kind, None)]


def test_plain_list_keeps_original_columns_and_single_ollama_compatible_request(cli_server, capsys):
    cli_server['responses']['/api/tags'] = (200, {'models': [row('main')], 'server_features': ['repository_helpers']})
    assert main(['list', '--host', cli_server['url']]) == 0
    assert capsys.readouterr().out.splitlines()[0].split() == ['NAME', 'ID', 'SIZE', 'MODIFIED']
    assert cli_server['calls'] == [('GET', '/api/tags', None)]


def test_cli_without_server_support_does_not_print_a_filtered_table(cli_server, capsys):
    cli_server['responses']['/api/tags'] = (200, {'models': []})
    assert main(['list', '--all', '--host', cli_server['url']]) == 1
    output = capsys.readouterr()
    assert output.out == '' and 'repository_helpers' in output.err
    assert cli_server['calls'] == [('GET', '/api/tags', None)]


@pytest.mark.parametrize('flags', [['--all', '--type', 'draft'], ['--type', 'all'], ['--type', 'other']])
def test_cli_filter_controls_are_validated_before_contacting_server(flags):
    with pytest.raises(SystemExit) as caught:
        main(['list', *flags])
    assert caught.value.code == 2
