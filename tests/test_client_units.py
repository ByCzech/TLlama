"""One client display choice covers tables and transfers without changing bytes."""

import json
from types import SimpleNamespace

import pytest

from tllama.cli import main
from tllama.cli_output import size_text
from tllama.client import ClientConfigError
from tllama.client_config import load_client_config
from tllama.client_import import create_from_file
from test_client_cli import cli_server, isolated_proxy_environment
from test_client_config import files


@pytest.mark.parametrize('value,expected', [
    (0, '0 B'), (1000, '1000 B'), (1023, '1023 B'), (1024, '1.0 KiB'),
    (1024**2, '1.0 MiB'), (32 * 1024**3, '32.0 GiB'), (1024**4, '1.0 TiB'),
])
def test_binary_thresholds_and_labels(value, expected):
    assert size_text(value) == expected


@pytest.mark.parametrize('value,expected', [
    (0, '0 B'), (999, '999 B'), (1000, '1.0 kB'), (1000**2, '1.0 MB'),
    (32 * 1024**3, '34.4 GB'), (1000**4, '1.0 TB'),
])
def test_decimal_thresholds_and_labels(value, expected):
    assert size_text(value, decimal_units=True) == expected


@pytest.mark.parametrize('bad', [None, True, -1, '1024', float('nan'), float('inf')])
def test_unknown_sizes_stay_unknown_in_both_modes(bad):
    assert size_text(bad) == size_text(bad, decimal_units=True) == 'unknown'


def test_false_overrides_true_without_discarding_other_settings(files):
    files[0].write_text('[client]\ntimeout=90\n[display]\ndecimal_units=true\n')
    files[1].write_text('[display]\ndecimal_units=false\n')
    config = load_client_config(environ={})
    assert config.decimal_units is False and config.timeout == 90


@pytest.mark.parametrize('value', ['"false"', '0', '1', '[]'])
def test_decimal_setting_requires_a_boolean(files, value):
    files[1].write_text(f'[display]\ndecimal_units={value}\n')
    with pytest.raises(ClientConfigError, match='display.decimal_units'):
        load_client_config(environ={})


@pytest.mark.parametrize('command,path', [('list', '/api/tags'), ('ps', '/api/ps')])
@pytest.mark.parametrize('flags,label', [([], '32.0 GiB'), (['--decimal-units'], '34.4 GB')])
def test_tables_use_chosen_units_and_preserve_json(cli_server, capsys, command, path, flags, label):
    payload = {'models': [{'name': 'm', 'size': 32 * 1024**3, 'size_vram': 32 * 1024**3}]}
    cli_server['responses'][path] = (200, payload)
    assert main([command, '--host', cli_server['url'], *flags]) == 0
    assert label in capsys.readouterr().out
    assert main([*flags, command, '--host', cli_server['url'], '--json']) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_config_and_both_cli_overrides_do_not_leak_between_invocations(files, cli_server, capsys):
    files[1].write_text('[display]\ndecimal_units=true\n')
    cli_server['responses']['/api/tags'] = (200, {'models': [{'name': 'm', 'size': 1024**3}]})
    for flags, expected in [([], '1.1 GB'), (['--no-decimal-units'], '1.0 GiB'),
                            ([], '1.1 GB'), (['--no-decimal-units', '--decimal-units'], '1.1 GB')]:
        assert main([*flags, 'list', '--host', cli_server['url']]) == 0
        assert expected in capsys.readouterr().out


@pytest.mark.parametrize('decimal,label', [(False, '1.0 KiB'), (True, '1.0 kB')])
def test_pull_uses_configured_units(files, cli_server, capsys, decimal, label):
    files[1].write_text(f'[display]\ndecimal_units={str(decimal).lower()}\n')
    events = [{'status': 'downloading', 'completed': 1024, 'total': 1024}, {'status': 'success'}]
    cli_server['responses']['/api/pull'] = (200, ('\n'.join(map(json.dumps, events)) + '\n').encode())
    assert main(['pull', 'm', '--host', cli_server['url']]) == 0
    assert f'{label} / {label}' in capsys.readouterr().err


@pytest.mark.parametrize('decimal,label', [(False, '1.0 KiB'), (True, '1.0 kB')])
async def test_upload_formats_size_but_sends_original_file(tmp_path, capsys, decimal, label):
    import gguf
    model = tmp_path / 'model.gguf'
    writer = gguf.GGUFWriter(str(model), arch='llama')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    payload = model.read_bytes().ljust(1024, b'\0')
    model.write_bytes(payload)
    definition = tmp_path / 'model.toml'
    definition.write_text(f'[llm]\nfrom={json.dumps(str(model))}\n')
    transferred = bytearray()
    async def missing(digest):
        return False
    async def upload(digest, stream):
        async for chunk in stream:
            transferred.extend(chunk)
    async def create(*args, **kwargs):
        return {'status': 'success'}
    client = SimpleNamespace(check_create=create, blob_exists=missing, upload_blob=upload, create=create)
    assert await create_from_file(client, 'm', definition, decimal_units=decimal) == {'status': 'success'}
    assert f'({label})' in capsys.readouterr().err and transferred == payload
