"""Client configuration selection, precedence and offline error handling."""

from pathlib import Path

import pytest

from tllama import client_config
from tllama.cli import build_parser, main
from tllama.client import ClientConfigError, DEFAULT_HOST
from tllama.client_config import load_client_config
from test_client_cli import cli_server, isolated_proxy_environment


@pytest.fixture
def files(tmp_path, monkeypatch):
    system, user = tmp_path / 'system.toml', tmp_path / 'user.toml'
    monkeypatch.setattr(client_config, 'default_config_paths', lambda env: (system, user))
    return system, user


def test_unset_values_keep_built_in_defaults(files):
    config = load_client_config(environ={})
    assert config.host == DEFAULT_HOST and config.timeout is None
    assert config.session_max_file_mib == 256


def test_session_limit_layers_and_explicit_file_selection(files, tmp_path):
    system, user = files
    system.write_text('[sessions]\nmax_file_mib=512\n')
    user.write_text('[history]\nenabled=false\n')
    assert load_client_config(environ={}).session_max_file_mib == 512
    user.write_text('[sessions]\nmax_file_mib=128\n')
    assert load_client_config(environ={}).session_max_file_mib == 128
    explicit = tmp_path / 'explicit.toml'
    explicit.write_text('[sessions]\nmax_file_mib=64\n')
    assert load_client_config(explicit, environ={}).session_max_file_mib == 64


@pytest.mark.parametrize('value', ['0', '-1', 'true', '256.0', '"256"', '[]', 'nan', '9223372036854775807'])
def test_session_limit_requires_positive_representable_integer_mib(files, value):
    files[0].write_text('[sessions]\nmax_file_mib=' + value + '\n')
    with pytest.raises(ClientConfigError, match='sessions.max_file_mib.*positive integer'):
        load_client_config(environ={})


def test_user_overrides_individual_settings_without_discarding_section(files):
    system, user = files
    system.write_text('[client]\nhost = "system:1234"\ntimeout = 80\n')
    user.write_text('[client]\ntimeout = 120.5\n')
    config = load_client_config(environ={})
    assert config.host == 'system:1234' and config.timeout == 120.5


def test_explicit_file_excludes_both_discovered_files(files, tmp_path):
    for path in files:
        path.write_text('invalid TOML [')
    explicit = tmp_path / 'explicit.toml'
    explicit.write_text('[client]\ntimeout = 45\n')
    config = load_client_config(explicit, environ={})
    assert config.host == DEFAULT_HOST and config.timeout == 45


def test_environment_and_cli_precedence(files):
    files[0].write_text('[client]\nhost = "system:1234"\n')
    files[1].write_text('[client]\nhost = "user:2345"\n')
    assert load_client_config(environ={'TLLAMA_HOST': 'env:3456'}).host == 'env:3456'
    assert load_client_config(environ={'TLLAMA_HOST': ' '}).host == 'user:2345'
    config = load_client_config(overrides={'host': 'cli:4567'}, environ={'TLLAMA_HOST': 'invalid:0'})
    assert config.host == 'cli:4567'


@pytest.mark.parametrize('text', [
    '[client]\nhost = ""', '[client]\nhost = 123', '[client]\nhost = "bad:0"',
    '[client]\ntimeout = true', '[client]\ntimeout = "30"', '[client]\ntimeout = 0',
    '[client]\ntimeout = -1', '[client]\ntimeout = inf', '[client]\ntimeout = nan',
    '[client]\nunknown = true', '[unknown]', 'client = 5', '[client]\ntimeout = [30]',
    '[client.host]\nx = "nested"', '[client',
])
def test_invalid_file_is_an_error_even_when_later_file_is_valid(files, text):
    files[0].write_text(text)
    files[1].write_text('[client]\nhost = "valid:1234"\n')
    with pytest.raises(ClientConfigError, match=str(files[0])):
        load_client_config(environ={})


def test_bad_utf8_is_a_configuration_error(files):
    files[0].write_bytes(b'\xff')
    with pytest.raises(ClientConfigError, match='Cannot read client config'):
        load_client_config(environ={})


def test_missing_explicit_file_does_not_fall_back(files, tmp_path):
    files[0].write_text('[client]\nhost = "system:1234"\n')
    with pytest.raises(ClientConfigError, match='Cannot read client config'):
        load_client_config(tmp_path / 'missing.toml', environ={})


def test_unreadable_candidate_is_not_silently_skipped(files, monkeypatch):
    original = Path.read_text
    def read(path, **kwargs):
        if path == files[0]:
            raise PermissionError('denied')
        return original(path, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(ClientConfigError, match='denied'):
        load_client_config(environ={})


def test_default_search_paths_follow_xdg_and_home(monkeypatch):
    # Undo the test-suite discovery stub for this path-resolution check.
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'home', lambda: Path('/home/tester'))
        # The implementation itself is retained by the module-level reference.
        assert ORIGINAL_PATHS({}) == (Path('/etc/tllama/client.toml'),
                                      Path('/home/tester/.config/tllama/client.toml'))
        assert ORIGINAL_PATHS({'XDG_CONFIG_HOME': '/custom'})[1] == Path('/custom/tllama/client.toml')


ORIGINAL_PATHS = client_config.default_config_paths


@pytest.mark.parametrize('before', [True, False])
def test_config_flag_works_before_or_after_command(before):
    flags = ['--config', 'client.toml']
    args = build_parser().parse_args(flags + ['ps'] if before else ['ps'] + flags)
    assert args.config == 'client.toml'


def test_cli_uses_config_host_and_reports_config_errors_without_network(files, cli_server, capsys):
    files[0].write_text(f'[client]\nhost = "{cli_server["url"]}"\n')
    cli_server['responses']['/api/tags'] = (200, {'models': []})
    assert main(['list']) == 0
    assert capsys.readouterr().out.startswith('NAME')
    files[1].write_text('[client]\ntimeout = false')
    assert main(['list']) == 2
    output = capsys.readouterr()
    assert 'client.timeout' in output.err and 'Traceback' not in output.err and output.out == ''
    assert len(cli_server['calls']) == 1


def test_cli_session_limit_reaches_repl_storage(files, cli_server, monkeypatch, capsys):
    from test_client_run import Input
    from tllama.cli_repl import command
    monkeypatch.setattr('sys.stdin', Input(tty=True))
    files[1].write_text('[sessions]\nmax_file_mib=1\n')
    cli_server['responses']['/api/show'] = (200, {'capabilities': ['completion']})
    cli_server['responses']['/api/generate'] = (200, {'done': True, 'done_reason': 'load'})
    async def shell(client, session):
        assert session.session_max_file_mib == 1
        session.messages = [{'role': 'user', 'content': 'x' * (1024 * 1024)}]
        await command(client, session, '/save too-large')
        return 0
    monkeypatch.setattr('tllama.cli_repl.repl', shell)
    assert main(['run', 'model', '--host', cli_server['url']]) == 1
    output = capsys.readouterr()
    assert '1 MiB' in output.err and not output.out


@pytest.mark.parametrize('flag', ['--help', '--version'])
def test_help_and_version_do_not_read_config(files, flag):
    files[0].write_text('invalid TOML [')
    with pytest.raises(SystemExit) as caught:
        main(['--config', str(files[0]), flag])
    assert caught.value.code == 0
