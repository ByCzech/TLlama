"""Public CLI behavior against an owned HTTP server, without a model runtime."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import httpx
import pytest

from tllama.client import ClientConfigError, ClientError, TLlamaClient, server_url
from tllama.cli import build_parser, main


@pytest.fixture(autouse=True)
def isolated_proxy_environment(monkeypatch):
    # Owned local TCP tests must not depend on the developer's proxy setup.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def cli_server():
    state = {'calls': [], 'responses': {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        def do_DELETE(self):
            self.reply()

        def reply(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            state['calls'].append((self.command, self.path, json.loads(body) if body else None))
            response = state['responses'].get(self.path, (404, {'error': 'missing route'}))
            if callable(response):
                response = response(self.command, json.loads(body) if body else None)
            status, value = response
            payload = json.dumps(value).encode() if not isinstance(value, bytes) else value
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{server.server_port}'
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize('raw,expected', [
    ('localhost', 'http://localhost:54800/'),
    ('host:1234', 'http://host:1234/'),
    (':1234', 'http://127.0.0.1:1234/'),
    ('0.0.0.0', 'http://127.0.0.1:54800/'),
    ('::', 'http://[::1]:54800/'),
    ('::1', 'http://[::1]:54800/'),
    ('[::1]:1234', 'http://[::1]:1234/'),
    ('https://example.test/prefix/', 'https://example.test:54800/prefix/'),
    ('http://[::]:1234', 'http://[::1]:1234/'),
    ('  localhost:1234  ', 'http://localhost:1234/'),
])
def test_server_addresses(raw, expected):
    assert server_url(raw) == expected


@pytest.mark.parametrize('raw', ['ftp://host', 'http://', 'http://host:0', 'host:65536',
    'host:abc', 'host:', 'http://user:secret@host', 'http://host/?q=a', 'http://host/#a',
    'bad host', '[::1', 'not:an:address'])
def test_invalid_addresses_are_clear_and_do_not_disclose_credentials(raw):
    with pytest.raises(ClientConfigError) as caught:
        server_url(raw)
    assert 'secret' not in str(caught.value)


def test_explicit_host_beats_environment_and_blank_environment_uses_default(monkeypatch):
    monkeypatch.setenv('TLLAMA_HOST', 'environment:1234')
    assert server_url() == 'http://environment:1234/'
    assert server_url('explicit:2345') == 'http://explicit:2345/'
    monkeypatch.setenv('TLLAMA_HOST', '')
    assert server_url() == 'http://127.0.0.1:54800/'


@pytest.mark.parametrize('command', [['list'], ['ls'], ['show', 'a'], ['ps'], ['version']])
@pytest.mark.parametrize('before', [False, True])
def test_client_flags_work_before_and_after_subcommand(command, before):
    flags = ['--host', 'test:1234', '--timeout', '12', '--json']
    args = build_parser().parse_args(flags + command if before else command + flags)
    assert (args.host, args.timeout, args.json) == ('test:1234', 12, True)


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'no'])
def test_invalid_timeout_is_a_usage_error(value, capsys):
    with pytest.raises(SystemExit) as caught:
        main(['list', '--timeout', value])
    assert caught.value.code == 2
    assert 'timeout' in capsys.readouterr().err


@pytest.mark.parametrize('command,path', [('list', '/api/tags'), ('ls', '/api/tags'), ('ps', '/api/ps')])
def test_empty_list_and_ps(cli_server, command, path, capsys):
    cli_server['responses'][path] = (200, {'models': []})
    assert main([command, '--host', cli_server['url']]) == 0
    out = capsys.readouterr()
    assert out.out.startswith('NAME') and out.err == ''
    assert cli_server['calls'] == [('GET', path, None)]


def test_list_json_preserves_server_fields_and_uses_prefix(cli_server, capsys, monkeypatch):
    data = {'models': [{'name': 'a', 'size': 1200, 'details': {'runner': 'llamacpp'}}], 'extension': True}
    cli_server['responses']['/proxy/api/tags'] = (200, data)
    monkeypatch.setenv('TLLAMA_HOST', 'invalid:0')
    assert main(['--host', cli_server['url'] + '/proxy', 'ls', '--json']) == 0
    assert json.loads(capsys.readouterr().out) == data
    assert cli_server['calls'] == [('GET', '/proxy/api/tags', None)]


@pytest.mark.parametrize('device,label', [(None, 'unknown'), (0, '100% CPU'), (1000, '100% GPU'),
                                          (500, '50.0% CPU / 50.0% GPU')])
def test_ps_uses_server_sizes_without_guessing_unknown_placement(cli_server, capsys, device, label):
    cli_server['responses']['/api/ps'] = (200, {'models': [{'name': 'a', 'size': 1000, 'size_vram': device,
        'runner': 'llamacpp', 'context_length': 2048, 'expires_at': None}]})
    assert main(['ps', '--host', cli_server['url']]) == 0
    out = capsys.readouterr().out
    assert label in out and '2048' in out and 'llamacpp' in out and 'pinned' in out


@pytest.mark.parametrize('field', ['modelfile', 'parameters', 'template', 'system', 'license'])
def test_show_text_is_verbatim(cli_server, field, capsys):
    text = '# Český komentář\n[llm]\nmodel="Local/a.gguf"\n'
    cli_server['responses']['/api/show'] = (200, {field: text})
    assert main(['show', 'a', '--' + field, '--host', cli_server['url']]) == 0
    assert capsys.readouterr().out == text
    assert cli_server['calls'] == [('POST', '/api/show', {'model': 'a'})]


def test_show_json_and_human_summary(cli_server, capsys):
    data = {'details': {'family': 'qwen3', 'parameter_size': '35B', 'quantization_level': 'IQ3_S'},
            'capabilities': ['completion', 'thinking'], 'system': 'Be precise.'}
    cli_server['responses']['/api/show'] = (200, data)
    assert main(['show', 'a', '--host', cli_server['url']]) == 0
    out = capsys.readouterr().out
    assert all(text in out for text in ['qwen3', '35B', 'IQ3_S', 'thinking', 'Be precise.'])
    assert main(['show', 'a', '--json', '--host', cli_server['url']]) == 0
    assert json.loads(capsys.readouterr().out) == data


def test_show_json_and_text_selector_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as caught:
        main(['--json', 'show', 'a', '--modelfile'])
    assert caught.value.code == 2


def test_version_reports_both_but_version_flag_is_offline(cli_server, capsys, monkeypatch):
    cli_server['responses']['/api/version'] = (200, {'version': '1.2.3'})
    assert main(['version', '--json', '--host', cli_server['url']]) == 0
    assert json.loads(capsys.readouterr().out)['server_version'] == '1.2.3'
    monkeypatch.setenv('TLLAMA_HOST', 'invalid:0')
    with pytest.raises(SystemExit) as caught:
        main(['--version'])
    assert caught.value.code == 0
    assert len(cli_server['calls']) == 1


@pytest.mark.parametrize('status,body,expected', [
    (404, {'error': 'model missing'}, 'HTTP 404: model missing'),
    (500, b'<html>broken</html>', 'HTTP 500'),
    (400, {'error': {'message': 'invalid input'}}, 'invalid input'),
    (200, b'not json', 'invalid JSON'),
    (200, [], 'JSON object'),
    (200, {'models': None}, 'models array'),
    (200, {'models': [1]}, 'models array'),
])
def test_server_failures_are_nonzero_stderr_without_traceback(cli_server, capsys, status, body, expected):
    cli_server['responses']['/api/tags'] = (status, body)
    assert main(['list', '--host', cli_server['url']]) == 1
    out = capsys.readouterr()
    assert out.out == '' and expected in out.err and 'Traceback' not in out.err


@pytest.mark.parametrize('exception', [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
async def test_transport_errors_are_client_errors(exception):
    def handler(request):
        raise exception('test failure', request=request)
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError):
            await client.list_models()


def test_network_client_imports_no_server_or_local_runtime(cli_server, tmp_path):
    cli_server['responses']['/api/tags'] = (200, {'models': []})
    code = '''
import importlib.abc
import sys
class RefuseServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn', 'gguf', 'llama_cpp'} or fullname in {
            'tllama.backend', 'tllama.main', 'tllama.worker', 'tllama.config'}:
            raise AssertionError('Server dependency imported: ' + fullname)
sys.meta_path.insert(0, RefuseServer())
from tllama.cli import main
raise SystemExit(main(['list', '--json', '--host', sys.argv[1]]))
'''
    result = subprocess.run([sys.executable, '-c', code, cli_server['url']],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
             'TLLAMA_MODELS': str(tmp_path / 'absent'), 'TLLAMA_MAX_LOADED_MODELS': 'invalid'},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'models': []}
    assert not (tmp_path / 'absent').exists()


def test_invalid_http_environment_is_a_configuration_error(monkeypatch, capsys):
    monkeypatch.setenv('ALL_PROXY', 'unsupported://127.0.0.1:1234')
    assert main(['list']) == 2
    assert 'Cannot initialize HTTP client' in capsys.readouterr().err
