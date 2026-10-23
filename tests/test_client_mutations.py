"""Client mutations, stream failures and ownership; no real downloads."""

import asyncio
import json

import httpx
import pytest

from tllama.cli import main
from tllama.client import TLlamaClient
from tllama.main import app
from test_client_cli import cli_server, isolated_proxy_environment
from test_native_manager import native_manager
from test_native_public_api import public


@pytest.mark.parametrize('as_json', [False, True])
def test_pull_progress_and_request(cli_server, capsys, as_json):
    events = [{'status': 'resolving repository'},
              {'status': 'downloading model', 'completed': 1500, 'total': 3000},
              {'status': 'success', 'path': '/models/file.gguf'}]
    cli_server['responses']['/api/pull'] = (200, ('\n'.join(json.dumps(e) for e in events) + '\n').encode())
    assert main(['pull', 'org/repo/file.gguf', '--host', cli_server['url']] + (['--json'] if as_json else [])) == 0
    out = capsys.readouterr()
    if as_json:
        assert [json.loads(line) for line in out.out.splitlines()] == events
        assert out.err == ''
    else:
        assert out.out == 'pulled: org/repo/file.gguf\n'
        assert '1.5 KB / 3.0 KB' in out.err and 'success' in out.err
    assert cli_server['calls'] == [('POST', '/api/pull', {'model': 'org/repo/file.gguf', 'stream': True})]


@pytest.mark.parametrize('status,payload,message', [
    (404, b'{"error":"file missing"}', 'file missing'),
    (200, b'{"status":"downloading model"}\n{"error":"download failed"}\n', 'download failed'),
    (200, b'{"status":"downloading model"}\n', 'without confirming success'),
    (200, b'{"status":"downloading model"}\nnot json\n', 'invalid NDJSON'),
    (200, b'[]\n', 'JSON object'),
    (200, b'{"completed":10}\n', 'missing its status'),
    (200, b'', 'without confirming success'),
])
def test_pull_failures_do_not_report_success(cli_server, capsys, status, payload, message):
    cli_server['responses']['/api/pull'] = (status, payload)
    assert main(['pull', 'a', '--host', cli_server['url']]) == 1
    out = capsys.readouterr()
    assert 'pulled:' not in out.out and message in out.err and 'Traceback' not in out.err


@pytest.mark.parametrize('as_json', [False, True])
def test_rm_continues_after_failure_and_reports_each_model(cli_server, capsys, as_json):
    def response(method, body):
        assert method == 'DELETE'
        if body['model'] == 'missing':
            return 404, {'error': 'not found'}
        return 200, {'status': 'success', 'deleted': body['model'] + '.toml', 'kept_model_file': True}
    cli_server['responses']['/api/delete'] = response
    assert main(['rm', 'first', 'missing', 'last', '--host', cli_server['url']] + (['--json'] if as_json else [])) == 1
    out = capsys.readouterr()
    if as_json:
        records = [json.loads(line) for line in out.out.splitlines()]
        assert [r['model'] for r in records] == ['first', 'missing', 'last']
        assert 'error' in records[1] and records[2]['response']['kept_model_file'] is True
    else:
        assert 'first (weights kept)' in out.out and 'last (weights kept)' in out.out
    assert 'missing' in out.err
    assert [call[2]['model'] for call in cli_server['calls']] == ['first', 'missing', 'last']


def test_rm_success(cli_server, capsys):
    cli_server['responses']['/api/delete'] = (200, {'status': 'success', 'kept_model_file': True})
    assert main(['rm', 'a', '--host', cli_server['url']]) == 0
    assert 'weights kept' in capsys.readouterr().out


@pytest.mark.parametrize('as_json', [False, True])
def test_stop_uses_empty_generate_unload(cli_server, capsys, as_json):
    reply = {'model': 'a', 'done': True, 'done_reason': 'unload', 'response': ''}
    cli_server['responses']['/api/generate'] = (200, reply)
    assert main(['stop', 'a', '--host', cli_server['url']] + (['--json'] if as_json else [])) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == reply if as_json else out.out == 'unload requested: a\n'
    assert cli_server['calls'] == [('POST', '/api/generate',
                                   {'model': 'a', 'prompt': '', 'keep_alive': 0, 'stream': False})]


@pytest.mark.parametrize('command,path', [('stop', '/api/generate'), ('rm', '/api/delete')])
def test_mutations_require_confirmation(cli_server, capsys, command, path):
    cli_server['responses'][path] = (200, {})
    assert main([command, 'a', '--host', cli_server['url']]) == 1
    assert 'did not confirm' in capsys.readouterr().err


class ProgressStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        # Split JSON across transport chunks; the NDJSON decoder must reassemble it.
        yield b'{"sta'
        yield b'tus":"downloading model"}\n'
        self.waiting.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


async def test_closing_pull_early_closes_owned_http_response():
    stream = ProgressStream()
    async with TLlamaClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))) as client:
        progress = client.pull('a')
        assert await anext(progress) == {'status': 'downloading model'}
        await progress.aclose()
        assert stream.closed


async def test_cancel_during_pull_closes_response_and_client_can_recover():
    stream = ProgressStream()
    def handler(request):
        return (httpx.Response(200, stream=stream) if request.url.path == '/api/pull'
                else httpx.Response(200, json={'version': 'recovered'}))
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        progress = client.pull('a')
        await anext(progress)
        task = asyncio.create_task(anext(progress))
        await stream.waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert await client.version() == 'recovered'


async def test_stop_retires_a_real_owned_worker_without_inference(public):
    manager, _, name, _ = public
    generation = await manager.preload_native_model(name, keep_alive=-1)
    async with TLlamaClient(host='http://public', transport=httpx.ASGITransport(app=app)) as client:
        assert (await client.stop(name))['done_reason'] == 'unload'
    await manager.wait_native_closed(generation)
    assert len(manager.test.ledger) == 1
    assert manager.test.ledger[0].process.returncode is not None
    assert manager.native_snapshot() == []


async def test_stop_needs_no_existing_definition_or_generation_capability(public, monkeypatch):
    manager, _, _, _ = public
    unloaded = []
    async def unload(name):
        unloaded.append(name)
    async def no_load(*args, **kwargs):
        raise AssertionError('Unload must not load a model')
    monkeypatch.setattr(manager, 'unload_native_model', unload)
    monkeypatch.setattr(manager, 'preload_native_model', no_load)
    monkeypatch.setattr(manager, 'acquire_native_model', no_load)
    async with TLlamaClient(host='http://public', transport=httpx.ASGITransport(app=app)) as client:
        await client.stop("missing-definition")
    assert unloaded == ["missing-definition"]
    assert manager.test.ledger == []
