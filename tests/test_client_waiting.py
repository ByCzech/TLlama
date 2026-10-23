"""Long server work is distinct from a failed connection or an explicit deadline."""

import asyncio
import json
import logging
import time

import httpx
import pytest

from tllama import cli
from tllama.backend import ModelManager
from tllama.client import ClientError, TLlamaClient
from test_client_cli import cli_server, isolated_proxy_environment


@pytest.mark.parametrize('operation,long_work', [
    ('list_models', True), ('show', True), ('pull', True),
    ('ps', False), ('version', False), ('remove', False), ('stop', False),
])
@pytest.mark.parametrize('override', [None, 2.5, 60.0])
async def test_read_policy_preserves_connect_and_explicit_timeouts(operation, long_work, override):
    calls = []
    def handler(request):
        calls.append(request.extensions['timeout'])
        if operation == 'pull':
            return httpx.Response(200, content=b'{"status":"success"}\n')
        return httpx.Response(200, json={'models': [], 'version': '1', 'status': 'success',
                                        'done': True, 'done_reason': 'unload'})
    async with TLlamaClient(timeout=override, transport=httpx.MockTransport(handler)) as client:
        if operation == 'pull':
            assert [event async for event in client.pull('a')] == [{'status': 'success'}]
        elif operation in {'show', 'remove', 'stop'}:
            await getattr(client, operation)('a')
        else:
            await getattr(client, operation)()
    limit = 30.0 if override is None else override
    assert calls == [{'connect': min(limit, 10.0), 'read': None if long_work and override is None else limit,
                      'write': limit, 'pool': limit}]


@pytest.mark.parametrize('operation,path,reply', [
    ('list', '/api/tags', {'models': []}),
    ('show', '/api/show', {'details': {}, 'capabilities': []}),
])
def test_slow_model_information_shows_notice_and_completes(cli_server, capsys, monkeypatch, operation, path, reply):
    monkeypatch.setattr(cli, 'WAIT_NOTICE_AFTER', 0.01)
    def response(*args):
        time.sleep(0.06)
        return 200, reply
    cli_server['responses'][path] = response
    args = [operation] + (['a'] if operation == 'show' else [])
    assert cli.main(args + ['--host', cli_server['url']]) == 0
    out = capsys.readouterr()
    assert 'Waiting for TLlama' in out.err and 'metadata indexing may take time' in out.err
    assert len(cli_server['calls']) == 1
    assert out.out and 'timed out' not in out.err


def test_slow_json_list_keeps_stdout_machine_readable(cli_server, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'WAIT_NOTICE_AFTER', 0.01)
    def response(*args):
        time.sleep(0.06)
        return 200, {'models': []}
    cli_server['responses']['/api/tags'] = response
    assert cli.main(['list', '--json', '--host', cli_server['url']]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == {'models': []} and out.err == ''


def test_explicit_deadline_reports_response_timeout_after_connection(cli_server, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'WAIT_NOTICE_AFTER', 0.005)
    def response(*args):
        time.sleep(0.1)
        return 200, {'models': []}
    cli_server['responses']['/api/tags'] = response
    assert cli.main(['list', '--timeout', '0.025', '--host', cli_server['url']]) == 1
    out = capsys.readouterr()
    assert len(cli_server['calls']) == 1
    assert 'waiting for an HTTP response' in out.err
    assert 'server may still be processing' in out.err
    assert 'Cannot connect' not in out.err and 'Traceback' not in out.err


@pytest.mark.parametrize('exception,phrase', [
    (httpx.ConnectTimeout, 'establishing a connection'),
    (httpx.ReadTimeout, 'waiting for an HTTP response'),
    (httpx.WriteTimeout, 'sending the HTTP request'),
    (httpx.PoolTimeout, 'available client connection'),
])
async def test_timeout_messages_identify_the_phase(exception, phrase):
    def handler(request):
        raise exception('test deadline', request=request)
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match=phrase):
            await client.list_models()


async def test_cancel_wait_notice_drains_the_owned_request(monkeypatch):
    monkeypatch.setattr(cli, 'WAIT_NOTICE_AFTER', 60)
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def operation():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    task = asyncio.create_task(cli._wait_for_model_info(operation(), notice=True))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.parametrize('restart', [False, True])
async def test_only_cold_cache_work_is_logged(manager, caplog, restart):
    import gguf
    path = manager.local_models_dir / 'a.gguf'
    writer = gguf.GGUFWriter(str(path), arch='llama')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    (manager.local_models_dir / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n')
    with caplog.at_level(logging.INFO, logger='tllama.backend'):
        assert len(await manager.list_local_models_with_metadata()) == 1
        messages = caplog.messages
        assert any('Reading GGUF metadata:' in m for m in messages)
        assert any('GGUF metadata ready:' in m for m in messages)
        assert any('Computing model SHA-256:' in m for m in messages)
        assert any('Model SHA-256 ready:' in m for m in messages)
        caplog.clear()
        if restart:
            manager = ModelManager(manager.config)
        assert len(await manager.list_local_models_with_metadata()) == 1
        assert not caplog.records
