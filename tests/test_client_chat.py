"""Chat protocol and ownership shared by one-shot and interactive clients."""

import asyncio
from contextlib import aclosing
import gzip
import json
import sys

import httpx
import pytest

from tllama.client import ClientError, TLlamaClient
from test_pull_stream_ownership import WireBody


def event(content='', *, done=False, **extra):
    return {'message': {'role': 'assistant', 'content': content}, 'done': done, **extra}


def ndjson(events):
    return ('\n' + '\n'.join(json.dumps(e, ensure_ascii=False) for e in events) + '\n').encode()


@pytest.mark.parametrize('controls', [{}, {
    'options': {'temperature': 0, 'num_predict': 0}, 'think': False,
    'format': {'type': 'object'}, 'keep_alive': 0,
}])
async def test_chat_preserves_messages_controls_and_original_events(controls):
    messages = [{'role': 'system', 'content': 'Be brief.'},
                {'role': 'user', 'content': '  /literal "český text"\n'}]
    events = [event(), {'message': {'role': 'assistant', 'thinking': 'úvaha'}, 'done': False},
              event('odpověď'), event(done=True, done_reason='length', eval_count=12)]
    events[-1]['message']['tool_calls'] = [{'function': {'name': 'test', 'arguments': {'x': 1}}}]
    def handler(request):
        assert request.url.path == '/prefix/api/chat'
        assert json.loads(request.content) == {
            'model': 'model', 'messages': messages, 'stream': True, **controls}
        assert request.extensions['timeout']['read'] is None
        return httpx.Response(200, content=ndjson(events))
    async with TLlamaClient('http://localhost/prefix', transport=httpx.MockTransport(handler)) as client:
        async with aclosing(client.chat('model', messages, **controls)) as stream:
            assert [e async for e in stream] == events


@pytest.mark.parametrize('payload,phrase', [
    (b'not JSON\n', 'invalid NDJSON'), (b'[]\n', 'JSON object'),
    (b'{"error":"model failed"}\n', 'model failed'),
    (ndjson([{'message': {'role': 'assistant'}, 'done': 1}]), 'boolean done'),
    (ndjson([{'done': False}]), 'assistant message'),
    (ndjson([{'done': False, 'message': {'role': 'user', 'content': 'a'}}]), 'assistant message'),
    (ndjson([{'done': False, 'message': {'role': 'assistant', 'content': None}}]), 'content must'),
    (ndjson([{'done': False, 'message': {'role': 'assistant', 'thinking': []}}]), 'thinking must'),
    (ndjson([event('unfinished')]), 'without confirming completion'),
    (b'', 'without confirming completion'),
])
async def test_chat_rejects_invalid_or_incomplete_stream(payload, phrase):
    async with TLlamaClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload))) as client:
        async with aclosing(client.chat('model', [])) as stream:
            with pytest.raises(ClientError, match=phrase):
                async for _ in stream:
                    pass


@pytest.mark.parametrize('outcome', ['done', 'early_close', 'cancel', 'error'])
@pytest.mark.parametrize('encoding', ['identity', 'gzip'])
async def test_chat_unwinds_real_httpx_layers(outcome, encoding):
    events = [event('česká odpověď')] * (4 if outcome != 'cancel' else 1)
    if outcome == 'done':
        events += [event(done=True)]
    elif outcome == 'error':
        events += [{'error': 'worker failed'}]
    payload = ndjson(events)
    if encoding == 'gzip':
        payload = gzip.compress(payload)
    body = WireBody(payload, keep_open=True)
    response = httpx.Response(200, stream=body, headers={'content-encoding': encoding})
    held = []
    hooks = sys.get_asyncgen_hooks()
    def firstiter(gen):
        held.append(gen)
        if hooks.firstiter:
            hooks.firstiter(gen)
    sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=hooks.finalizer)
    try:
        async with TLlamaClient(transport=httpx.MockTransport(lambda request: response)) as client:
            async with aclosing(client.chat('model', [])) as stream:
                if outcome == 'early_close':
                    await anext(stream)
                elif outcome == 'cancel':
                    await anext(stream)
                    pending = asyncio.create_task(anext(stream))
                    await body.waiting.wait()
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    async def consume():
                        return [e async for e in stream]
                    if outcome == 'error':
                        with pytest.raises(ClientError, match='worker failed'):
                            await asyncio.wait_for(consume(), 1)
                    else:
                        assert (await asyncio.wait_for(consume(), 1))[-1]['done']
            assert response.is_closed and body.closed and body.iterator_closed
            assert [g.ag_code.co_qualname for g in held if g.ag_frame is not None] == []
    finally:
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)
        for gen in reversed(held):
            if not gen.ag_running:
                await gen.aclose()


@pytest.mark.parametrize('exception', [httpx.ConnectError, httpx.ReadTimeout])
async def test_chat_maps_transport_errors_and_honors_explicit_deadline(exception):
    def handler(request):
        assert request.extensions['timeout'] == {'connect': 2.5, 'read': 2.5, 'write': 2.5, 'pool': 2.5}
        raise exception('interrupted', request=request)
    async with TLlamaClient(timeout=2.5, transport=httpx.MockTransport(handler)) as client:
        async with aclosing(client.chat('model', [])) as stream:
            with pytest.raises(ClientError):
                await anext(stream)


async def test_chat_preserves_http_error_detail():
    async with TLlamaClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={'error': 'model not found'}))) as client:
        async with aclosing(client.chat('missing', [])) as stream:
            with pytest.raises(ClientError, match='HTTP 404: model not found'):
                await anext(stream)
