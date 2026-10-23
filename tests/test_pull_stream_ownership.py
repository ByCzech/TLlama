"""Retain real HTTPX generators to detect leaks before GC or loop shutdown."""

import asyncio
from contextlib import aclosing
import gzip
import sys

import httpx
import pytest

from tllama.client import ClientError, TLlamaClient


class WireBody(httpx.AsyncByteStream):
    def __init__(self, payload, *, keep_open):
        self.payload = payload
        self.keep_open = keep_open
        self.waiting = asyncio.Event()
        self.closed = False
        self.iterator_closed = False

    async def __aiter__(self):
        try:
            for offset in range(0, len(self.payload), 7):
                yield self.payload[offset:offset + 7]
            if self.keep_open:
                self.waiting.set()
                await asyncio.Event().wait()
        finally:
            self.iterator_closed = True

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize('encoding', ['identity', 'gzip'])
@pytest.mark.parametrize('buffered,outcome', [
    (buffered, outcome) for buffered in (False, True)
    for outcome in ('success', 'early_close', 'error', 'invalid_json', 'cancel')
    if not (buffered and outcome == 'cancel')
])
async def test_all_httpx_layers_are_closed_before_returning(outcome, encoding, buffered):
    first = '{"status":"příprava"}\n'
    if outcome == 'error':
        text = first + '{"error":"download failed"}\n'
    elif outcome == 'invalid_json':
        text = first + 'not JSON\n'
    elif outcome in {'early_close', 'success'}:
        # More than one line also exercises cancellation while the bounded
        # producer handoff is full, rather than during a network read.
        text = first * 4 + '{"status":"success"}\n'
    else:
        text = first
    raw = text.encode()
    payload = gzip.compress(raw) if encoding == 'gzip' else raw
    headers = {'content-encoding': encoding, 'content-type': 'application/x-ndjson; charset=utf-8'}
    body = WireBody(payload, keep_open=True)
    response = (httpx.Response(200, content=payload, headers=headers) if buffered else
                httpx.Response(200, stream=body, headers=headers))
    held = []
    hooks = sys.get_asyncgen_hooks()
    def firstiter(gen):
        held.append(gen)
        if hooks.firstiter:
            hooks.firstiter(gen)
    sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=hooks.finalizer)
    try:
        async with TLlamaClient(transport=httpx.MockTransport(lambda request: response)) as client:
            async with aclosing(client.pull('model')) as progress:
                if outcome == 'early_close':
                    assert (await anext(progress))['status'] == 'příprava'
                elif outcome == 'cancel':
                    await anext(progress)
                    pending = asyncio.create_task(anext(progress))
                    await body.waiting.wait()
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                elif outcome in {'error', 'invalid_json'}:
                    with pytest.raises(ClientError):
                        async for _ in progress:
                            pass
                else:
                    async def consume():
                        return [event async for event in progress]
                    events = await asyncio.wait_for(consume(), timeout=1.0)
                    assert events[-1] == {'status': 'success'}
            # Strong references prevent async-generator finalizers from
            # accidentally hiding a leak by running at a convenient time.
            leaking = [gen.ag_code.co_qualname for gen in held if gen.ag_frame is not None]
            assert leaking == []
            if not buffered:
                assert body.closed and body.iterator_closed
            assert response.is_closed
    finally:
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)
        # Also keep the intentionally failing pre-fix run from polluting
        # later tests with delayed finalizers.
        for gen in reversed(held):
            if not gen.ag_running:
                await gen.aclose()


@pytest.mark.parametrize('exception,expected', [
    (httpx.ConnectError, ClientError), (httpx.ReadTimeout, ClientError),
    (asyncio.CancelledError, asyncio.CancelledError),
])
async def test_reader_failure_is_forwarded_without_hanging(exception, expected):
    async def handler(request):
        raise exception('transport interrupted')
    async with TLlamaClient(transport=httpx.MockTransport(handler)) as client:
        async with aclosing(client.pull('model')) as progress:
            with pytest.raises(expected):
                await asyncio.wait_for(anext(progress), timeout=1.0)
