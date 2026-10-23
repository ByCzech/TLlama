"""The pull iterator is closed before the HTTP response and loop teardown."""

import asyncio
import json

import httpx
import pytest

from tllama.client import ClientError, TLlamaClient


class TrackedResponse(httpx.Response):
    def __init__(self, line):
        super().__init__(200, stream=httpx.ByteStream(b''))
        self.line = line
        self.closed_order = []
        self.waiting = asyncio.Event()
        self.lines = None

    def aiter_lines(self):
        async def lines():
            try:
                yield self.line
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.closed_order.append('lines')
        # Retain a strong reference: garbage collection must not make a
        # missing explicit close look correct before the assertion.
        self.lines = lines()
        return self.lines

    async def aclose(self):
        if not self.is_closed:
            self.closed_order.append('response')
        await super().aclose()


@pytest.mark.parametrize('outcome,line', [
    ('success', json.dumps({'status': 'success'})),
    ('early_close', json.dumps({'status': 'downloading model'})),
    ('error', json.dumps({'error': 'download failed'})),
    ('invalid_json', 'not JSON'),
    ('cancel', json.dumps({'status': 'downloading model'})),
])
async def test_pull_closes_owned_lines_before_returning(outcome, line):
    response = TrackedResponse(line)
    async with TLlamaClient(transport=httpx.MockTransport(lambda request: response)) as client:
        progress = client.pull('model')
        try:
            if outcome == 'success':
                assert [event async for event in progress] == [{'status': 'success'}]
            elif outcome == 'early_close':
                await anext(progress)
                await progress.aclose()
            elif outcome == 'cancel':
                await anext(progress)
                task = asyncio.create_task(anext(progress))
                await response.waiting.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ClientError):
                    await anext(progress)
            assert response.closed_order == ['lines', 'response']
            assert response.lines.ag_frame is None
        finally:
            await progress.aclose()
            # Keep the deliberately failing pre-fix run clean as well.
            await response.lines.aclose()
