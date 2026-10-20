"""Shared async HTTP operations for the CLI and future interactive client."""

from __future__ import annotations

from ipaddress import IPv6Address
import json
import math
import os
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_HOST = 'http://127.0.0.1:54800'
DEFAULT_TIMEOUT = 30.0


class ClientError(Exception):
    """A server, connection or response failure suitable for a CLI message."""


class ClientConfigError(ValueError):
    """Invalid client configuration, independent of daemon configuration."""


def server_url(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get('TLLAMA_HOST', '')
    raw = raw.strip() or DEFAULT_HOST
    try:
        if any(c.isspace() for c in raw):
            raise ValueError
        if '://' not in raw:
            if raw.startswith(':') and raw.count(':') == 1:
                raw = '127.0.0.1' + raw
            if raw.count(':') > 1 and not raw.startswith('['):
                IPv6Address(raw)
                raw = '[' + raw + ']'
            raw = 'http://' + raw
        parts = urlsplit(raw)
        if (parts.scheme not in {'http', 'https'} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment):
            raise ValueError
        port = parts.port
        # An explicitly empty port is a typo, not the default.
        if parts.netloc.endswith(':'):
            raise ValueError
        port = 54800 if port is None else port
        if not 1 <= port <= 65535:
            raise ValueError
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(parts.hostname, parts.hostname)
        if ':' in host:
            host = '[' + host + ']'
        return urlunsplit((parts.scheme, f'{host}:{port}', parts.path.rstrip('/') + '/', '', ''))
    except ValueError:
        raise ClientConfigError('Invalid server address: use HOST[:PORT] or an HTTP(S) URL without credentials, query or fragment.') from None


class TLlamaClient:
    def __init__(self, host: str | None = None, *, timeout: float = DEFAULT_TIMEOUT, transport=None):
        self.url = server_url(host)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ClientConfigError('Client timeout must be a finite number greater than zero.')
        self.timeout = timeout
        try:
            self._http = httpx.AsyncClient(base_url=self.url, transport=transport,
                                           timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)))
        except (httpx.InvalidURL, ImportError, ValueError, OSError) as exc:
            raise ClientConfigError(f'Cannot initialize HTTP client: {exc}') from None

    async def __aenter__(self):
        await self._http.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._http.__aexit__(*args)

    def _transport_error(self, exc: httpx.HTTPError) -> ClientError:
        if isinstance(exc, httpx.TimeoutException):
            return ClientError(f'Request to {self.url} timed out; adjust --timeout if needed.')
        if isinstance(exc, httpx.ConnectError):
            return ClientError(f'Cannot connect to TLlama at {self.url}; start tllamad or select --host.')
        return ClientError(f'HTTP communication with {self.url} failed: {exc}')

    @staticmethod
    def _check_status(response):
        if response.is_success:
            return
        detail = None
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get('error') or body.get('detail')
                if isinstance(detail, dict):
                    detail = detail.get('message')
        except ValueError:
            pass
        message = str(detail)[:2000] if detail else response.reason_phrase
        raise ClientError(f'HTTP {response.status_code}: {message}')

    @staticmethod
    def _decode(response):
        try:
            data = response.json()
        except ValueError:
            raise ClientError('Server returned invalid JSON.') from None
        if not isinstance(data, dict):
            raise ClientError('Server response must be a JSON object.')
        if data.get('error'):
            raise ClientError(str(data['error']))
        return data

    async def _request(self, method, path, **kwargs):
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from None
        self._check_status(response)
        return self._decode(response)

    @staticmethod
    def _models(data):
        models = data.get('models')
        if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
            raise ClientError('Server response must contain a models array of objects.')
        return models

    async def list_models(self):
        data = await self._request('GET', 'api/tags')
        self._models(data)
        return data

    async def show(self, model):
        return await self._request('POST', 'api/show', json={'model': model})

    async def ps(self):
        data = await self._request('GET', 'api/ps')
        self._models(data)
        return data

    async def version(self):
        data = await self._request('GET', 'api/version')
        if not isinstance(data.get('version'), str) or not data['version']:
            raise ClientError('Server response is missing its version string.')
        return data['version']

    async def remove(self, model):
        data = await self._request('DELETE', 'api/delete', json={'model': model})
        if data.get('status') != 'success':
            raise ClientError('Server did not confirm model definition deletion.')
        return data

    async def stop(self, model):
        # Empty generate unloads before model-mode admission, including
        # embedding and Decision definitions. It must never load a worker.
        data = await self._request('POST', 'api/generate', json={
            'model': model, 'prompt': '', 'keep_alive': 0, 'stream': False})
        if data.get('done') is not True or data.get('done_reason') != 'unload':
            raise ClientError('Server did not confirm the unload request.')
        return data

    async def pull(self, model):
        """Yield validated progress; callers must consume or close the generator."""
        try:
            async with self._http.stream('POST', 'api/pull', json={'model': model, 'stream': True}) as response:
                if not response.is_success:
                    await response.aread()
                    self._check_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        raise ClientError('Pull returned invalid NDJSON.') from None
                    if not isinstance(event, dict):
                        raise ClientError('Pull progress must be a JSON object.')
                    if event.get('error'):
                        raise ClientError(str(event['error']))
                    if not isinstance(event.get('status'), str) or not event['status']:
                        raise ClientError('Pull progress is missing its status.')
                    yield event
                    if event['status'] == 'success':
                        return
                raise ClientError('Pull stream ended without confirming success.')
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from None
