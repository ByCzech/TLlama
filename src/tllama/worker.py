"""Async lifecycle and transport for one private native inference process."""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import re
import secrets
import signal
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Mapping, Any

import httpx

from tllama.worker_config import WorkerLaunchSpec


logger = logging.getLogger(__name__)
_ADDRESS = re.compile(r"^srv\s+\w+:\s+listening on http://127\.0\.0\.1:(\d+)\s*$")
_MAX_LINE = 64 * 1024
_MAX_RESPONSE = 16 * 1024 * 1024
_MAX_EVENT = 4 * 1024 * 1024


class WorkerError(RuntimeError):
    pass


class WorkerStartupError(WorkerError):
    pass


class WorkerStartupTimeout(WorkerStartupError):
    pass


class WorkerExited(WorkerError):
    def __init__(self, returncode: int, diagnostics: str):
        self.returncode = returncode
        super().__init__(f"Native worker exited with status {returncode}\n{diagnostics}")


class WorkerProtocolError(WorkerError):
    pass


class WorkerRequestError(WorkerError):
    def __init__(self, http_status: int, payload: Any):
        self.http_status = http_status
        self.payload = payload
        error = payload.get("error", payload) if isinstance(payload, dict) else payload
        code = error.get("code") if isinstance(error, dict) else None
        self.status_code = code if type(code) is int and 400 <= code <= 599 else (
            http_status if http_status >= 400 else 500
        )
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        super().__init__(f"Native request failed (HTTP {http_status}): {str(message)[:8192]}")


class NativeWorker:
    """Own a single worker and serialize its initial one-slot inference API."""

    def __init__(self, spec: WorkerLaunchSpec):
        self.spec = spec
        self.process: asyncio.subprocess.Process | None = None
        self.base_url: str | None = None
        self.props: dict | None = None
        self.runtime_directory: Path | None = None
        self.command: tuple[str, ...] | None = None
        self.load_log_complete = True
        self._state = "new"
        self._closing = False
        self._client: httpx.AsyncClient | None = None
        self._api_key = ""
        self._directory: tempfile.TemporaryDirectory | None = None
        self._spawn_task: asyncio.Task | None = None
        self._exit_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._drainers: list[asyncio.Task] = []
        self._address_seen = asyncio.Event()
        self._slot = asyncio.Lock()
        self._tail: deque[str] = deque(maxlen=128)
        self._load_lines: list[str] = []
        self._load_bytes = 0
        self._capture_load = True

    @property
    def state(self) -> str:
        if self._state == "ready" and self.process is not None and self.process.returncode is not None:
            return "failed"
        return self._state

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def diagnostics(self) -> str:
        return "\n".join(list(self._tail)[-16:])

    @property
    def load_log(self) -> str:
        return "".join(self._load_lines)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *args):
        await self.close()

    async def start(self):
        if self._state != "new" or self._closing:
            raise WorkerError("A worker can be started only once")
        self._state = "starting"
        try:
            await asyncio.wait_for(self._start_impl(), timeout=self.spec.load_timeout)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise WorkerStartupTimeout(
                f"Native worker did not become ready in {self.spec.load_timeout:g}s\n{self.diagnostics}"
            ) from exc
        except BaseException as exc:
            await self.close()
            if isinstance(exc, Exception):
                raise WorkerStartupError(f"Native worker startup failed: {exc}\n{self.diagnostics}") from exc
            raise
        return self

    async def _spawn(self, env: dict[str, str]):
        self.process = await asyncio.create_subprocess_exec(
            *self.command, cwd=self.runtime_directory, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._drainers = [asyncio.create_task(self._drain(pipe, name)) for pipe, name in
                          ((self.process.stdout, "stdout"), (self.process.stderr, "stderr"))]
        self._exit_task = asyncio.create_task(self.process.wait())

    async def _start_impl(self):
        if self._closing:
            raise WorkerError("Worker is closing")
        self._directory = tempfile.TemporaryDirectory(prefix="tllama-worker-")
        self.runtime_directory = Path(self._directory.name)
        key_file = self.runtime_directory / "api-key"
        self._api_key = secrets.token_urlsafe(32)
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(self._api_key + "\n")
        template_file = None
        if self.spec.template is not None:
            template_file = self.runtime_directory / "chat-template.jinja"
            fd = os.open(template_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as file:
                file.write(self.spec.template)
        self.command = self.spec.command(key_file, template_file)
        env = self.spec.child_environment(os.environ)
        # Shield creation so cancellation cannot abandon a child before its handle arrives.
        self._spawn_task = asyncio.create_task(self._spawn(env))
        await asyncio.shield(self._spawn_task)
        if self._closing:
            raise WorkerError("Worker is closing")
        if self.spec.port:
            self.base_url = f"http://127.0.0.1:{self.spec.port}"
        while self.base_url is None:
            self._check_alive()
            try:
                await asyncio.wait_for(self._address_seen.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers={"Authorization": f"Bearer {self._api_key}"},
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(self.spec.request_timeout, connect=5, write=30, pool=5),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )
        while True:
            self._check_alive()
            if self._closing:
                raise WorkerError("Worker is closing")
            try:
                response = await self._client.get("/health", timeout=1.0)
            except httpx.TransportError:
                await asyncio.sleep(0.1)
                continue
            if response.status_code == 503:
                await asyncio.sleep(0.1)
                continue
            if response.status_code != 200 or response.json() != {"status": "ok"}:
                raise WorkerProtocolError(f"Unexpected health response: HTTP {response.status_code}")
            response = await self._client.get("/props", timeout=1.0)
            if response.status_code != 200:
                raise WorkerRequestError(response.status_code, response.text)
            props = response.json()
            if not isinstance(props, dict):
                raise WorkerProtocolError("Native props must be an object")
            if props.get("build_info") != self.spec.runtime_fingerprint:
                raise WorkerProtocolError(f"Unexpected runtime fingerprint: {props.get('build_info')!r}")
            if (props.get("model_alias") != self.spec.alias
                    or type(props.get("total_slots")) is not int or props["total_slots"] != 1):
                raise WorkerProtocolError("Native alias/slot configuration differs from the launch")
            context = props.get("default_generation_settings", {}).get("n_ctx")
            if type(context) is not int or context <= 0:
                raise WorkerProtocolError("Native props lacks a valid effective slot context")
            self._check_alive()
            if self._closing:
                raise WorkerError("Worker is closing")
            self.props = props
            self._capture_load = False
            self._state = "ready"
            return

    def _check_alive(self):
        if self.process is not None and self.process.returncode is not None:
            raise WorkerExited(self.process.returncode, self.diagnostics)

    def _check_ready(self):
        self._check_alive()
        if self._state != "ready" or self._closing:
            raise WorkerError(f"Worker is not ready ({self.state})")

    def _record_line(self, line: str, channel: str):
        if self._api_key:
            line = line.replace(self._api_key, "[redacted]")
        self._tail.append(f"[{channel}] {line[-2048:].rstrip()}")
        logger.debug("worker %s %s: %s", self.pid, channel, line.rstrip())
        if self._capture_load:
            size = len(line.encode("utf-8"))
            if self._load_bytes + size <= self.spec.log_limit_bytes:
                self._load_lines.append(line)
                self._load_bytes += size
            else:
                self.load_log_complete = False
        if self.base_url is None:
            match = _ADDRESS.fullmatch(line.rstrip("\r\n"))
            if match and 1 <= int(match[1]) <= 65535:
                self.base_url = f"http://127.0.0.1:{int(match[1])}"
                self._address_seen.set()

    async def _drain(self, pipe: asyncio.StreamReader, channel: str):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        while True:
            data = await pipe.read(8192)
            pending += decoder.decode(data, final=not data)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self._record_line(line + "\n", channel)
            if len(pending) > _MAX_LINE:
                self._record_line(pending[-_MAX_LINE:] + "\n", channel)
                self.load_log_complete = False
                pending = ""
            if not data:
                if pending:
                    self._record_line(pending, channel)
                return

    async def wait_for_exit(self) -> int:
        if self._exit_task is None:
            raise WorkerError("Worker has not been spawned")
        return await asyncio.shield(self._exit_task)

    def _signal(self, sig: signal.Signals):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass

    async def close(self):
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_impl())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.done():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _close_impl(self):
        self._state = "closing"
        try:
            if self._spawn_task is not None:
                await asyncio.gather(self._spawn_task, return_exceptions=True)
            if self._client is not None:
                try:
                    await asyncio.wait_for(self._client.aclose(), self.spec.kill_timeout)
                except Exception:
                    logger.exception("Could not close native worker HTTP client")
            if self.process is not None:
                if self.process.returncode is None:
                    self._signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(self._exit_task), self.spec.terminate_timeout)
                except asyncio.TimeoutError:
                    self._signal(signal.SIGKILL)
                    await asyncio.wait_for(asyncio.shield(self._exit_task), self.spec.kill_timeout)
                # Descendants can keep inherited pipes open after the parent exits.
                self._signal(signal.SIGKILL)
        finally:
            for task in self._drainers:
                if not task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(task), self.spec.kill_timeout)
                    except asyncio.TimeoutError:
                        task.cancel()
            await asyncio.gather(*self._drainers, return_exceptions=True)
            if self._directory is not None:
                self._directory.cleanup()
            self._state = "closed" if self.process is None or self.process.returncode is not None else "failed"

    async def _json_body(self, response: httpx.Response) -> dict:
        body = bytearray()
        async for data in response.aiter_bytes():
            body.extend(data)
            if len(body) > _MAX_RESPONSE:
                raise WorkerProtocolError("Native response exceeded the JSON size limit")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            if response.status_code >= 400:
                raise WorkerRequestError(response.status_code, body[:8192].decode("utf-8", "replace")) from exc
            raise WorkerProtocolError("Native response is not valid JSON") from exc
        if response.status_code >= 400 or isinstance(payload, dict) and "error" in payload:
            raise WorkerRequestError(response.status_code, payload)
        if not isinstance(payload, dict):
            raise WorkerProtocolError("Native response must be an object")
        return payload

    @asynccontextmanager
    async def _response(self, path: str, body: dict):
        async with self._slot:
            self._check_ready()
            try:
                async with self._client.stream("POST", path, json=body) as response:
                    if response.status_code >= 400:
                        await self._json_body(response)
                    yield response
            except httpx.HTTPError as exc:
                self._check_alive()
                raise WorkerError(f"Native transport failed: {exc}") from exc

    def _body(self, payload: Mapping[str, Any], *, chat: bool, stream: bool) -> dict:
        body = dict(payload)
        body["stream"] = stream
        if chat:
            body["model"] = self.spec.alias
            if stream:
                options = body.get("stream_options", {})
                if not isinstance(options, dict):
                    raise ValueError("stream_options must be an object")
                body["stream_options"] = {**options, "include_usage": True}
        return body

    async def chat(self, payload: Mapping[str, Any]) -> dict:
        async with self._response("/v1/chat/completions", self._body(payload, chat=True, stream=False)) as response:
            return await self._json_body(response)

    async def completion(self, payload: Mapping[str, Any]) -> dict:
        async with self._response("/completion", self._body(payload, chat=False, stream=False)) as response:
            return await self._json_body(response)

    @asynccontextmanager
    async def chat_stream(self, payload: Mapping[str, Any]):
        async with self._response("/v1/chat/completions", self._body(payload, chat=True, stream=True)) as response:
            events = self._events(response, chat=True)
            try:
                yield events
            finally:
                await events.aclose()

    @asynccontextmanager
    async def completion_stream(self, payload: Mapping[str, Any]):
        async with self._response("/completion", self._body(payload, chat=False, stream=True)) as response:
            events = self._events(response, chat=False)
            try:
                yield events
            finally:
                await events.aclose()

    async def _events(self, response: httpx.Response, *, chat: bool) -> AsyncIterator[dict]:
        if response.headers.get("content-type", "").split(";")[0].strip() != "text/event-stream":
            payload = await self._json_body(response)
            raise WorkerProtocolError(f"Expected native SSE, received JSON: {str(payload)[:1024]}")
        data: list[str] = []
        size = 0
        terminal = False
        async for line in response.aiter_lines():
            if line == "":
                if not data:
                    continue
                text = "\n".join(data)
                data.clear()
                size = 0
                if text == "[DONE]":
                    if not chat:
                        raise WorkerProtocolError("Native completion stream used a chat terminal event")
                    terminal = True
                    break
                try:
                    payload = json.loads(text)
                except ValueError as exc:
                    raise WorkerProtocolError("Invalid JSON in native SSE") from exc
                if not isinstance(payload, dict):
                    raise WorkerProtocolError("Native SSE data must be an object")
                if "error" in payload:
                    raise WorkerRequestError(response.status_code, payload)
                if not chat and payload.get("stop") is True:
                    terminal = True
                yield payload
                if terminal:
                    break
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data.append(value)
                size += len(value.encode("utf-8")) + 1
                if size > _MAX_EVENT:
                    raise WorkerProtocolError("Native SSE event exceeded the size limit")
        if data or not terminal:
            raise WorkerProtocolError("Native stream ended without a complete terminal event")
