"""Async lifecycle and transport for one private native inference process."""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import re
import signal
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Mapping, Any

import httpx

from tllama.worker_config import WorkerLaunchSpec
from tllama.runtime_fingerprint import evaluate_runtime_fingerprint, RuntimeFingerprintMismatch
from tllama.worker_ipc import WorkerCredentials, runtime_directory
from tllama.native_memory import NativeMemory
from tllama.native_memory_api import MemoryReportError, NativeMemoryReport, UnsupportedMemorySchema
from tllama.native_telemetry import MemoryTelemetry
from tllama.decision_contract import prepare_decision_request, validate_decision_response


logger = logging.getLogger(__name__)
_ADDRESS = re.compile(r"^srv\s+\w+:\s+listening on http://127\.0\.0\.1:(\d+)\s*$")
_MAX_LINE = 64 * 1024
_MAX_RESPONSE = 16 * 1024 * 1024
_MAX_EVENT = 4 * 1024 * 1024
_MEMORY_TIMEOUT = 1.0
_MEMORY_INTERVAL = 1.0


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
        self.socket_path: Path | None = None
        self.props: dict | None = None
        self.runtime_directory: Path | None = None
        self.command: tuple[str, ...] | None = None
        self.load_log_complete = True
        self._state = "new"
        self._closing = False
        self._client: httpx.AsyncClient | None = None
        self._api_key = ""
        self._credentials: WorkerCredentials | None = None
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
        self.runtime_fingerprint: str | None = None
        self.memory = NativeMemory(None, projector=spec.mmproj is not None)
        self.telemetry = MemoryTelemetry(self.memory, mode=spec.memory_source, compare=spec.memory_compare)
        self._memory_client: httpx.AsyncClient | None = None
        self._memory_task: asyncio.Task | None = None
        self._memory_lock = asyncio.Lock()
        self._memory_failure = None
        self._memory_parser_failed = False

    @property
    def state(self) -> str:
        if self._state == "ready" and self.process is not None and self.process.returncode is not None:
            return "failed"
        return self._state

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def address(self) -> str | None:
        return f"unix://{self.socket_path}" if self.socket_path is not None else self.base_url

    def _http_client(self, *, timeout, limits):
        # Each client owns its own pool, including the independent telemetry
        # pool. A dummy HTTP origin supplies Host/relative URL handling only;
        # the UDS transport never resolves it or connects to a TCP address.
        kwargs = {}
        if self.socket_path is not None:
            kwargs['transport'] = httpx.AsyncHTTPTransport(
                uds=str(self.socket_path), trust_env=False, limits=limits)
        return httpx.AsyncClient(
            base_url=self.base_url, headers={"Authorization": f"Bearer {self._api_key}"},
            trust_env=False, follow_redirects=False, timeout=timeout, limits=limits, **kwargs)

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
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command, cwd=self.runtime_directory, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True, pass_fds=self._credentials.pass_fds,
            )
        finally:
            self._credentials.close_fd()
        self._drainers = [asyncio.create_task(self._drain(pipe, name)) for pipe, name in
                          ((self.process.stdout, "stdout"), (self.process.stderr, "stderr"))]
        self._exit_task = asyncio.create_task(self.process.wait())

    async def _start_impl(self):
        if self._closing:
            raise WorkerError("Worker is closing")
        self._directory = runtime_directory()
        self.runtime_directory = Path(self._directory.name)
        if self.spec.transport == "uds":
            self.socket_path = self.runtime_directory / "worker.sock"
            # macOS/BSD sun_path is shorter than Linux's. Refuse a long path
            # explicitly instead of failing opaquely after spawning the child.
            if len(os.fsencode(self.socket_path)) > 103:
                raise WorkerError("Worker UDS path is too long; select a shorter TMPDIR or runtime directory")
            self.base_url = "http://tllama-worker"
        self._credentials = WorkerCredentials(self.runtime_directory)
        self._api_key = self._credentials.token
        template_file = None
        if self.spec.template is not None:
            template_file = self.runtime_directory / "chat-template.jinja"
            fd = os.open(template_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as file:
                file.write(self.spec.template)
        self.command = self.spec.command(self._credentials.path, template_file, self.socket_path)
        env = self.spec.child_environment(os.environ)
        # Shield creation so cancellation cannot abandon a child before its handle arrives.
        self._spawn_task = asyncio.create_task(self._spawn(env))
        await asyncio.shield(self._spawn_task)
        if self._closing:
            raise WorkerError("Worker is closing")
        if self.spec.transport == "tcp" and self.spec.port:
            self.base_url = f"http://127.0.0.1:{self.spec.port}"
        while self.base_url is None:
            self._check_alive()
            try:
                await asyncio.wait_for(self._address_seen.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
        self._client = self._http_client(
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
            observed = props.get("build_info")
            self.runtime_fingerprint = observed if isinstance(observed, str) and observed else None
            self.memory.identify_runtime(self.runtime_fingerprint)
            try:
                evaluate_runtime_fingerprint(self.runtime_fingerprint,
                    expected_fingerprints=self.spec.accepted_runtime_fingerprints,
                    policy=self.spec.version_policy, context=self.spec.mode)
            except RuntimeFingerprintMismatch as exc:
                raise WorkerProtocolError(str(exc)) from exc
            if (props.get("model_alias") != self.spec.alias
                    or type(props.get("total_slots")) is not int or props["total_slots"] != 1):
                raise WorkerProtocolError("Native alias/slot configuration differs from the launch")
            context = props.get("default_generation_settings", {}).get("n_ctx")
            if type(context) is not int or context <= 0:
                raise WorkerProtocolError("Native props lacks a valid effective slot context")
            if self.spec.decision_type is not None:
                async with self._client.stream("GET", "/v1/models") as response:
                    models = await self._json_body(response)
                data = models.get("data")
                if (not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict)
                        or data[0].get("id") != self.spec.alias
                        or not isinstance(data[0].get("architecture"), dict)
                        or data[0]["architecture"].get("output_modalities") != ["decisions"]):
                    raise WorkerProtocolError("Native model does not advertise the loaded Decision alias/output")
            self._check_alive()
            if self._closing:
                raise WorkerError("Worker is closing")
            self.props = props
            if self.socket_path is not None:
                self.socket_path.chmod(0o600)
            self._credentials.close()
            self._capture_load = False
            self._state = "ready"
            if self.telemetry.needs_native:
                try:
                    self._memory_client = self._http_client(
                        timeout=_MEMORY_TIMEOUT,
                        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                    )
                    self._memory_task = asyncio.create_task(self._poll_memory())
                except Exception:
                    self.telemetry.fail("collection_failed")
                    logger.warning("Could not initialize native memory HTTP client")
            return

    def memory_snapshot(self) -> dict:
        """Selected internal telemetry; public reporting is integrated separately."""
        return self.telemetry.snapshot()

    def memory_sizes(self):
        return self.telemetry.sizes()

    async def _fetch_memory(self):
        async with self._memory_client.stream("GET", "/tllama/memory") as response:
            if response.status_code != 200:
                reason = {401: "authentication_failed", 404: "endpoint_not_supported",
                          503: "not_ready"}.get(response.status_code, "http_error")
                self.telemetry.fail(reason, http_status=response.status_code)
                return
            report = NativeMemoryReport(await self._json_body(response))
            fingerprint = report.snapshot()["runtime_fingerprint"]
            if self.runtime_fingerprint is not None and fingerprint != self.runtime_fingerprint:
                self.telemetry.fail("runtime_fingerprint_mismatch", http_status=200)
                return
            self.telemetry.accept(report)

    async def refresh_memory(self):
        """Bounded diagnostic read; never acquire the inference slot or a lease."""
        async with self._memory_lock:
            if self._closing or self._state != "ready" or not self.telemetry.needs_native:
                return
            try:
                await asyncio.wait_for(self._fetch_memory(), timeout=_MEMORY_TIMEOUT)
            except UnsupportedMemorySchema:
                self.telemetry.fail("unsupported_schema", http_status=200)
            except MemoryReportError:
                self.telemetry.fail("invalid_report", http_status=200)
            except asyncio.TimeoutError:
                self.telemetry.fail("timeout")
            except httpx.HTTPError:
                self.telemetry.fail("transport_error")
            except WorkerError:
                self.telemetry.fail("invalid_response", http_status=200)
            except Exception:
                # Telemetry is optional even when its collector fails unexpectedly.
                self.telemetry.fail("collection_failed")
            reason = self.telemetry.failure_reason
            if reason is not None and reason != self._memory_failure:
                logger.warning("Native worker %s memory telemetry: %s (mode=%s)",
                               self.pid, reason, self.spec.memory_source)
            self._memory_failure = reason

    async def _poll_memory(self):
        while not self._closing and self.process.returncode is None and self.telemetry.needs_native:
            await self.refresh_memory()
            await asyncio.sleep(_MEMORY_INTERVAL)

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
        if not self._memory_parser_failed:
            try:
                self.memory.observe(line)
            except Exception:
                # Diagnostic parsing must never terminate a pipe drainer/inference.
                self._memory_parser_failed = True
                self.memory.invalidate('parser_error')
                logger.exception("Could not parse native memory diagnostics")
        if self._capture_load:
            size = len(line.encode("utf-8"))
            if self._load_bytes + size <= self.spec.log_limit_bytes:
                self._load_lines.append(line)
                self._load_bytes += size
            else:
                self.load_log_complete = False
                self.memory.invalidate('truncated_load_log')
        if self.spec.transport == "tcp" and self.base_url is None:
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
                self.memory.invalidate('truncated_log_line')
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
            if self._memory_task is not None:
                self._memory_task.cancel()
                await asyncio.gather(self._memory_task, return_exceptions=True)
            if self._memory_client is not None:
                try:
                    await asyncio.wait_for(self._memory_client.aclose(), self.spec.kill_timeout)
                except Exception:
                    logger.exception("Could not close native memory HTTP client")
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
            if self._credentials is not None:
                self._credentials.close()
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

    def _require_mode(self, *, embedding):
        if self.spec.mode != ("embedding" if embedding else "generation"):
            operation = "Embeddings" if embedding else "Generation"
            raise WorkerRequestError(400, f"{operation} is incompatible with this worker's configured mode")

    async def decision(self, payload: Mapping[str, Any]) -> dict:
        """Typed text Decision protocol; the caller owns the residency lease."""
        if self.spec.mode != "decision":
            raise WorkerRequestError(400, "Decision is incompatible with this worker's configured mode")
        body = prepare_decision_request(payload)
        body['model'] = self.spec.alias
        async with self._response('/v1/systemone', body) as response:
            result = await self._json_body(response)
        try:
            return validate_decision_response(result, body['questions'], self.spec.decision_type, self.spec.alias)
        except ValueError as exc:
            raise WorkerProtocolError(str(exc)) from exc

    async def embeddings(self, payload: Mapping[str, Any]) -> dict:
        """Raw pooled embedding protocol; the caller owns validation and its lease."""
        self._require_mode(embedding=True)
        body = {**payload, "model": self.spec.alias}
        async with self._response("/v1/embeddings", body) as response:
            return await self._json_body(response)

    async def chat(self, payload: Mapping[str, Any]) -> dict:
        self._require_mode(embedding=False)
        async with self._response("/v1/chat/completions", self._body(payload, chat=True, stream=False)) as response:
            return await self._json_body(response)

    async def completion(self, payload: Mapping[str, Any]) -> dict:
        self._require_mode(embedding=False)
        async with self._response("/completion", self._body(payload, chat=False, stream=False)) as response:
            return await self._json_body(response)

    async def render_chat(self, payload: Mapping[str, Any]) -> str:
        """Render with the same loaded template and options as chat inference."""
        async with self._response("/apply-template", self._body(payload, chat=True, stream=False)) as response:
            result = await self._json_body(response)
        prompt = result.get("prompt")
        if not isinstance(prompt, str):
            raise WorkerProtocolError("Native chat template must return a text prompt")
        return prompt

    async def tokenize(self, content: str, *, add_special: bool = True, parse_special: bool = True) -> list[int]:
        """Use this loaded worker's tokenizer, without evaluating the prompt."""
        if not isinstance(content, str) or type(add_special) is not bool or type(parse_special) is not bool:
            raise ValueError("Tokenization needs text and boolean special-token flags")
        async with self._response("/tokenize", {"content": content, "add_special": add_special,
                                                "parse_special": parse_special}) as response:
            payload = await self._json_body(response)
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise WorkerProtocolError("Native tokenizer must return nonnegative integer tokens")
        return tokens

    async def token_pieces(self, tokens: list[int]) -> list[str]:
        """Verify exact vocabulary entries without re-tokenizing their spellings."""
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Token pieces need nonnegative integer tokens")
        async with self._response("/tokenize", {"content": tokens, "add_special": False,
                                                "with_pieces": True}) as response:
            payload = await self._json_body(response)
        entries = payload.get("tokens")
        if (not isinstance(entries, list) or len(entries) != len(tokens)
                or any(not isinstance(entry, dict) or type(entry.get("id")) is not int
                       or entry["id"] != token or not isinstance(entry.get("piece"), str)
                       for token, entry in zip(tokens, entries))):
            raise WorkerProtocolError("Native token pieces must preserve requested IDs and string pieces")
        return [entry["piece"] for entry in entries]

    @asynccontextmanager
    async def chat_stream(self, payload: Mapping[str, Any]):
        self._require_mode(embedding=False)
        async with self._response("/v1/chat/completions", self._body(payload, chat=True, stream=True)) as response:
            events = self._events(response, chat=True)
            try:
                yield events
            finally:
                await events.aclose()

    @asynccontextmanager
    async def completion_stream(self, payload: Mapping[str, Any]):
        self._require_mode(embedding=False)
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
