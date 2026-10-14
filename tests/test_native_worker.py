"""Exercise owned subprocesses and real loopback sockets without a model/GPU."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest

from tllama.worker import (NativeWorker, WorkerError, WorkerExited, WorkerProtocolError,
                           WorkerRequestError, WorkerStartupError, WorkerStartupTimeout)
from tllama.worker_config import WorkerLaunchSpec


@pytest.fixture
async def factory(tmp_path):
    executable = tmp_path / "native-stub"
    stub = Path(__file__).with_name("native_worker_stub.py")
    executable.write_text(f"#!{sys.executable}\n" + stub.read_text())
    executable.chmod(0o700)
    children = []

    def make(mode="normal", **kwargs):
        root = tmp_path / str(len(children))
        root.mkdir()
        env = {"STUB_MODE": mode, "STUB_ROOT": str(root), **kwargs.pop("environment", {})}
        values = {"model": tmp_path / "m.gguf", "device": "none", "executable": executable,
                  "load_timeout": 5, "terminate_timeout": 0.2, "kill_timeout": 2, **kwargs}
        worker = NativeWorker(WorkerLaunchSpec(environment=env, **values))
        children.append(worker)
        return worker, root

    yield make
    await asyncio.gather(*(worker.close() for worker in children), return_exceptions=True)


def records(root, name):
    path = root / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def wait_marker(root, name):
    async def wait():
        while not (root / name).exists():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=2)


async def test_ready_and_close_own_all_resources(factory, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "invalid")
    monkeypatch.setenv("LLAMA_API_KEY", "foreign")
    monkeypatch.setenv("GGML_BACKEND_PATH", "/foreign")
    worker, root = factory(environment={"CUDA_VISIBLE_DEVICES": "2"})
    async with worker:
        assert worker.state == "ready"
        assert worker.pid
        assert worker.base_url.startswith("http://127.0.0.1:")
        assert worker.props["default_generation_settings"]["n_ctx"] == 2048
        launch = records(root, "launch")[0]
        assert launch["cwd"] == str(worker.runtime_directory)
        assert launch["cwd_mode"] == "0o700" and launch["key_mode"] == "0o600"
        assert launch["env"] == {"CUDA_VISIBLE_DEVICES": "2"}
        assert "no CUDA-capable device" in worker.load_log
        assert worker.load_log_complete
    assert worker.state == "closed"
    assert worker.process.returncode is not None
    assert not worker.runtime_directory.exists()
    await worker.close()


async def test_503_loading_is_retried(factory):
    worker, root = factory("loading")
    async with worker:
        assert len([r for r in records(root, "requests") if r["path"] == "/health"]) == 4


async def test_custom_template_is_an_owned_private_file_and_is_cleaned_up(factory):
    template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}ě\n"
    worker, root = factory(template=template, runtime_options={"n_threads": 2})
    async with worker:
        args = records(root, "launch")[0]["argv"]
        path = Path(args[args.index("--chat-template-file") + 1])
        assert path.parent == worker.runtime_directory
        assert path.read_text() == template and path.stat().st_mode & 0o777 == 0o600
        assert args[args.index("--threads") + 1] == "2" and "--jinja" in args
        assert (await worker.chat({}))["choices"]
    assert not path.exists()


@pytest.mark.parametrize("mode", ["wrong-build", "wrong-alias", "wrong-slots", "bad-context", "bad-health"])
async def test_readiness_checks_actual_runtime_and_model(factory, mode):
    worker, _ = factory(mode)
    with pytest.raises(WorkerStartupError):
        await worker.start()
    assert worker.process.returncode is not None
    assert not worker.runtime_directory.exists()


async def test_load_failure_includes_exit_and_log(factory):
    worker, _ = factory("fatal")
    with pytest.raises(WorkerStartupError, match="device kernel image is invalid"):
        await worker.start()
    assert worker.process.returncode == 23


async def test_missing_executable_cleans_up(factory):
    worker, _ = factory(executable=Path("/nonexistent/native-worker"))
    with pytest.raises(WorkerStartupError):
        await worker.start()
    assert worker.process is None
    assert worker.state == "closed"
    assert not worker.runtime_directory.exists()


async def test_load_timeout_kills_unresponsive_child(factory):
    worker, _ = factory("hang-load", load_timeout=0.3, environment={"STUB_IGNORE_TERM": "1"})
    with pytest.raises(WorkerStartupTimeout, match="Loading model"):
        await worker.start()
    assert worker.process.returncode == -signal.SIGKILL
    assert not worker.runtime_directory.exists()


async def test_cancelled_start_reaps_the_process(factory):
    worker, root = factory("hang-load", environment={"STUB_IGNORE_TERM": "1"})
    task = asyncio.create_task(worker.start())
    await wait_marker(root, "launch")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.process.returncode == -signal.SIGKILL
    assert not worker.runtime_directory.exists()


async def test_cancel_during_spawn_waits_for_and_reaps_late_child(factory, monkeypatch):
    worker, _ = factory("hang-load")
    entered, proceed = asyncio.Event(), asyncio.Event()
    spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        entered.set()
        await proceed.wait()
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    starting = asyncio.create_task(worker.start())
    await asyncio.wait_for(entered.wait(), timeout=2)
    starting.cancel()
    await asyncio.sleep(0)
    assert not starting.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=3)
    assert worker.process.returncode is not None
    assert not worker.runtime_directory.exists()


async def test_close_during_start_is_bounded(factory):
    worker, root = factory("hang-load")
    starting = asyncio.create_task(worker.start())
    await wait_marker(root, "launch")
    await asyncio.wait_for(worker.close(), timeout=3)
    with pytest.raises(WorkerStartupError):
        await starting
    assert worker.process.returncode is not None


async def test_cancelling_close_does_not_abandon_cleanup(factory):
    worker, _ = factory(environment={"STUB_IGNORE_TERM": "1"})
    await worker.start()
    closing = asyncio.create_task(worker.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert worker.process.returncode == -signal.SIGKILL
    assert not worker.runtime_directory.exists()


@pytest.mark.parametrize("mode", ["flood", "long-log"])
async def test_both_log_pipes_are_drained_and_capture_is_bounded(factory, mode):
    worker, _ = factory(mode, log_limit_bytes=2048)
    async with worker:
        assert worker.load_log_complete is False
        assert len(worker.load_log.encode()) <= 2048
        assert worker.state == "ready"
        assert worker.memory.snapshot()['allocated_bytes'] is None


async def test_private_authentication_key_is_redacted_from_logs(factory):
    worker, _ = factory("key-log")
    async with worker:
        key = (worker.runtime_directory / "api-key").read_text().strip()
        assert key not in worker.load_log and key not in worker.diagnostics
        assert "[redacted]" in worker.load_log


async def test_non_stream_chat_and_completion_preserve_native_results(factory):
    worker, root = factory()
    async with worker:
        chat = await worker.chat({"messages": [], "model": "foreign", "stream": True})
        assert chat["choices"][0]["message"] == {"content": "Ahoj světe", "reasoning_content": "myšlenka"}
        assert chat["usage"]["prompt_tokens"] == 25
        assert chat["usage"]["prompt_tokens_details"]["cached_tokens"] == 19
        assert chat["timings"]["prompt_n"] == 6
        raw = await worker.completion({"prompt": "hi"})
        assert raw["content"] == "Ahoj světe" and raw["stop_type"] == "eos"
        request = records(root, "requests")[-2]
        assert request["body"]["model"] == "tllama-worker"
        assert request["body"]["stream"] is False
        assert "x-conversation-id" not in request["header_names"]


async def test_sse_handles_fragmented_utf8_multiline_data_and_empty_choices(factory):
    worker, root = factory()
    async with worker:
        async with worker.chat_stream({"messages": []}) as events:
            chunks = [chunk async for chunk in events]
        assert chunks[0]["choices"][0]["delta"]["content"] == "Ahoj světe"
        assert chunks[1]["choices"][0]["delta"]["reasoning_content"] == "myšlenka"
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["prompt_tokens"] == 25
        assert chunks[-1]["timings"]["cache_n"] == 19
        request = records(root, "requests")[-1]
        assert request["body"]["stream_options"]["include_usage"] is True
        async with worker.completion_stream({"prompt": "hi"}) as events:
            chunks = [chunk async for chunk in events]
        assert chunks[-1]["stop"] is True
        assert chunks[-1]["timings"]["predicted_n"] == 10


@pytest.mark.parametrize("stream", [False, True])
async def test_request_errors_are_not_retried(factory, stream):
    worker, root = factory()
    async with worker:
        with pytest.raises(WorkerRequestError) as failure:
            if stream:
                async with worker.chat_stream({"stub": "error"}) as events:
                    await anext(events)
            else:
                await worker.chat({"stub": "error"})
        assert failure.value.status_code == 400
        assert len([r for r in records(root, "requests") if r["path"] == "/v1/chat/completions"]) == 1
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("mode,error", [("bad-json", WorkerProtocolError),
                                      ("truncated", WorkerProtocolError),
                                      ("error-stream", WorkerRequestError),
                                      ("json-stream", WorkerProtocolError)])
async def test_stream_failure_releases_the_local_slot(factory, mode, error):
    worker, _ = factory()
    async with worker:
        with pytest.raises(error):
            async with worker.chat_stream({"stub": mode}) as events:
                async for chunk in events:
                    pass
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]


async def test_early_stream_exit_closes_connection_and_releases_both_slots(factory):
    worker, root = factory()
    async with worker:
        async with worker.chat_stream({"stub": "hold"}) as events:
            await anext(events)
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]
        assert records(root, "cancelled") == ["stream"]


async def test_cancelled_stream_task_closes_connection_and_releases_both_slots(factory):
    worker, root = factory()
    async with worker:
        async def consume():
            async with worker.chat_stream({"stub": "hold"}) as events:
                async for chunk in events:
                    pass

        pending = asyncio.create_task(consume())
        await wait_marker(root, "started")
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]
        assert records(root, "cancelled") == ["stream"]


async def test_raw_stop_finishes_without_waiting_for_connection_eof(factory):
    worker, root = factory()
    async with worker:
        async def consume():
            async with worker.completion_stream({"stub": "hold-after-stop"}) as events:
                return [chunk async for chunk in events]

        chunks = await asyncio.wait_for(consume(), timeout=2)
        assert chunks[-1]["stop"] is True
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]
        assert records(root, "cancelled") == ["raw-terminal"]


async def test_non_stream_cancel_before_headers_closes_connection(factory):
    worker, root = factory()
    async with worker:
        task = asyncio.create_task(worker.chat({"stub": "hold"}))
        await wait_marker(root, "started")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]
        assert records(root, "cancelled") == ["non-stream"]


async def test_queued_request_waits_for_stream_owner_and_can_be_cancelled(factory):
    worker, root = factory()
    async with worker:
        async with worker.chat_stream({"stub": "hold"}) as events:
            await anext(events)
            pending = asyncio.create_task(worker.chat({}))
            try:
                await asyncio.sleep(0.05)
                assert not pending.done()
                assert len([r for r in records(root, "requests")
                            if r["path"] == "/v1/chat/completions"]) == 1
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        assert (await worker.chat({}))["choices"]


@pytest.mark.parametrize("stream", [False, True])
async def test_response_size_limit_closes_upstream_and_releases_slot(factory, monkeypatch, stream):
    worker, _ = factory()
    async with worker:
        with monkeypatch.context() as limited:
            limited.setattr("tllama.worker._MAX_EVENT" if stream else "tllama.worker._MAX_RESPONSE", 32)
            with pytest.raises(WorkerProtocolError, match="size limit"):
                if stream:
                    async with worker.chat_stream({}) as events:
                        await anext(events)
                else:
                    await worker.chat({})
        assert (await asyncio.wait_for(worker.chat({}), timeout=2))["choices"]


async def test_request_timeout_closes_connection(factory):
    worker, root = factory(request_timeout=0.1)
    async with worker:
        with pytest.raises(WorkerError, match="transport failed"):
            await worker.chat({"stub": "hold"})
        assert (await worker.chat({}))["choices"]
        assert records(root, "cancelled") == ["non-stream"]


async def test_close_interrupts_active_inference(factory):
    worker, root = factory()
    await worker.start()
    task = asyncio.create_task(worker.chat({"stub": "hold"}))
    await wait_marker(root, "started")
    await worker.close()
    with pytest.raises(WorkerError):
        await task
    assert worker.state == "closed"


async def test_crash_is_observable_and_has_no_automatic_restart(factory):
    worker, _ = factory()
    async with worker:
        pid = worker.pid
        os.kill(pid, signal.SIGKILL)
        assert await asyncio.wait_for(worker.wait_for_exit(), timeout=2) == -signal.SIGKILL
        assert worker.state == "failed"
        with pytest.raises(WorkerExited):
            await worker.chat({})
        assert worker.pid == pid


async def test_worker_is_single_use(factory):
    worker, _ = factory()
    await worker.close()
    with pytest.raises(WorkerError, match="only once"):
        await worker.start()


@pytest.mark.parametrize("case", ["normal", "crash", "cancel-start"])
async def test_hardware_smoke_entry_point(factory, case):
    worker, root = factory("hang-load" if case == "cancel-start" else "normal")
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "scripts" / "worker-smoke.py"),
               "--model", str(worker.spec.model), "--executable", str(worker.spec.executable),
               "--load-timeout", "5", "--request-timeout", "2"]
    if case == "crash":
        command.append("--crash")
    elif case == "cancel-start":
        command += ["--cancel-start-after", "0.2"]
    child = await asyncio.create_subprocess_exec(
        *command, cwd=repo, env={**os.environ, **worker.spec.environment, "PYTHONPATH": str(repo / "src")},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=10)
    finally:
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            await child.wait()
    assert child.returncode == 0, (stdout.decode(), stderr.decode())
    stages = [json.loads(line) for line in stdout.splitlines()]
    assert stages[-1]["stage"] == "closed" and stages[-1]["directory_removed"] is True
    assert stages[-1]["returncode"] is not None
    if case == "cancel-start":
        assert stages[0]["stage"] == "startup-cancelled"
        assert stages[0]["pid"] == stages[-1]["pid"] and stages[-1]["pid"] is not None
        # Cancellation may reap the child before its Python startup writes
        # the launch marker. The smoke still has to own and reap that PID.
        assert len(records(root, "launch")) <= 1
    else:
        assert len(records(root, "launch")) == 1
        assert records(root, "cancelled") == ["stream", "non-stream"]
        assert "after-non-stream-cancel" in [s["stage"] for s in stages]
        if case == "crash":
            assert stages[-2] == {"stage": "crash-detected", "returncode": -signal.SIGKILL}
