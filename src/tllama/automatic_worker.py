"""One residency generation owning bounded, sequential backend startup attempts."""

import asyncio
from dataclasses import replace
import logging
import os
import re
import signal
import tempfile

from tllama.backend_policy import device_candidates
from tllama.device_inventory import parse_device_list
from tllama.native_memory import NativeMemory
from tllama.native_telemetry import MemoryTelemetry
from tllama.worker import NativeWorker, WorkerError, WorkerStartupError, WorkerStartupTimeout


logger = logging.getLogger(__name__)
_OUTPUT_LIMIT = 4 * 1024 * 1024
# Positive evidence of an acceleration/driver failure, not a generic model,
# capacity, protocol or timeout error. Never replay an inference request.
_KERNEL_FAILURE = re.compile(
    r'no kernel image is available|hipErrorNoBinaryForGpu|cudaErrorNoKernelImageForDevice|'
    r'invalid device function|hipErrorInvalidDeviceFunction', re.I)
_ROCM_FAILURE = re.compile(
    r'(?:rocBLAS|rocblas|Tensile).*(?:cannot|could not|unable|failed|not found|No such file)|'
    r'(?:cannot|could not|unable|failed).*TensileLibrary|'
    r'hipErrorInvalidDevice', re.I)


def acceleration_failure(backend, message):
    # An unrelated CUDA warning while trying Vulkan is not retry evidence.
    prefix = {'Metal': 'MTL', 'Hexagon': 'HTP', 'OpenCL': 'GPUOpenCL'}.get(backend, backend)
    if re.search(r'failed to initialize ' + re.escape(prefix) + r'[0-9:]* backend', message, re.I):
        return True
    if backend in {'CUDA', 'ROCm', 'MUSA'} and _KERNEL_FAILURE.search(message):
        return True
    if backend == 'ROCm' and _ROCM_FAILURE.search(message):
        return True
    marker = {'CUDA': r'CUDA driver version is insufficient|no CUDA-capable device is detected',
              'Vulkan': r'VK_ERROR_INCOMPATIBLE_DRIVER',
              'SYCL': r'ZE_RESULT_ERROR_UNSUPPORTED_FEATURE'}.get(backend)
    return bool(marker and re.search(marker, message, re.I))


async def _finish(task):
    """Own cleanup to completion even if its caller is cancelled repeatedly."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


async def discover_devices(spec):
    """Probe the actual executable in the worker environment, with no model."""
    process, readers = None, []
    output = [bytearray(), bytearray()]

    async def collect(pipe, index):
        while chunk := await pipe.read(65536):
            if sum(map(len, output)) + len(chunk) > _OUTPUT_LIMIT:
                raise WorkerError('Native device discovery exceeded its output limit')
            output[index].extend(chunk)

    async def cleanup(spawn):
        child = await spawn
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

        async def discard(pipe):
            while await pipe.read(65536):
                pass

        # Drain without retaining bytes after overflow. A paused pipe can
        # otherwise prevent asyncio's process wait from completing after exit.
        drains = [asyncio.create_task(discard(pipe)) for pipe in (child.stdout, child.stderr)]
        def stop(sig):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        try:
            stop(signal.SIGTERM)
            try:
                await asyncio.wait_for(child.wait(), 1)
            except asyncio.TimeoutError:
                stop(signal.SIGKILL)
                await asyncio.wait_for(child.wait(), 1)
            # A descendant may retain a pipe after the direct child exits.
            stop(signal.SIGKILL)
            await asyncio.wait_for(asyncio.gather(*drains), 1)
        except asyncio.TimeoutError as exc:
            # Cleanup failure is terminal, never permission to start more children.
            raise RuntimeError('Could not clean the native device discovery process group') from exc
        finally:
            for drain in drains:
                drain.cancel()
            await asyncio.gather(*drains, return_exceptions=True)

    with tempfile.TemporaryDirectory(prefix='tllama-device-probe-') as private:
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            str(spec.executable), '--list-devices', cwd=private,
            env=spec.child_environment(os.environ), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True))
        try:
            process = await asyncio.shield(spawn)
            readers = [asyncio.create_task(collect(process.stdout, 0)),
                       asyncio.create_task(collect(process.stderr, 1))]
            await asyncio.wait_for(asyncio.gather(*readers, process.wait()), min(10, spec.load_timeout / 4))
            if process.returncode != 0:
                raise WorkerError(f'Native device discovery exited with status {process.returncode}')
            inventory = parse_device_list(bytes(output[0]))
            if inventory['status'] != 'complete':
                raise WorkerError('Native device discovery table is incomplete or ambiguous')
            return tuple(row['selector'] for row in inventory['devices'])
        finally:
            # Even cancellation during spawn cannot orphan a process or cwd.
            if process is not None or not spawn.done() or not spawn.exception():
                await _finish(asyncio.create_task(cleanup(spawn)))


class AutomaticWorker:
    """Proxy the successful native worker; own every failed child until reaped."""

    def __init__(self, spec, *, worker_factory=NativeWorker):
        self._requested = spec
        self._factory = worker_factory
        self._worker = None
        self._run_task = None
        self._close_task = None
        self._closing = False
        self._state = 'new'
        self.selection_attempts = []
        self._telemetry = MemoryTelemetry(NativeMemory(None, projector=spec.mmproj is not None),
                                          mode=spec.memory_source, compare=spec.memory_compare)

    def __getattr__(self, name):
        if self._worker is None:
            raise AttributeError(name)
        return getattr(self._worker, name)

    @property
    def spec(self):
        return self._worker.spec if self._worker else self._requested

    @property
    def state(self):
        return self._worker.state if self._state == 'ready' else self._state

    @property
    def process(self):
        return self._worker.process if self._worker else None

    @property
    def pid(self):
        return self._worker.pid if self._worker else None

    @property
    def props(self):
        return self._worker.props if self._worker else None

    @property
    def diagnostics(self):
        return self._worker.diagnostics if self._worker else ''

    def memory_snapshot(self):
        return self._worker.memory_snapshot() if self._worker else self._telemetry.snapshot()

    def memory_sizes(self):
        return self._worker.memory_sizes() if self._worker else self._telemetry.sizes()

    async def start(self):
        if self._state != 'new' or self._closing:
            raise WorkerError('An automatic worker can be started only once')
        self._state = 'starting'
        self._run_task = asyncio.create_task(self._start_impl())
        try:
            await asyncio.wait_for(asyncio.shield(self._run_task), self._requested.load_timeout)
        except BaseException as exc:
            await self.close()
            if isinstance(exc, asyncio.TimeoutError):
                raise WorkerStartupTimeout('Automatic backend selection exceeded the model load deadline') from exc
            raise
        return self

    async def _start_impl(self):
        spec = self._requested
        # CPU first/empty never probes GPUs or initializes them for discovery.
        selectors = ()
        if spec.backend_priority and spec.backend_priority[0] != 'CPU':
            try:
                selectors = await discover_devices(spec)
            except (OSError, UnicodeError, WorkerError, asyncio.TimeoutError) as exc:
                logger.warning('Automatic device discovery failed; trying CPU: %s', exc)
        for backend, device in device_candidates(spec.backend_priority, selectors):
            if self._closing:
                raise WorkerError('Automatic worker is closing')
            options = dict(spec.runtime_options)
            if device == 'none':
                options.update(op_offload=False, offload_kqv=False)
            candidate = replace(spec, device=device, n_gpu_layers=0 if device == 'none' else spec.n_gpu_layers,
                                fit=device != 'none' and (spec.fit if spec.fit is not None
                                                        else spec.n_gpu_layers is None), runtime_options=options)
            self._worker = self._factory(candidate)
            record = {'backend': backend, 'device': device, 'outcome': 'starting'}
            self.selection_attempts.append(record)
            logger.info('Automatic backend selection: trying %s with device=%s', backend, device)
            try:
                await self._worker.start()
            except WorkerStartupError as exc:
                # NativeWorker.start already closes on failure; custom factories
                # must also be closed before another child is constructed.
                await self._worker.close()
                record['outcome'] = 'failed'
                evidence = str(exc) + '\n' + getattr(self._worker, 'load_log', '')
                if device == 'none' or not acceleration_failure(backend, evidence):
                    raise
                logger.warning('Automatic backend %s failed to initialize acceleration; trying next: %s', backend, exc)
                continue
            record['outcome'] = 'ready'
            self._state = 'ready'
            logger.info('Automatic backend selection: selected %s with device=%s', backend, device)
            return

    async def close(self):
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_impl())
        await _finish(self._close_task)

    async def _close_impl(self):
        self._state = 'closing'
        if self._run_task is not None:
            self._run_task.cancel()
            await asyncio.gather(self._run_task, return_exceptions=True)
        if self._worker is not None:
            await self._worker.close()
        self._state = 'closed'
