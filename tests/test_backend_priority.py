"""Backend preference parsing, native selector families and real child ownership."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest

from tllama.automatic_worker import AutomaticWorker, acceleration_failure
from tllama.backend import ModelManager
from tllama.backend_policy import DEFAULT_BACKEND_PRIORITY, backend_priority, device_candidates, matches_backend
from tllama.config import BackendConfig
from tllama.daemon_config import load_daemon_config
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import resolve_worker_config
from tllama.worker import NativeWorker, WorkerError, WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec


pytestmark = pytest.mark.usefixtures('clean_tllama_config_environment')


@pytest.mark.parametrize('value,expected', [
    (None, DEFAULT_BACKEND_PRIORITY), ('', ()), ('  ', ()),
    (' vulkan , HIP, rocm,CPU,CUDA ', ('Vulkan', 'ROCm', 'CPU', 'CUDA')),
    ('Vulcan,CUDA', ('Vulcan', 'CUDA')), ('BLAS,BLIS,ZenDNN', ()),
    ('Vulkan,,../../bad,CUDA', ('Vulkan', 'CUDA')),
])
def test_environment_distinguishes_absence_empty_and_order(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv('TLLAMA_BACKEND_PRIORITY', value)
    config = load_daemon_config(False, discover=False).backend
    assert config.backend_priority == expected
    assert config.runtime_overrides == {}


@pytest.mark.parametrize('backend,selector', [
    ('CUDA', 'CUDA0'), ('ROCm', 'ROCm10'), ('SYCL', 'SYCL2'), ('Vulkan', 'Vulkan0'),
    ('Metal', 'MTL1'), ('OpenCL', 'GPUOpenCL'), ('WebGPU', 'WebGPU'),
    ('CANN', 'CANN2'), ('MUSA', 'MUSA0'), ('Hexagon', 'HTP0:1'),
    ('OpenVINO', 'OPENVINO0'), ('zDNN', 'zDNN'), ('ET', 'ET0'), ('RPC', 'RPC1'),
    ('VirtGPU', '[virtgpu] CUDA0'), ('Future', 'Future'), ('Future', 'Future42'),
])
def test_native_selector_conventions_and_conservative_new_names(backend, selector):
    assert matches_backend(backend, selector)


@pytest.mark.parametrize('selector', ['FutureExperimental0', 'Future0:1', 'Future 0', 'OtherFuture0', 'Future０'])
def test_unknown_names_do_not_use_broad_prefix_matching(selector):
    assert not matches_backend('Future', selector)


def test_cpu_is_terminal_and_unknown_names_never_restore_defaults(caplog):
    selectors = ('CUDA0', 'CUDA1', 'ROCm0', 'Vulkan0', 'VulkanExperimental0')
    assert list(device_candidates(backend_priority('Vulkan,CUDA'), selectors)) == [
        ('Vulkan', 'Vulkan0'), ('CUDA', 'CUDA0,CUDA1'), ('CPU', 'none')]
    assert list(device_candidates(backend_priority('CUDA,CPU,Vulkan'), selectors)) == [
        ('CUDA', 'CUDA0,CUDA1'), ('CPU', 'none')]
    assert list(device_candidates(backend_priority('Vulcan'), selectors)) == [('CPU', 'none')]
    assert 'Vulcan' in caplog.text
    assert list(device_candidates((), selectors)) == [('CPU', 'none')]
    assert list(device_candidates(('Future',), ('Future0,CUDA0',))) == [('CPU', 'none')]


@pytest.mark.parametrize('device', ['none', 'ROCm0', 'Vulkan0'])
def test_explicit_device_overrides_allowlist_and_model_overrides_global(device):
    config = BackendConfig(runtime_overrides={'device': 'CUDA0'}, backend_priority=())
    toml = parse_model_toml(f'[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="{device}"')
    resolved = resolve_worker_config(config, Path('/models/a.gguf'), toml)
    assert resolved.launch.device == device


@pytest.mark.parametrize('backend,message,expected', [
    ('ROCm', 'ROCm error: no kernel image is available for execution on the device', True),
    ('ROCm', 'rocBLAS error: Cannot read TensileLibrary.dat: No such file', True),
    ('Vulkan', 'failed to initialize Vulkan0 backend', True),
    ('Vulkan', 'CUDA driver version is insufficient; invalid GGUF model', False),
    ('ROCm', 'unsupported model architecture', False),
    ('ROCm', 'out of memory', False),
    ('Vulkan', 'failed to load model', False),
])
def test_retry_requires_selected_accelerator_evidence(backend, message, expected):
    assert acceleration_failure(backend, message) is expected


@pytest.fixture
def auto_factory(tmp_path):
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    root = tmp_path / 'stub'
    root.mkdir()
    children = []

    def make(*, priority=('ROCm', 'Vulkan'), selectors=('ROCm0', 'Vulkan0'), failures=None,
             discovery=None, **values):
        environment = {'STUB_ROOT': str(root), 'STUB_DISCOVERY': discovery or json.dumps(selectors),
                       'STUB_DEVICE_FAILURES': json.dumps(failures or {}), **values.pop('environment', {})}
        spec = WorkerLaunchSpec(model=Path('/models/a.gguf'), device=None, executable=executable,
                                backend_priority=priority, load_timeout=5,
                                terminate_timeout=.2, kill_timeout=1, environment=environment, **values)
        def child(launch):
            worker = NativeWorker(launch)
            children.append(worker)
            return worker
        return AutomaticWorker(spec, worker_factory=child), root, children
    return make


async def wait_file(path):
    async def waiting():
        while not path.exists():
            await asyncio.sleep(.01)
    await asyncio.wait_for(waiting(), 3)


async def test_rocm_failure_uses_vulkan_and_reaps_before_constructing_next(auto_factory):
    worker, root, children = auto_factory(failures={'ROCm0': 'ROCm error: no kernel image is available'})
    await worker.start()
    try:
        assert worker.spec.device == 'Vulkan0'
        assert children[0].process.returncode == 23 and not children[0].runtime_directory.exists()
        assert len(children) == 2 and children[1].state == 'ready'
        assert [item['outcome'] for item in worker.selection_attempts] == ['failed', 'ready']
        args = worker.command
        assert args[args.index('--n-gpu-layers') + 1] == 'auto'
        assert args[args.index('--fit') + 1] == 'on'
        assert (await worker.completion({'prompt': 'hello'}))['content']
        probe = json.loads((root / 'discovery').read_text())
        assert probe['argv'] == ['--list-devices'] and not Path(probe['cwd']).exists()
    finally:
        await worker.close()
    assert all(child.process.returncode is not None and not child.runtime_directory.exists() for child in children)


@pytest.mark.parametrize('priority', [(), ('CPU',), ('CPU', 'Vulkan')])
async def test_cpu_first_and_empty_do_not_probe_accelerators(auto_factory, priority):
    worker, root, children = auto_factory(priority=priority)
    await worker.start()
    try:
        assert worker.spec.device == 'none' and not (root / 'discovery').exists()
        assert worker.command[worker.command.index('--n-gpu-layers') + 1] == '0'
        assert '--no-op-offload' in worker.command and '--no-kv-offload' in worker.command
        assert len(children) == 1
    finally:
        await worker.close()


@pytest.mark.parametrize('discovery', ['bad', 'error', 'flood', 'hang', '[]'])
async def test_unavailable_discovery_still_allows_cpu_and_cleans_probe(auto_factory, discovery):
    worker, root, children = auto_factory(discovery=discovery)
    await worker.start()
    try:
        assert worker.spec.device == 'none' and len(children) == 1
        probe = json.loads((root / 'discovery').read_text())
        assert not Path(probe['cwd']).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(probe['pid'], 0)
    finally:
        await worker.close()


async def test_all_gpu_initialization_failures_reach_cpu(auto_factory):
    worker, _, children = auto_factory(failures={
        'ROCm0': 'rocBLAS error: Cannot read TensileLibrary.dat',
        'Vulkan0': 'failed to initialize Vulkan0 backend'})
    await worker.start()
    try:
        assert worker.spec.device == 'none' and len(children) == 3
    finally:
        await worker.close()


@pytest.mark.parametrize('layers', [-1, 3])
async def test_automatic_selection_preserves_explicit_gpu_counts_and_context(auto_factory, layers):
    worker, _, _ = auto_factory(n_gpu_layers=layers, n_ctx=4096, type_k='q8_0', flash_attn=True)
    await worker.start()
    try:
        args = worker.command
        assert args[args.index('--n-gpu-layers') + 1] == ('all' if layers == -1 else '3')
        assert args[args.index('--fit') + 1] == 'off'
        assert args[args.index('--ctx-size') + 1] == '4096'
        assert args[args.index('--cache-type-k') + 1] == 'q8_0'
        assert args[args.index('--flash-attn') + 1] == 'on'
    finally:
        await worker.close()


async def test_close_during_loading_reaps_without_trying_another_backend(auto_factory):
    worker, root, children = auto_factory(environment={'STUB_MODE': 'hang-load'})
    starting = asyncio.create_task(worker.start())
    await wait_file(root / 'launch')
    await asyncio.wait_for(worker.close(), 3)
    with pytest.raises((WorkerError, asyncio.CancelledError)):
        await starting
    assert len(children) == 1 and children[0].process.returncode is not None
    assert not children[0].runtime_directory.exists()


async def test_model_error_is_not_retried_on_another_backend(auto_factory):
    worker, _, children = auto_factory(failures={'ROCm0': 'invalid GGUF model header'})
    with pytest.raises(WorkerStartupError, match='invalid GGUF'):
        await worker.start()
    assert len(children) == 1 and children[0].process.returncode is not None


async def test_cancel_discovery_reaps_and_removes_private_directory(auto_factory):
    worker, root, children = auto_factory(discovery='hang')
    starting = asyncio.create_task(worker.start())
    await wait_file(root / 'discovery')
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    probe = json.loads((root / 'discovery').read_text())
    assert not children and not Path(probe['cwd']).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(probe['pid'], 0)


async def test_discovery_uses_same_clean_environment(auto_factory, monkeypatch):
    monkeypatch.setenv('LLAMA_ARG_MODEL', '/foreign/weights.gguf')
    monkeypatch.setenv('GGML_BACKEND_PATH', '/foreign/backend.so')
    worker, root, _ = auto_factory(environment={'CUDA_VISIBLE_DEVICES': '2'})
    await worker.start()
    try:
        assert json.loads((root / 'discovery').read_text())['env'] == {'CUDA_VISIBLE_DEVICES': '2'}
    finally:
        await worker.close()


async def test_failed_model_does_not_stop_manager_and_explicit_device_skips_probe(tmp_path, auto_factory):
    template, root, _ = auto_factory()
    config = BackendConfig(models_dir=str(tmp_path / 'models'), backend_priority=(),
                           runtime_executable=template.spec.executable)
    children = []
    def factory(spec):
        child = NativeWorker(replace(spec, environment={**template.spec.environment,
            'STUB_DEVICE_FAILURES': json.dumps({'ROCm0': 'ROCm error: no kernel image is available'})}))
        children.append(child)
        return child
    manager = ModelManager(config, native_worker_factory=factory,
                           native_launch_options={'environment': template.spec.environment, 'load_timeout': 5})
    await manager.start()
    for name, extra in [('bad', '[runtime]\ndevice="ROCm0"'), ('good', '')]:
        (manager.local_models_dir / (name + '.gguf')).write_bytes(b'fake GGUF')
        (manager.local_models_dir / (name + '.toml')).write_text(f'[llm]\nmodel="Local/{name}.gguf"\n' + extra)
    try:
        with pytest.raises(WorkerStartupError):
            await manager.preload_native_model('bad')
        assert not (root / 'discovery').exists() and manager.native_snapshot() == []
        async with manager.acquire_native_model('good', keep_alive=0) as lease:
            assert lease.worker.spec.device == 'none'
            assert (await lease.worker.completion({'prompt': 'hello'}))['content']
        await manager.wait_native_closed(lease.generation)
        assert manager.native_snapshot() == []
    finally:
        await manager.shutdown()
    assert all(child.process.returncode is not None for child in children)


async def test_concurrent_preloads_share_one_automatic_generation_and_failed_attempt(tmp_path, auto_factory):
    template, _, _ = auto_factory(failures={'ROCm0': 'ROCm error: no kernel image is available'})
    config = BackendConfig(models_dir=str(tmp_path / 'models'), backend_priority=('ROCm', 'Vulkan'),
                           runtime_executable=template.spec.executable)
    children = []
    def factory(spec):
        if children:
            assert children[-1].process.returncode is not None
            assert not children[-1].runtime_directory.exists()
        child = NativeWorker(replace(spec, environment=template.spec.environment))
        children.append(child)
        return child
    manager = ModelManager(config, native_worker_factory=factory,
                           native_launch_options={'environment': template.spec.environment, 'load_timeout': 5})
    await manager.start()
    assert not children
    (manager.local_models_dir / 'a.gguf').write_bytes(b'fake GGUF')
    (manager.local_models_dir / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n')
    try:
        numbers = await asyncio.gather(*(manager.preload_native_model('a') for _ in range(5)))
        assert len(set(numbers)) == 1 and len(children) == 2
        assert len(manager.native_snapshot()) == 1
        async with manager.acquire_native_model('a') as lease:
            assert lease.worker.spec.device == 'Vulkan0' and lease.resolved.launch.device is None
            assert (await lease.worker.completion({'prompt': 'hello'}))['content']
        await manager.unload_native_model('a')
        await manager.wait_native_closed(numbers[0])
        assert manager.native_snapshot() == []
    finally:
        await manager.shutdown()
    assert all(child.process.returncode is not None for child in children)
