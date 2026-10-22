"""Private credentials, runtime roots and UDS client isolation."""

import errno
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tllama.worker import NativeWorker, WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec
from tllama.worker_ipc import WorkerCredentials, runtime_directory
from test_native_worker import factory, records


@pytest.mark.parametrize('script', ['embeddings-smoke.py', 'embeddings-multimodal-smoke.py', 'mtp-smoke.py'])
def test_diagnostic_launch_extensions_forward_the_private_socket(script, monkeypatch):
    path = Path(__file__).parents[1] / 'scripts' / script
    name = 'ipc_probe_' + path.stem.replace('-', '_')
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, name, module)
    module_spec.loader.exec_module(module)
    spec = module.ProbeLaunchSpec(model=Path('/m.gguf'), device='none', transport='uds')
    args = spec.command(Path('/private/key'), socket_file=Path('/private/worker.sock'))
    assert args[args.index('--host') + 1] == '/private/worker.sock'


@pytest.mark.skipif(not hasattr(os, 'memfd_create'), reason='Linux memfd required')
def test_anonymous_key_is_readable_only_through_the_inherited_descriptor(tmp_path):
    credentials = WorkerCredentials(tmp_path)
    try:
        assert credentials.fd is not None
        assert credentials.path.read_text() == credentials.token + '\n'
        assert credentials.path.stat().st_mode & 0o777 == 0o600
        assert not (tmp_path / 'api-key').exists()
        # Return only success, never the secret value in test diagnostics.
        child = subprocess.run([sys.executable, '-c',
            'import pathlib,sys; data=pathlib.Path(sys.argv[1]).read_text(); '
            'assert len(data.strip()) == 43 and data.endswith("\\n")', str(credentials.path)],
            pass_fds=credentials.pass_fds, capture_output=True)
        assert child.returncode == 0, child.stderr.decode()
        child = subprocess.run([sys.executable, '-c',
            'import pathlib,sys; assert not pathlib.Path(sys.argv[1]).exists()', str(credentials.path)],
            capture_output=True)
        assert child.returncode == 0, child.stderr.decode()
        import fcntl
        if hasattr(fcntl, 'F_ADD_SEALS'):
            with pytest.raises(OSError):
                os.write(credentials.fd, b'x')
    finally:
        credentials.close()
    assert credentials.fd is None


@pytest.mark.parametrize('error', [None, errno.ENOSYS, errno.EPERM])
def test_private_file_fallback_is_exclusive_and_removed(tmp_path, monkeypatch, error):
    if error is None:
        monkeypatch.delattr(os, 'memfd_create', raising=False)
    else:
        def unsupported(*args):
            raise OSError(error, 'unavailable')
        monkeypatch.setattr(os, 'memfd_create', unsupported, raising=False)
    credentials = WorkerCredentials(tmp_path)
    try:
        assert credentials.pass_fds == ()
        assert credentials.path == tmp_path / 'api-key'
        assert credentials.path.read_text() == credentials.token + '\n'
        assert credentials.path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            WorkerCredentials(tmp_path)
    finally:
        credentials.close()
    assert not credentials.path.exists()


def test_failed_key_write_closes_the_memfd(tmp_path, monkeypatch):
    if not hasattr(os, 'memfd_create'):
        pytest.skip('Linux memfd required')
    opened = []
    create = os.memfd_create
    def recording(*args):
        fd = create(*args)
        opened.append(fd)
        return fd
    def failure(*args):
        raise OSError(errno.EIO, 'test write preparation failure')
    monkeypatch.setattr(os, 'memfd_create', recording)
    monkeypatch.setattr(os, 'fchmod', failure)
    with pytest.raises(OSError, match='test write preparation failure'):
        WorkerCredentials(tmp_path)
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_runtime_directory_prefers_private_service_then_session_root(tmp_path, monkeypatch):
    service, session = tmp_path / 'service', tmp_path / 'session'
    service.mkdir(mode=0o700)
    session.mkdir(mode=0o700)
    monkeypatch.setenv('RUNTIME_DIRECTORY', str(service))
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(session))
    with runtime_directory() as directory:
        assert Path(directory).parent == service
        assert Path(directory).stat().st_mode & 0o777 == 0o700
    service.chmod(0o755)
    with runtime_directory() as directory:
        assert Path(directory).parent == session
    assert not Path(directory).exists()


def test_runtime_directory_does_not_trust_relative_or_foreign_roots(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNTIME_DIRECTORY', 'relative')
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'missing'))
    with runtime_directory() as directory:
        assert Path(directory).is_absolute()
        assert Path(directory).stat().st_mode & 0o777 == 0o700


async def test_uds_clients_use_separate_private_pools_without_proxy_or_dns(monkeypatch, tmp_path):
    seen = []
    def transport(**kwargs):
        seen.append(kwargs)
        return httpx.MockTransport(lambda request: httpx.Response(200, json={'status':'ok'}))
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', transport)
    monkeypatch.setenv('HTTP_PROXY', 'http://foreign.invalid:80')
    worker = NativeWorker(WorkerLaunchSpec(model=Path('/m.gguf'), device='none'))
    worker.socket_path = tmp_path / 'worker.sock'
    worker.base_url = 'http://tllama-worker'
    worker._api_key = 'test-only-token'
    inference = worker._http_client(timeout=2, limits=httpx.Limits(max_connections=4))
    telemetry = worker._http_client(timeout=1, limits=httpx.Limits(max_connections=1))
    async with inference, telemetry:
        assert (await inference.get('/health')).status_code == 200
        response = await telemetry.get('/tllama/memory')
        assert response.request.headers['authorization'] == 'Bearer test-only-token'
        assert len(inference._mounts) == len(telemetry._mounts) == 0
    assert [item['limits'].max_connections for item in seen] == [4, 1]
    assert all(item['uds'] == str(worker.socket_path) and item['trust_env'] is False for item in seen)
    assert inference.is_closed and telemetry.is_closed


async def test_transport_still_requires_authentication_and_unique_credentials(factory):
    first, _ = factory()
    second, _ = factory()
    async with first, second:
        assert first._api_key != second._api_key
        assert first.address != second.address
        for worker in (first, second):
            transport = httpx.AsyncHTTPTransport(uds=str(worker.socket_path)) if worker.socket_path else None
            async with httpx.AsyncClient(base_url=worker.base_url, transport=transport, trust_env=False) as client:
                assert (await client.get('/props')).status_code == 401
                assert (await client.get('/props', headers={'Authorization':'Bearer wrong'})).status_code == 401
            assert (await worker.chat({}))['choices']


async def test_file_fallback_is_unlinked_after_authenticated_readiness(factory, monkeypatch):
    monkeypatch.delattr(os, 'memfd_create', raising=False)
    worker, root = factory()
    async with worker:
        launch = records(root, 'launch')[0]
        key_path = Path(launch['argv'][launch['argv'].index('--api-key-file') + 1])
        assert key_path == worker.runtime_directory / 'api-key'
        assert launch['key_mode'] == '0o600'
        assert not key_path.exists()
        assert (await worker.chat({}))['choices']


async def test_long_uds_path_fails_before_spawn_and_cleans_the_directory(tmp_path, monkeypatch):
    import tempfile
    parent = tmp_path / ('a' * 110)
    parent.mkdir(mode=0o700)
    monkeypatch.setattr('tllama.worker.runtime_directory',
                        lambda: tempfile.TemporaryDirectory(dir=parent))
    worker = NativeWorker(WorkerLaunchSpec(model=Path('/m.gguf'), device='none'))
    with pytest.raises(WorkerStartupError, match='UDS path is too long'):
        await worker.start()
    assert worker.process is None and not worker.runtime_directory.exists()
