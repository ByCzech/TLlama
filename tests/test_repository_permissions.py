"""Exercise server-side publication with real POSIX masks in isolated processes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.helpers import repository_files
from tllama.helpers import model_toml


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


@pytest.mark.parametrize("mask", [0o002, 0o022, 0o027, 0o077])
def test_create_pull_upload_cache_and_copy_respect_server_umask(tmp_path, mask):
    # umask is process-wide. Even the test must not change pytest's mask
    # while another test's background thread may still be doing disk I/O.
    script = r'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import gguf
import httpx
from starlette.requests import Request

root = Path(sys.argv[1])
mask = int(sys.argv[2])
os.environ['TLLAMA_MODELS'] = str(root / 'initial-store')
os.umask(mask)
def forbidden(*args):
    raise AssertionError('Repository writers must never change the process umask')
os.umask = forbidden

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml, write_model_toml
from tllama.model_uploads import receive_blob, blob_path
from tllama.model_definitions import create_definition, copy_definition
from tllama.main import app
from tllama.routers import ollama

expected = 0o666 & ~mask
manager = ModelManager(BackendConfig(models_dir=str(root / 'models')))
manager.ensure_storage()
ollama.model_manager = manager
source = root / 'source.gguf'
writer = gguf.GGUFWriter(str(source), arch='llama')
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.close()
payload = source.read_bytes()
digest = 'sha256:' + sha256(payload).hexdigest()

def mode(path):
    return stat.S_IMODE(path.stat().st_mode)

async def exercise():
    async def receive():
        return {'type': 'http.request', 'body': payload, 'more_body': False}
    await receive_blob(manager, digest, Request({'type': 'http', 'headers': []}, receive))
    blob = blob_path(manager, digest)
    assert mode(blob) == expected, (blob, oct(mode(blob)))
    result = create_definition(manager, 'created', '[llm]\nfrom="source.gguf"\n',
                               {'llm': [SimpleNamespace(filename='source.gguf', digest=digest)]})
    assert result['status'] == 'success'
    definition = manager.local_models_dir / 'created.toml'
    assert mode(definition) == expected
    imported = manager.models_dir / parse_model_toml(definition.read_text()).llm_model
    assert os.path.samefile(imported, blob) and mode(imported) == expected
    assert imported.read_bytes() == payload
    copy_definition(manager, 'created', 'copied')
    assert mode(manager.local_models_dir / 'copied.toml') == expected

    # Force replaces only the definition inode; selected access bits survive
    # even when they differ from the current process mask. Weights stay shared.
    for access in (0o600, 0o640, 0o664):
        definition.chmod(access)
        create_definition(manager, 'created', definition.read_text(), {}, force=True)
        assert mode(definition) == access
        assert os.path.samefile(imported, blob) and mode(blob) == expected

    # The HF downloader owns GGUF permissions; substitute only that external
    # transfer, then exercise actual /api/pull registration and cache writes.
    async def download(repo_id, filename, **kwargs):
        target = manager.hf_models_dir / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)
    manager.pull_hf_file = download
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/pull', json={'model': 'ns/repo/model.gguf', 'stream': False})
    assert response.status_code == 200, response.text
    pulled = manager.hf_models_dir / 'ns' / 'repo' / 'model.toml'
    assert mode(pulled) == expected
    caches = list(manager.metadata_cache_dir.rglob('*.json'))
    assert any(p.name.endswith('.digest.json') for p in caches)
    assert any(not p.name.endswith('.digest.json') for p in caches)
    assert all(mode(p) == expected for p in caches), [(str(p), oct(mode(p))) for p in caches]
    assert mode(imported.parent) == 0o777 & ~mask

asyncio.run(exercise())

# Parallel publishers retain whole documents and unique paths without any
# process-wide mask manipulation or temporary artifacts left behind.
directory = root / 'parallel'
directory.mkdir()
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(lambda i: write_model_toml(directory / f'{i}.toml', f'# {i}\n'), range(32)))
assert len(list(directory.iterdir())) == 32
assert all(mode(p) == expected and p.read_text() == f'# {p.stem}\n' for p in directory.iterdir())
assert not list(manager.models_dir.rglob('*.tmp'))
assert not list(manager.models_dir.rglob('.upload-*'))
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-W", "error", "-c", script, str(tmp_path), str(mask)],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("symlink", [False, True])
def test_temporary_name_collision_never_clobbers_or_follows_existing(tmp_path, monkeypatch, symlink):
    victim = tmp_path / 'victim'
    victim.write_text('retain')
    collision = tmp_path / '.tmp-collision'
    if symlink:
        collision.symlink_to(victim)
    else:
        collision.write_text('retain')
    names = iter(['collision', 'unique'])
    monkeypatch.setattr(repository_files.secrets, 'token_hex', lambda _: next(names))
    fd, path = repository_files.repository_tempfile(directory=tmp_path, prefix='.tmp-')
    with os.fdopen(fd, 'w') as output:
        output.write('new')
    assert path.read_text() == 'new'
    assert collision.read_text() == victim.read_text() == 'retain'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux default ACL API')
def test_new_definition_inherits_default_acl_without_post_creation_chmod(tmp_path):
    script = r'''
import ctypes
import os
from pathlib import Path
import sys
try:
    library = ctypes.CDLL('libacl.so.1', use_errno=True)
except OSError:
    sys.exit(77)
library.acl_from_text.argtypes = [ctypes.c_char_p]
library.acl_from_text.restype = ctypes.c_void_p
library.acl_set_file.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
library.acl_free.argtypes = [ctypes.c_void_p]
acl = library.acl_from_text(b'u::rwx,g::rwx,o::---')
assert acl
try:
    status = library.acl_set_file(os.fsencode(sys.argv[1]), 0x4000, acl)
    if status != 0:
        import errno
        error = ctypes.get_errno()
        if error in (errno.ENOTSUP, errno.EPERM):
            sys.exit(77)
        raise OSError(error, os.strerror(error))
finally:
    library.acl_free(acl)
os.umask(0o077)
from tllama.helpers.model_toml import write_model_toml
target = Path(sys.argv[1]) / 'model.toml'
write_model_toml(target, '# ACL inheritance\n')
assert target.stat().st_mode & 0o777 == 0o660, oct(target.stat().st_mode)
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-W', 'error', '-c', script, str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=15)
    if result.returncode == 77:
        pytest.skip('libacl/default ACLs unavailable on this host')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("operation", ['replace', 'fchmod'])
def test_failed_overwrite_keeps_original_and_cleans_temporary(tmp_path, monkeypatch, operation):
    target = tmp_path / 'model.toml'
    target.write_text('original')
    target.chmod(0o640)
    def fail(*args):
        raise PermissionError('injected publication failure')
    monkeypatch.setattr(model_toml.os, operation, fail)
    with pytest.raises(PermissionError, match='injected'):
        model_toml.write_model_toml(target, 'replacement', overwrite=True)
    assert target.read_text() == 'original'
    assert target.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [target]


def test_private_definition_has_no_more_permissive_replacement_temporary(tmp_path, monkeypatch):
    target = tmp_path / 'model.toml'
    target.write_text('private document')
    target.chmod(0o600)
    original = os.fchmod
    checked = []
    def inspect(fd, mode):
        before = os.fstat(fd).st_mode & 0o777
        assert before & ~0o600 == 0
        checked.append(mode)
        original(fd, mode)
    monkeypatch.setattr(model_toml.os, 'fchmod', inspect)
    model_toml.write_model_toml(target, 'replacement', overwrite=True)
    assert checked == [0o600]
    assert target.read_text() == 'replacement' and target.stat().st_mode & 0o777 == 0o600
