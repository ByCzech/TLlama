"""Verified streaming GGUF staging; only create publishes files into Local."""

import asyncio
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile

from tllama.residency import _settle_owned
from tllama.helpers.metadata_cache import build_model_file_fingerprint, load_digest_cache, save_digest_cache


def blob_path(manager, digest):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Expected sha256: followed by 64 lowercase hexadecimal characters.')
    directory = manager.local_models_dir / '.uploads'
    path = directory / digest[7:]
    if path.is_symlink() or not path.resolve().is_relative_to(manager.models_dir.resolve()):
        raise ValueError('Upload staging must stay inside the model repository.')
    return path


def verified_blob(manager, digest):
    path = blob_path(manager, digest)
    before = build_model_file_fingerprint(path)
    cached = load_digest_cache(manager.metadata_cache_dir, path)
    if cached is None:
        cached = manager._build_model_digest_sync(path)
    if cached.get('content_sha256') != digest[7:] or before != build_model_file_fingerprint(path):
        raise ValueError('An uploaded blob changed or does not match its digest.')
    save_digest_cache(manager.metadata_cache_dir, str(path), path, cached, fingerprint=before)
    return path, before


async def receive_blob(manager, digest, request):
    target = blob_path(manager, digest)
    length = request.headers.get('content-length')
    if length is not None:
        try:
            length = int(length)
            if length < 0:
                raise ValueError
        except ValueError:
            raise ValueError('Invalid upload Content-Length.') from None
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.upload-', dir=target.parent)
    temporary = Path(temporary)
    digest_state = sha256()
    prefix = bytearray()
    size = 0
    try:
        with os.fdopen(descriptor, 'wb') as output:
            def write(chunk):
                output.write(chunk)
                digest_state.update(chunk)
            async for chunk in request.stream():
                if not chunk:
                    continue
                prefix.extend(chunk[:max(0, 4 - len(prefix))])
                if len(prefix) == 4 and prefix != b'GGUF':
                    raise ValueError('Uploads must contain GGUF files.')
                size += len(chunk)
                # Join a pending disk write before cancellation can close or
                # unlink its file. Hashing and file I/O stay off the event loop.
                await _settle_owned(asyncio.create_task(asyncio.to_thread(write, chunk)))
            if size < 24 or length is not None and length != size:
                raise ValueError('Incomplete GGUF upload.')
            if digest_state.hexdigest() != digest[7:]:
                raise ValueError('Uploaded bytes do not match the requested SHA-256.')
            def finish():
                output.flush()
                os.fsync(output.fileno())
            await _settle_owned(asyncio.create_task(asyncio.to_thread(finish)))
        fingerprint = build_model_file_fingerprint(temporary)
        fingerprint['path'] = str(target.resolve())
        try:
            os.link(temporary, target)
        except FileExistsError:
            # Another identical verified upload won publication. Never
            # replace its inode, which may already be used by a definition.
            if not target.is_file() or target.is_symlink():
                raise ValueError('Invalid existing upload target.')
            await _settle_owned(asyncio.create_task(asyncio.to_thread(verified_blob, manager, digest)))
        else:
            # Record our measured upload digest against the exact published
            # file, so create/list do not read many gigabytes again to hash.
            await _settle_owned(asyncio.create_task(asyncio.to_thread(
                save_digest_cache, manager.metadata_cache_dir, str(target), target,
                {'content_sha256': digest[7:], 'source': 'local'}, fingerprint=fingerprint)))
    finally:
        temporary.unlink(missing_ok=True)
