"""Verified streaming GGUF staging; create publishes complete managed groups."""

import asyncio
from hashlib import sha256
import logging
import os
import re

from tllama.residency import _settle_owned
from tllama.helpers.metadata_cache import build_model_file_fingerprint, save_digest_cache
from tllama.helpers.repository_files import repository_tempfile
from tllama.repository_pins import Reservation


def blob_path(manager, digest):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Expected sha256: followed by 64 lowercase hexadecimal characters.')
    directory = manager.repository.root / '.tllama' / 'uploads'
    path = directory / digest[7:]
    if (directory.is_symlink() or directory.parent.is_symlink() or path.is_symlink()
            or not path.resolve().is_relative_to(manager.repository.root)):
        raise ValueError('Upload staging must stay inside the model repository.')
    return path


def verified_blob(manager, digest):
    from tllama.repository_blobs import find_published_blob, verify_file
    path = blob_path(manager, digest)
    with manager.repository.lock():
        # Prefer data so a legacy alias can disappear without losing reuse.
        published = find_published_blob(manager.repository, digest[7:])
        return verify_file(manager.repository, published if published is not None else path, digest[7:])


def has_blob(manager, digest):
    path = blob_path(manager, digest)
    with manager.repository.lock():
        if path.is_file():
            return True
        from tllama.repository_blobs import find_published_blob
        return find_published_blob(manager.repository, digest[7:]) is not None


async def receive_blob(manager, digest, request):
    from tllama.repository_runtime import ensure_format
    await _settle_owned(asyncio.create_task(asyncio.to_thread(ensure_format, manager.repository)))
    async with Reservation(manager.repository, 'upload', paths=['.tllama/uploads']):
        await _receive_blob(manager, digest, request)
    # The last concurrent upload release can now reclaim aliases deferred by
    # a successful create. Unpublished blobs remain available for that create.
    from tllama.repository_blobs import remove_published_uploads
    warnings = await _settle_owned(asyncio.create_task(asyncio.to_thread(
        remove_published_uploads, manager.repository)))
    for warning in warnings:
        logging.getLogger(__name__).warning('%s', warning)


async def _receive_blob(manager, digest, request):
    target = blob_path(manager, digest)
    length = request.headers.get('content-length')
    if length is not None:
        try:
            length = int(length)
            if length < 0:
                raise ValueError
        except ValueError:
            raise ValueError('Invalid upload Content-Length.') from None
    manager.repository._directory('.tllama/uploads')
    descriptor, temporary = repository_tempfile(prefix='.upload-', directory=target.parent)
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
            from tllama.repository_blobs import verify_file
            await _settle_owned(asyncio.create_task(asyncio.to_thread(
                verify_file, manager.repository, target, digest[7:])))
        else:
            # Record our measured upload digest against the exact published
            # file, so create/list do not read many gigabytes again to hash.
            await _settle_owned(asyncio.create_task(asyncio.to_thread(
                save_digest_cache, manager.metadata_cache_dir, str(target), target,
                {'content_sha256': digest[7:], 'source': 'local'}, fingerprint=fingerprint,
                repository=manager.repository)))
    finally:
        temporary.unlink(missing_ok=True)
