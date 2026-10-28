"""Read from paths on the client, upload verified bytes, then publish TOML."""

import asyncio
from contextlib import aclosing
from hashlib import sha256
import os
from pathlib import Path
import sys
import threading

from tllama.client import ClientError
from tllama.cli_status import WaitStatus
from tllama.cli_output import size_text
from tllama.helpers.model_toml import parse_model_toml, validate_definition_name

CHUNK = 1024 * 1024
MAX_DEFINITION = 2 * 1024 * 1024


async def _file_job(operation, *args):
    """Join file work before cancellation can close its resources."""
    stop = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(operation, *args, stop=stop))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            stop.set()
            cancelled = True
            if task.done():
                if not task.cancelled():
                    task.exception()
                raise
    if cancelled:
        raise asyncio.CancelledError
    return result


def _signature(file):
    stat = os.fstat(file.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash(path, *, stop):
    digest = sha256()
    with path.open('rb') as file:
        before = _signature(file)
        while not stop.is_set():
            chunk = file.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        if stop.is_set():
            return None
        if _signature(file) != before:
            raise ClientError(f'Import file changed while hashing: {path}')
        return 'sha256:' + digest.hexdigest(), before


def _read(file, *, stop):
    return file.read(CHUNK)


async def _chunks(path, signature):
    with path.open('rb') as file:
        if _signature(file) != signature:
            raise ClientError(f'Import file changed before upload: {path}')
        while True:
            chunk = await _file_job(_read, file)
            if not chunk:
                break
            yield chunk
        if _signature(file) != signature or file.tell() != signature[2]:
            raise ClientError(f'Import file changed during upload: {path}')


def _groups(spec, directory, *, stop):
    # Header inspection is local-file work only, loaded lazily for from.
    # list/show/run and definition-only create never import GGUF here.
    from tllama.helpers.gguf_import import local_group
    groups = {}
    for section, source in (('llm', spec.llm_from), ('mmproj', spec.mmproj_from)):
        if source is not None:
            if stop.is_set():
                return {}
            path = Path(source).expanduser()
            groups[section] = local_group(path if path.is_absolute() else directory / path)
    return groups


async def create_from_file(client, model, path, *, force=False, as_json=False, decimal_units=False):
    directory = Path.cwd()
    path = Path(path).expanduser().absolute()
    try:
        validate_definition_name(model)
        with path.open('rb') as definition_file:
            raw = definition_file.read(MAX_DEFINITION + 1)
        if len(raw) > MAX_DEFINITION:
            raise ClientError('TOML definition exceeds the 2 MiB limit.')
        text = raw.decode('utf-8')
        spec = parse_model_toml(text, source=str(path))
        # Check conflicts and existing repository references before hashing
        # or transferring potentially many gigabytes. Publication rechecks
        # the same constraints atomically; preflight reserves no model name.
        await client.check_create(model, text, force=force)
        groups = await _file_job(_groups, spec, directory) if spec.llm_from is not None or spec.mmproj_from is not None else {}
        uploads = {}
        # Validate every source and hash before uploading any file, so a
        # missing projector/shard cannot leave a partly prepared model.
        plans = {}
        for section, files in groups.items():
            plans[section] = []
            for file in files:
                async with WaitStatus('Hashing model file…', enabled=not as_json):
                    digest, signature = await _file_job(_hash, file)
                plans[section].append((file, digest, signature))
        for section, files in plans.items():
            uploads[section] = []
            for file, digest, signature in files:
                if not await client.blob_exists(digest):
                    if not as_json:
                        print(f'Uploading {file.name} ({size_text(signature[2], decimal_units=decimal_units)})',
                              file=sys.stderr, flush=True)
                    async with WaitStatus('Uploading model file…', enabled=not as_json), \
                            aclosing(_chunks(file, signature)) as content:
                        await client.upload_blob(digest, content)
                uploads[section].append({'filename': file.name, 'digest': digest})
        async with WaitStatus('Creating model definition…', enabled=not as_json):
            return await client.create(model, text, uploads=uploads, force=force)
    except (OSError, ValueError) as exc:
        raise ClientError(f'Cannot create model: {exc}') from None
