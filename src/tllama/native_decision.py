"""Public text Decision over validated transport and an owned model lease."""

import asyncio
import json

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from tllama.decision_contract import MAX_REQUEST_BYTES
from tllama.native_api import public_error
from tllama.residency import _settle_owned
from tllama.schemas.decision import DecisionRequest


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Decision JSON contains duplicate object keys')
        value[key] = item
    return value


def _finite_constant(value):
    raise ValueError('Decision JSON numbers must be finite')


async def read_decision_request(http_request):
    """Bound wire bytes before decoding; do not silently collapse duplicate IDs."""
    media = http_request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    if media != 'application/json' and not (media.startswith('application/') and media.endswith('+json')):
        raise HTTPException(415, 'Decision requests require application/json')
    data = bytearray()
    async for chunk in http_request.stream():
        if len(data) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(413, 'Decision request exceeds the 2 MiB JSON body limit')
        data.extend(chunk)
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object, parse_constant=_finite_constant)
        return DecisionRequest.model_validate(value)
    except ValidationError as exc:
        message = '; '.join(error['msg'] for error in exc.errors(include_input=False, include_url=False))
        raise HTTPException(400, message) from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(400, 'Invalid Decision JSON: ' + str(exc)) from exc


async def serve_decision(manager, request):
    phase = 'load'
    try:
        async with manager.acquire_native_model(request.model, operation='decision') as lease:
            phase = 'inference'
            payload = await lease.worker.decision(request.model_dump())
            # Native validates its private alias; only the admitted public name
            # replaces it. Do not expose unvalidated extra native fields.
            return {'model': lease.definition.name, 'answers': payload['answers'], 'usage': payload['usage']}
    except Exception as exc:
        error = public_error(exc, api='openai', phase=phase)
        raise HTTPException(error.status, error.message) from exc


async def decision_response(http_request, manager):
    request = await read_decision_request(http_request)

    async def disconnected():
        while (await http_request.receive())['type'] != 'http.disconnect':
            pass

    work = asyncio.create_task(serve_decision(manager, request))
    watcher = asyncio.create_task(disconnected())

    async def cleanup():
        for task in (work, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)

    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        await watcher
        raise ClientDisconnect()
    finally:
        await _settle_owned(asyncio.create_task(cleanup()))
