"""Public text embeddings over an immutable model lease and bounded native transport."""

import asyncio
import base64
import math
import struct
import time

from fastapi import HTTPException
from starlette.requests import ClientDisconnect

from tllama.embedding_contract import embedding_contract
from tllama.native_api import public_error
from tllama.residency import _settle_owned
from tllama.worker import WorkerProtocolError


_NATIVE_EMBEDDING_JSON_BUDGET = 8 * 1024 * 1024


def _inputs(value):
    if isinstance(value, str) or isinstance(value, list) and value and type(value[0]) is int:
        return [value]
    return value or []


def _token_limit(lease, contract):
    context = lease.worker.props['default_generation_settings']['n_ctx']
    # Pinned server can_split() uses an exclusive context bound for LAST
    # pooling with memory. Props do not expose memory presence, so reserve
    # one slot for LAST conservatively, without guessing from architecture.
    # Batch/ubatch and trained-context bounds remain inclusive.
    if contract.pooling == 3:
        context -= 1
    limits = [context,
              lease.resolved.runtime['n_batch'], lease.resolved.runtime['n_ubatch']]
    trained = lease.definition.metadata.get('context_length')
    if type(trained) is int and trained > 0:
        limits.append(trained)
    # An encoder may require the whole sequence in one physical ubatch.
    # Until chunkability is exposed by native, use this conservative bound
    # for every pooled model rather than guessing from its architecture.
    return min(limits)


async def _truncate_text(worker, text, full, limit):
    raw = await worker.tokenize(text, add_special=False)
    offsets = [i for i in range(len(full) - len(raw) + 1) if full[i:i + len(raw)] == raw]
    if not raw or len(offsets) != 1:
        raise ValueError('Cannot truncate this input while preserving its tokenizer special tokens')
    start = offsets[0]
    prefix, suffix = full[:start], full[start + len(raw):]
    budget = limit - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError('Embedding token limit cannot fit text and its required special tokens')
    return prefix + raw[:budget] + suffix


async def _prepare_inputs(lease, inputs, contract, *, truncate):
    limit, prepared, total = _token_limit(lease, contract), [], 0
    for value in inputs:
        if isinstance(value, str):
            tokens = await lease.worker.tokenize(value, add_special=True, parse_special=True)
        else:
            if contract.vocab_size is None:
                raise ValueError('Token-array input requires a known GGUF vocabulary size')
            if any(token >= contract.vocab_size for token in value):
                raise ValueError('Token-array input contains an ID outside this model vocabulary')
            tokens = list(value)
        if not tokens:
            raise ValueError('Embedding input has no tokens')
        if isinstance(value, str) and contract.vocab_size is not None and any(t >= contract.vocab_size for t in tokens):
            raise WorkerProtocolError('Native tokenizer returned an ID outside its GGUF vocabulary')
        if len(tokens) > limit:
            if not truncate or not isinstance(value, str):
                raise ValueError(f'Embedding input exceeds the supported token limit ({limit}, including special tokens)')
            tokens = await _truncate_text(lease.worker, value, tokens, limit)
        total += len(tokens)
        if total > 300_000:
            raise ValueError('Embedding request exceeds the 300000-token batch limit')
        prepared.append(tokens)
    return prepared


def _vectors(payload, count, dimension, tokens, *, normalized):
    if payload.get('object') != 'list' or not isinstance(payload.get('data'), list) or len(payload['data']) != count:
        raise WorkerProtocolError('Native embedding envelope/count is invalid')
    output = [None] * count
    for row in payload['data']:
        if not isinstance(row, dict) or row.get('object') != 'embedding':
            raise WorkerProtocolError('Native embedding object is invalid')
        index, vector = row.get('index'), row.get('embedding')
        if type(index) is not int or not 0 <= index < count or output[index] is not None:
            raise WorkerProtocolError('Native embedding indices are missing, duplicated or invalid')
        if not isinstance(vector, list) or len(vector) != dimension:
            raise WorkerProtocolError('Native embedding dimension differs from its GGUF contract')
        try:
            finite = all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 3.4028234663852886e38 for v in vector)
        except OverflowError:
            finite = False
        if not finite:
            raise WorkerProtocolError('Native embedding contains invalid or nonfinite float32 values')
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0 or normalized and abs(norm - 1) > 1e-3:
            raise WorkerProtocolError('Native embedding normalization is invalid')
        output[index] = vector
    usage = payload.get('usage')
    if usage is not None:
        if (not isinstance(usage, dict)
                or any(type(usage.get(key)) is not int or usage[key] != tokens
                       for key in ('prompt_tokens', 'total_tokens'))):
            raise WorkerProtocolError('Native embedding usage differs from the submitted token count')
    return output


async def serve_embeddings(manager, request, *, api, legacy=False):
    phase, started = 'prepare', time.monotonic_ns()
    try:
        value = request.prompt if legacy else request.input
        inputs = _inputs(value)
        keep_alive = request.keep_alive if api == 'ollama' else None
        num_ctx = request.options.get('num_ctx') if api == 'ollama' else None
        if api == 'ollama' and (value is None or value == '' or value == []):
            phase = 'load'
            if manager.resolve_keep_alive(keep_alive) == 0:
                await manager.unload_native_model(request.model)
            else:
                await manager.preload_native_model(request.model, num_ctx=num_ctx, keep_alive=keep_alive)
            elapsed = time.monotonic_ns() - started
            return {'embedding': []} if legacy else {
                'model': request.model, 'embeddings': [], 'prompt_eval_count': 0,
                'load_duration': elapsed, 'total_duration': elapsed}
        phase = 'load'
        async with manager.acquire_native_model(request.model, num_ctx=num_ctx, keep_alive=keep_alive,
                                                operation='text_embedding') as lease:
            loaded = time.monotonic_ns()
            phase = 'prepare'
            contract = embedding_contract(lease.definition.metadata, lease.resolved.launch.pooling_type)
            dimensions = None if legacy else request.dimensions
            if dimensions is not None and dimensions != contract.dimension:
                raise ValueError(f'dimensions must equal this model\'s full output ({contract.dimension}); dimension reduction is not integrated')
            prepared = await _prepare_inputs(lease, inputs, contract,
                                             truncate=api == 'ollama' and not legacy and request.truncate)
            tokens = sum(map(len, prepared))
            phase = 'inference'
            # Bound each private response even for large public batches. Native
            # float32 JSON uses fewer than 32 bytes per value; leave envelope room.
            chunk_size = max(1, _NATIVE_EMBEDDING_JSON_BUDGET // (32 * contract.dimension + 512))
            vectors = []
            for offset in range(0, len(prepared), chunk_size):
                chunk = prepared[offset:offset + chunk_size]
                payload = await lease.worker.embeddings({'input': chunk, 'encoding_format': 'float',
                                                          'embd_normalize': -1 if legacy else 2})
                vectors.extend(_vectors(payload, len(chunk), contract.dimension, sum(map(len, chunk)), normalized=not legacy))
            if legacy:
                return {'embedding': vectors[0]}
            if api == 'ollama':
                return {'model': request.model, 'embeddings': vectors, 'prompt_eval_count': tokens,
                        'total_duration': time.monotonic_ns() - started, 'load_duration': loaded - started}
            if request.encoding_format == 'base64':
                vectors = [base64.b64encode(struct.pack(f'<{len(v)}f', *v)).decode('ascii') for v in vectors]
            return {'object': 'list', 'model': request.model,
                    'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
                    'usage': {'prompt_tokens': tokens, 'total_tokens': tokens}}
    except Exception as exc:
        if phase == 'load' and isinstance(exc, FileNotFoundError):
            raise HTTPException(404, str(exc)) from exc
        error = public_error(exc, api=api, phase=phase)
        raise HTTPException(error.status, error.message) from exc


async def embedding_response(http_request, manager, request, *, api, legacy=False):
    """Cancel non-streaming work on real ASGI disconnect, including slot waits."""
    async def disconnected():
        while (await http_request.receive())['type'] != 'http.disconnect':
            pass

    work = asyncio.create_task(serve_embeddings(manager, request, api=api, legacy=legacy))
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
