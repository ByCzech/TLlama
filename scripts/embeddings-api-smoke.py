#!/usr/bin/env python3
"""Compare public text embeddings and official Python SDKs with direct native calls."""

import argparse
import asyncio
import base64
from dataclasses import replace
import importlib.util
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import signal
import socket
import struct
import sys

import httpx
import tomlkit
import uvicorn
from starlette.requests import ClientDisconnect

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.native_config import _plain
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError


# Reuse only the independent 0054 measurements, never the public adapter's
# token preparation, vector validation, encoding or response implementation.
spec = importlib.util.spec_from_file_location('embeddings_reference', Path(__file__).with_name('embeddings-smoke.py'))
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
require = reference.require
TEXTS = (*reference.TEXTS[:2], 'First line\nSecond line\t"quoted" \\path',
         r'First line\nSecond line\t"quoted" \path', '日本語 🐈')
PATHS = ('/api/embed', '/v1/embeddings', '/api/embeddings')


async def wait_until(predicate, timeout):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


def load_sdks():
    # Ollama creates global clients on import. This diagnostic uses loopback
    # only; a host SOCKS proxy must not become an SDK import dependency.
    names = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        import ollama
        import openai
        return ollama, openai
    except ImportError as exc:
        raise WorkerError('This diagnostic requires the optional ollama and openai Python packages: ' + str(exc)) from exc
    finally:
        os.environ.update(saved)


def diagnostic_definition(source, ctx):
    """A reviewable temporary TOML, resolved by the ordinary production path."""
    require(source is not None and source.llm_model is not None, 'Model needs an imported registered GGUF')
    runtime = _plain(source.runtime)
    for key in ('draft_n_max', 'spm_infill'):
        runtime.pop(key, None)
    runtime.update(embedding=True, pooling_type=-1, n_ctx=ctx, n_batch=max(512, ctx),
                   n_ubatch=max(512, ctx), type_k='f16', type_v='f16', flash_attn=False, spec_type='none')
    # No projector, chat template, sampling, system prompt or FIM definition.
    return tomlkit.dumps({'llm': {'model': source.llm_model}, 'runtime': runtime})


class DiagnosticManager(ModelManager):
    def __init__(self, config, model, definition, **kwargs):
        self.probe_model, self.probe_definition = model, definition
        super().__init__(config, **kwargs)

    def _toml_path_for_reference(self, model_ref):
        # Keep weights in their original store, but make every public/native
        # admission read the same temporary definition and normal fingerprint.
        return self.probe_definition if model_ref == self.probe_model else None


def decode_vectors(payload, path, count, dimension, *, encoded=False):
    if path == '/api/embed':
        vectors = payload.get('embeddings')
    elif path == '/api/embeddings':
        vectors = [payload.get('embedding')]
    else:
        if encoded:
            payload = {**payload, 'data': [dict(row) for row in payload['data']]}
            for row in payload['data']:
                require(isinstance(row.get('embedding'), str), 'Base64 output is not a string')
                data = base64.b64decode(row['embedding'], validate=True)
                require(len(data) == 4 * dimension, 'Wrong float32 base64 byte length')
                row['embedding'] = list(struct.unpack(f'<{dimension}f', data))
        return reference.inspect_vectors(payload, count, dimension)[:2]
    require(isinstance(vectors, list), 'Public response lacks embeddings')
    envelope = {'object': 'list', 'data': [
        {'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)]}
    return reference.inspect_vectors(envelope, count, dimension)[:2]


def check_usage(payload, path, tokens):
    if path == '/api/embed':
        require(type(payload.get('prompt_eval_count')) is int and payload['prompt_eval_count'] == tokens,
                'Ollama token count differs from independently measured tokens')
        require(all(type(payload.get(k)) is int and payload[k] >= 0 for k in ('load_duration', 'total_duration'))
                and payload['load_duration'] <= payload['total_duration'], 'Invalid Ollama durations')
    elif path == '/v1/embeddings':
        usage = payload.get('usage')
        require(isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] == tokens
                for k in ('prompt_tokens', 'total_tokens')), 'OpenAI token usage differs from measured tokens')


class Audit:
    def __init__(self, args, model, worker, dimension, emit, observations):
        self.args, self.model, self.worker = args, model, worker
        self.dimension, self.emit, self.observations = dimension, emit, observations

    async def tokenize(self, text, special=True):
        tokens = await self.worker.tokenize(text, add_special=special, parse_special=True)
        self.emit('tokenization', text=text, add_special=special, tokens=tokens)
        return tokens

    async def native(self, value, rows, label, raw=False):
        body = {'input': value, 'encoding_format': 'float', 'embd_normalize': -1 if raw else 2}
        self.emit('request', source='native', probe=label, path='/v1/embeddings', body=body)
        payload = await self.worker.embeddings(body)
        self.emit('native-response', probe=label, payload=payload)
        vectors, norms, _ = reference.inspect_vectors(payload, len(rows), self.dimension)
        usage = payload.get('usage')
        if usage is None:
            self.observations.append(label + ':native_usage_unknown')
        else:
            check_usage(payload, '/v1/embeddings', sum(map(len, rows)))
        self.measure(label, norms, not raw)
        return vectors

    def measure(self, label, norms, normalized):
        valid = not normalized or all(abs(n - 1) <= self.args.norm_tolerance for n in norms)
        if not valid:
            self.observations.append(label + ':not_unit_normalized')
        self.emit('embedding', probe=label, dimension=self.dimension, count=len(norms), norms=norms,
                  normalization='l2' if normalized else 'raw', normalization_valid=valid)

    def compare(self, expected, actual, label):
        reference.compare_vectors(expected, actual, self.args, self.emit, self.observations, label)

    def inspect(self, payload, path, count, tokens, label, encoded=False):
        if path != '/api/embeddings':
            require(payload.get('model') == self.model, 'Public response changed the model name')
        else:
            require(set(payload) == {'embedding'}, 'Legacy response changed its envelope')
        check_usage(payload, path, tokens)
        vectors, norms = decode_vectors(payload, path, count, self.dimension, encoded=encoded)
        self.measure(label, norms, path != '/api/embeddings')
        return vectors

    async def public(self, client, path, value, expected, rows, label, **extra):
        body = {'model': self.model, 'prompt' if path == '/api/embeddings' else 'input': value, **extra}
        self.emit('request', source='public', probe=label, path=path, body=body)
        response = await client.post(path, json=body)
        payload = response.json()
        self.emit('public-response', probe=label, path=path, status=response.status_code, payload=payload)
        require(response.status_code == 200, f'{label}: HTTP {response.status_code}: {payload}')
        vectors = self.inspect(payload, path, len(rows), sum(map(len, rows)), label,
                               encoded=extra.get('encoding_format') == 'base64')
        self.compare(expected, vectors, label)
        return vectors

    async def error(self, client, path, value, label, **extra):
        body = {'model': self.model, 'prompt' if path == '/api/embeddings' else 'input': value, **extra}
        self.emit('request', source='public', probe=label, path=path, body=body)
        response = await client.post(path, json=body)
        payload = response.json()
        self.emit('public-error', probe=label, path=path, status=response.status_code, payload=payload)
        require(response.status_code == 400, f'{label}: expected 400, got {response.status_code}')
        error = payload.get('error')
        require(isinstance(error, dict if path == '/v1/embeddings' else str), 'Wrong API error shape')


def sdk_hooks(emit, api):
    async def request(req):
        emit('request', source='sdk', api=api, path=req.url.path, body=json.loads(req.content))
    async def response(res):
        await res.aread()
        emit('sdk-wire-response', api=api, status=res.status_code, payload=res.json())
    return {'request': [request], 'response': [response]}


async def exercise_sdks(audit, address, rows, normalized, raw, ollama, openai):
    emit, model, dimension = audit.emit, audit.model, audit.dimension
    ol = ollama.AsyncClient(host=address, trust_env=False, timeout=audit.args.request_timeout,
                            event_hooks=sdk_hooks(emit, 'ollama'))
    try:
        response = await ol.embed(model=model, input=list(TEXTS), truncate=False, dimensions=dimension)
        payload = response.model_dump(mode='json', exclude_none=True)
        emit('sdk-response', api='ollama', probe='embed', payload=payload)
        vectors = audit.inspect(payload, '/api/embed', len(rows), sum(map(len, rows)), 'sdk-ollama')
        audit.compare(normalized, vectors, 'sdk-ollama')
        response = await ol.embeddings(model=model, prompt=TEXTS[0])
        payload = response.model_dump(mode='json', exclude_none=True)
        emit('sdk-response', api='ollama', probe='legacy', payload=payload)
        # SDK models may retain optional envelope fields with default values.
        vectors, _ = decode_vectors(payload, '/api/embeddings', 1, dimension)
        audit.compare(raw, vectors, 'sdk-ollama-legacy')
        try:
            await ol.embed(model=model, input=TEXTS[0], dimensions=dimension - 1)
        except ollama.ResponseError as exc:
            require(exc.status_code == 400, 'Wrong Ollama SDK error status')
            emit('sdk-error', api='ollama', status=exc.status_code)
        else:
            raise WorkerError('Ollama SDK accepted unsupported dimensions')
    finally:
        await ol._client.aclose()
    async with openai.AsyncOpenAI(base_url=address + '/v1', api_key='local-test', max_retries=0,
            timeout=audit.args.request_timeout, http_client=openai.DefaultAsyncHttpxClient(
                trust_env=False, event_hooks=sdk_hooks(emit, 'openai'))) as oa:
        for encoding in (None, 'float', 'base64'):
            extra = {} if encoding is None else {'encoding_format': encoding}
            response = await oa.embeddings.create(model=model, input=list(TEXTS), dimensions=dimension, **extra)
            payload = response.model_dump(mode='json', exclude_none=True)
            label = 'sdk-openai-' + (encoding or 'auto')
            emit('sdk-response', api='openai', probe=label, payload=payload)
            vectors = audit.inspect(payload, '/v1/embeddings', len(rows), sum(map(len, rows)), label,
                                    encoded=encoding == 'base64')
            audit.compare(normalized, vectors, label)
        response = await oa.embeddings.create(model=model, input=rows, encoding_format='float')
        payload = response.model_dump(mode='json', exclude_none=True)
        emit('sdk-response', api='openai', probe='token-arrays', payload=payload)
        vectors = audit.inspect(payload, '/v1/embeddings', len(rows), sum(map(len, rows)), 'sdk-token-arrays')
        audit.compare(normalized, vectors, 'sdk-token-arrays')
        try:
            await oa.embeddings.create(model=model, input=TEXTS[0], dimensions=dimension - 1)
        except openai.BadRequestError as exc:
            emit('sdk-error', api='openai', status=exc.status_code)
        else:
            raise WorkerError('OpenAI SDK accepted unsupported dimensions')


async def exercise_requests(audit, client, manager, rows, normalized, raw, limit):
    for path in PATHS[:2]:
        for index, text in enumerate(TEXTS):
            await audit.public(client, path, text, [normalized[index]], [rows[index]], f'{path}:single-{index}')
        for label, order in (('batch', list(range(len(TEXTS)))), ('repeat', list(range(len(TEXTS)))),
                             ('reordered-duplicates', [2, 0, 4, 1, 0, 3])):
            await audit.public(client, path, [TEXTS[i] for i in order], [normalized[i] for i in order],
                               [rows[i] for i in order], f'{path}:{label}')
        await audit.public(client, path, list(TEXTS), normalized, rows, path + ':full-dimensions', dimensions=audit.dimension)
    await audit.public(client, '/api/embeddings', TEXTS[0], raw, rows[:1], 'legacy-raw')
    await audit.public(client, '/v1/embeddings', list(TEXTS), normalized, rows, 'base64', encoding_format='base64')
    await audit.public(client, '/v1/embeddings', rows, normalized, rows, 'token-arrays')
    await audit.public(client, '/v1/embeddings', rows[0], normalized[:1], rows[:1], 'token-array')
    empty = await audit.tokenize('')
    require(empty, 'Empty text has no native tokens for this checkpoint')
    expected = await audit.native([''], [empty], 'native-empty-item')
    await audit.public(client, '/api/embed', [''], expected, [empty], 'empty-batch-item')

    boundary = [rows[0][-1]] * limit
    expected = await audit.native([boundary], [boundary], 'native-at-limit')
    await audit.public(client, '/v1/embeddings', boundary, expected, [boundary], 'at-token-limit')
    long_text = ' diagnostic' * (limit + 8)
    full, content = await audit.tokenize(long_text), await audit.tokenize(long_text, False)
    require(len(full) > limit, 'Limit probe did not exceed context')
    offsets = [i for i in range(len(full) - len(content) + 1) if full[i:i + len(content)] == content]
    require(content and len(offsets) == 1, 'Cannot independently identify truncation affixes')
    start = offsets[0]
    prefix, suffix = full[:start], full[start + len(content):]
    truncated = prefix + content[:limit - len(prefix) - len(suffix)] + suffix
    require(len(truncated) == limit, 'Truncation oracle did not fill the token limit')
    audit.emit('truncation-reference', limit=limit, original_tokens=len(full), prefix=prefix, suffix=suffix,
               submitted_tokens=truncated)
    expected = await audit.native([truncated], [truncated], 'native-truncated')
    await audit.public(client, '/api/embed', long_text, expected, [truncated], 'truncated-default')
    for path in PATHS:
        await audit.error(client, path, long_text, path + ':over-limit', **({'truncate': False} if path == '/api/embed' else {}))
    for path in PATHS[:2]:
        await audit.error(client, path, TEXTS[0], path + ':unsupported-dimensions', dimensions=audit.dimension - 1)
    await audit.error(client, '/v1/embeddings', TEXTS[0], 'invalid-format', encoding_format='invalid')
    await audit.error(client, '/v1/embeddings', [], 'empty-openai')
    await audit.error(client, '/v1/embeddings', [-1], 'negative-token')
    await audit.error(client, '/v1/embeddings', boundary + boundary[:1], 'over-token-limit')
    # Verify an actual private 400 too, then recovery through public adapters.
    try:
        body = {'input': TEXTS[0], 'encoding_format': 'invalid'}
        audit.emit('request', source='native', probe='native-invalid-format', path='/v1/embeddings', body=body)
        await audit.worker.embeddings(body)
    except WorkerRequestError as exc:
        audit.emit('native-error', probe='native-invalid-format', status=exc.http_status, payload=exc.payload)
        require(exc.http_status == 400, 'Native invalid encoding did not report 400')
    else:
        raise WorkerError('Native unexpectedly accepted invalid encoding')
    await audit.public(client, '/api/embed', list(TEXTS), normalized, rows, 'after-errors')

    for path in PATHS:
        value = (list(TEXTS) * 7)[:32] if path != '/api/embeddings' else TEXTS[0]
        # Legacy has a single prompt. Other paths use a bounded batch; no
        # inference-start event is available for any non-streaming endpoint.
        body = {'model': audit.model, 'prompt' if path == '/api/embeddings' else 'input': value}
        audit.emit('request', source='public', probe='cancel', path=path, body=body)
        task = asyncio.create_task(client.post(path, json=body))
        try:
            done, _ = await asyncio.wait({task}, timeout=audit.args.cancel_after_ms / 1000)
            leases = manager.native_snapshot()[0]['leases']
            if done:
                response = await task
                require(response.status_code == 200, 'Cancellation request failed before closure')
                audit.emit('public-response', probe='cancel-completed', path=path, status=response.status_code,
                           payload=response.json())
            else:
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0, audit.args.request_timeout)
            audit.emit('transport-cancel', path=path, pending_at_cancel=not bool(done), leases_at_cancel=leases,
                       native_early_stop_observed=None, inference_started_observed=None,
                       note='Transport closure and same-worker reuse do not prove early native compute cancellation')
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await audit.public(client, '/api/embed', list(TEXTS), normalized, rows, path + ':after-cancel')


async def exercise_model(args, model, profile, sdks, emit):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    source = ModelManager(config)
    original_path = source._toml_path_for_reference(model)
    original = original_path.read_bytes() if original_path and original_path.is_file() else None
    text = diagnostic_definition(source._resolve_virtual_model_spec(model), args.ctx)
    definition = args.output / f'{profile}.toml'
    definition.write_text(text, encoding='utf-8')
    emit('diagnostic-definition', path=definition.name, toml=text,
         omitted_sections=['mmproj', 'template', 'sampling', 'system', 'fim'])
    # The temporary TOML overrides these fields; discard only global draft/FIM
    # companions that would otherwise make explicit embedding mode invalid.
    globals_ = {k: v for k, v in config.runtime_overrides.items() if k not in {'draft_n_max', 'spm_infill'}}
    config = replace(config, keep_alive=-1, max_loaded_models=1, memory_source='native', memory_compare=False,
                     runtime_overrides=globals_)
    workers, observations, ok = [], [], True
    def factory(launch):
        worker = NativeWorker(launch)  # No post-resolution launch mutation.
        workers.append(worker)
        return worker
    manager = DiagnosticManager(config, model, definition, native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    manager.metadata_cache_dir = args.output / f'{profile}-metadata-cache'
    previous = main.model_manager, ollama.model_manager, openai.model_manager
    main.model_manager = ollama.model_manager = openai.model_manager = manager
    async def public_app(scope, receive, send):
        try:
            await main.app(scope, receive, send)
        except ClientDisconnect:
            # Expected for the diagnostic's deliberate TCP closures.
            pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(public_app, http='h11', loop='asyncio', lifespan='off',
        log_level='warning', access_log=False, timeout_graceful_shutdown=5))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await wait_until(lambda: server.started or serving.done(), 15)
        require(server.started, 'Private public HTTP listener failed to start')
        address = f'http://127.0.0.1:{sock.getsockname()[1]}'
        async with httpx.AsyncClient(base_url=address, trust_env=False, timeout=args.request_timeout) as client:
            response = await client.post('/api/embed', json={'model': model, 'input': '', 'keep_alive': -1})
            emit('public-response', probe='preload', status=response.status_code, payload=response.json())
            require(response.status_code == 200 and response.json()['embeddings'] == []
                    and response.json()['prompt_eval_count'] == 0, 'Public preload failed')
            async with manager.acquire_native_model(model, keep_alive=-1, operation='text_embedding') as lease:
                worker, pid, generation = lease.worker, lease.worker.pid, lease.generation
                metadata = _plain(lease.definition.metadata)
                raw_metadata = metadata['metadata_raw']
                arch = raw_metadata['general.architecture']
                # Independently follow native n_embd_out(): explicit positive
                # output width, or hidden width for absent/zero metadata.
                hidden_width = raw_metadata[f'{arch}.embedding_length']
                output_width = raw_metadata.get(f'{arch}.embedding_length_out', 0)
                require(type(output_width) is int and output_width >= 0, 'Invalid native output width metadata')
                dimension = output_width or hidden_width
                pooling = lease.resolved.launch.pooling_type
                if pooling == -1:
                    pooling = raw_metadata[f'{arch}.pooling_type']
                # Independent native boundary: LAST with memory rejects
                # n_tokens >= n_ctx; mean/CLS allow the full context.
                context = worker.props['default_generation_settings']['n_ctx']
                context_reserve = 1 if pooling == 3 else 0
                limit = min(context - context_reserve, args.ctx,
                            metadata.get('context_length') or args.ctx)
                require(1 < dimension <= 65536 and 32 <= limit <= args.ctx, 'Unexpected bounded diagnostic contract')
                emit('ready', pid=pid, generation=generation, argv=worker.command,
                     fingerprint=worker.runtime_fingerprint, props=worker.props, metadata=metadata,
                     runtime=_plain(lease.resolved.runtime), sources=_plain(lease.resolved.sources),
                     model_path=str(worker.spec.model), model_size_bytes=worker.spec.model.stat().st_size,
                     dimension=dimension, hidden_width=hidden_width, token_limit=limit, pooling=pooling,
                     context_reserve=context_reserve)
            await reference.memory(worker, emit, observations, 'after-load')
            audit = Audit(args, model, worker, dimension, emit, observations)
            rows = [await audit.tokenize(text) for text in TEXTS]
            require(all(0 < len(row) <= limit for row in rows), 'A control text exceeds the model token limit')
            normalized = await audit.native(list(TEXTS), rows, 'native-text-batch')
            exact = await audit.native(rows, rows, 'native-exact-tokens')
            audit.compare(normalized, exact, 'native-text-vs-exact-tokens')
            raw = await audit.native(TEXTS[0], rows[:1], 'native-legacy-raw', raw=True)
            await exercise_requests(audit, client, manager, rows, normalized, raw, limit)
            await exercise_sdks(audit, address, rows, normalized, raw, *sdks)
            require(len(workers) == 1 and workers[0] is worker and worker.pid == pid,
                    'Public requests replaced the reference worker')
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0, args.request_timeout)
            emit('recovery', pid=pid, generation=generation, leases=0)
            await reference.memory(worker, emit, observations, 'after-requests')
            response = await client.post('/api/embed', json={'model': model, 'input': [], 'keep_alive': 0})
            emit('public-response', probe='unload', status=response.status_code, payload=response.json())
            require(response.status_code == 200 and response.json()['embeddings'] == [], 'Public unload failed')
            await asyncio.wait_for(manager.wait_native_closed(generation), args.request_timeout)
            require(not manager.native_snapshot(), 'Public unload retained residency')
            emit('unloaded', generation=generation, residency=manager.native_snapshot())
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, 15)
        finally:
            sock.close()
            await manager.shutdown()
            main.model_manager, ollama.model_manager, openai.model_manager = previous
        for index, worker in enumerate(workers):
            code = worker.process.returncode if worker.process else None
            removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
            clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
            ok &= clean
            (args.output / f'{profile}-{index}-load.log').write_text(worker.load_log, encoding='utf-8')
            emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
        unchanged = original is not None and original_path.read_bytes() == original
        ok &= unchanged and not manager.native_snapshot()
        emit('registered-definition', unchanged=unchanged)
    emit('model-finished', protocol_ok=ok, review_required=not ok or bool(observations),
         observations=observations, residency=manager.native_snapshot())
    return ok, observations


async def exercise(args, emit, sdks):
    emit('diagnostic-version', version=3, checkpoint='0061', models=args.model, public_api='tcp',
         mode='text_only', native_cancellation='early_compute_stop_not_proven',
         sdk_versions={name: version(name) for name in ('ollama', 'openai')})
    ok, observations = True, []
    for index, model in enumerate(args.model):
        def record(stage, **values):
            emit(stage, model=model, profile=f'model-{index}', **values)
        try:
            passed, notes = await exercise_model(args, model, f'model-{index}', sdks, record)
        except Exception as exc:
            passed, notes = False, [f'{type(exc).__name__}: {exc}']
            record('error', error=notes[0])
        ok &= passed
        observations.extend(f'{model}:{note}' for note in notes)
    emit('finished', protocol_ok=ok, review_required=not ok or bool(observations), observations=observations)
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Registered model; repeat for sequential runs')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=256)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--cancel-after-ms', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--norm-tolerance', type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.model = args.model or list(reference.MODELS)
    if not 64 <= args.ctx <= 16384:
        parser.error('--ctx must be 64..16384')
    for name in ('load_timeout', 'request_timeout', 'cancel_after_ms', 'atol', 'rtol', 'norm_tolerance'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + ' must be finite and positive')
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error('--output already exists; choose a new directory')
    try:
        sdks = load_sdks()
    except WorkerError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.DEBUG, filename=args.output / 'worker.log', encoding='utf-8')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8') as raw, \
         (args.output / 'summary.jsonl').open('w', encoding='utf-8') as summary:
        def emit(stage, **values):
            row = {'stage': stage, **values}
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            raw.flush()
            if stage not in {'request', 'native-response', 'public-response', 'sdk-response',
                             'sdk-wire-response', 'native-memory', 'tokenization', 'truncation-reference'}:
                concise = {k: v for k, v in row.items() if k not in {'argv', 'props', 'metadata'}}
                line = json.dumps(concise, ensure_ascii=False, allow_nan=False)
                summary.write(line + '\n')
                summary.flush()
                print(line, flush=True)
        return 0 if asyncio.run(exercise(args, emit, sdks)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
