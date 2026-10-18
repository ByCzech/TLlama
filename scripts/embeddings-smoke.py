#!/usr/bin/env python3
"""Audit native text embeddings independently of TLlama's public APIs."""

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
import signal
import time

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.native_config import _plain
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError
from tllama.worker_config import WorkerLaunchSpec


MODELS = (
    'second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0',
    'Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0',
)
TEXTS = (
    'A cat sleeps on the warm windowsill.',
    'Kočka spí na teplém parapetu. Příliš žluťoučký kůň.',
    'První řádek\nDruhý řádek\tUnicode: 日本語 🐈',
    'A database transaction commits its changes.',
)


def require(condition, message):
    if not condition:
        raise WorkerError(message)


@dataclass(frozen=True)
class ProbeLaunchSpec(WorkerLaunchSpec):
    """Diagnostic mode only; production load configuration stays unchanged."""

    pooling: str = 'auto'

    def command(self, api_key_file, template_file=None):
        require(self.pooling in {'auto', 'mean', 'cls', 'last'}, 'Invalid diagnostic pooling')
        extra = () if self.pooling == 'auto' else ('--pooling', self.pooling)
        return super().command(api_key_file, template_file) + ('--embeddings',) + extra


def probe_launch(spec, args):
    values = {field: getattr(spec, field) for field in WorkerLaunchSpec.__dataclass_fields__}
    options = {key: value for key, value in spec.runtime_options.items()
               if key not in {'spec_type', 'draft_n_max'}}
    # Whole encoder sequences must fit in an ubatch. The oversized probe
    # exceeds the slot context, but remains below this physical batch limit.
    options.update(n_batch=max(512, args.ctx + 8), n_ubatch=max(512, args.ctx + 8), spec_type='none')
    values.update(runtime_options=options, n_ctx=args.ctx, type_k='f16', type_v='f16',
                  flash_attn=False, template=None, mmproj=None, mmproj_options={},
                  memory_source='native', memory_compare=False)
    return ProbeLaunchSpec(**values, pooling=args.pooling)


class ProbeWorker(NativeWorker):
    async def embeddings(self, payload):
        # Reuse the authenticated, bounded transport and the worker's slot.
        async with self._response('/v1/embeddings', payload) as response:
            return await self._json_body(response)


def inspect_vectors(payload, count, dimension=None):
    require(isinstance(payload, dict) and payload.get('object') == 'list', 'Invalid embedding envelope')
    data = payload.get('data')
    require(isinstance(data, list) and len(data) == count, 'Wrong embedding count')
    vectors, norms = [], []
    for index, row in enumerate(data):
        require(isinstance(row, dict) and type(row.get('index')) is int and row['index'] == index,
                'Missing, duplicated or unordered embedding index')
        require(row.get('object') == 'embedding', 'Invalid embedding object')
        vector = row.get('embedding')
        require(isinstance(vector, list) and vector, 'Empty or non-vector embedding')
        require(all(type(value) in (int, float) and math.isfinite(value) for value in vector),
                'Embedding contains nonfinite or nonnumeric values')
        dimension = len(vector) if dimension is None else dimension
        require(len(vector) == dimension, 'Embedding dimension changed')
        norm = math.sqrt(math.fsum(value * value for value in vector))
        require(math.isfinite(norm) and norm > 0, 'Embedding has invalid or zero norm')
        vectors.append(vector)
        norms.append(norm)
    return vectors, norms, dimension


def compare_vectors(reference, actual, args, emit, observations, label):
    require(len(reference) == len(actual) and all(len(a) == len(b) for a, b in zip(reference, actual)),
            'Cannot compare different embedding shapes')
    maximum = max(abs(a - b) for va, vb in zip(reference, actual) for a, b in zip(va, vb))
    close = all(math.isclose(a, b, abs_tol=args.atol, rel_tol=args.rtol)
                for va, vb in zip(reference, actual) for a, b in zip(va, vb))
    emit('comparison', probe=label, within_tolerance=close, max_abs_difference=maximum,
         atol=args.atol, rtol=args.rtol)
    if not close:
        observations.append(label + ':vector_difference')


async def collect(worker, texts, args, emit, observations, label, dimension=None):
    body = {'input': texts, 'encoding_format': 'float'}
    emit('request', probe=label, path='/v1/embeddings', body=body)
    started = time.perf_counter()
    payload = await worker.embeddings(body)
    emit('native-response', probe=label, payload=payload)
    items = [texts] if isinstance(texts, str) else texts
    vectors, norms, dimension = inspect_vectors(payload, len(items), dimension)
    # /tokenize and the embedding route both use add_special/parse_special.
    # Record measured tokenizer counts separately; never fabricate API usage.
    token_counts = [len(await worker.tokenize(text)) for text in items]
    usage = payload.get('usage')
    if usage is None:
        observations.append(label + ':usage_unknown')
    else:
        require(isinstance(usage, dict), 'Invalid embedding usage')
        for key in ('prompt_tokens', 'total_tokens'):
            value = usage.get(key)
            if value is None:
                observations.append(label + ':' + key + '_unknown')
            else:
                require(type(value) is int and value >= 0, 'Invalid embedding token usage')
                if value != sum(token_counts):
                    observations.append(label + ':' + key + '_differs_from_tokenizer')
    normalized = all(abs(norm - 1) <= args.norm_tolerance for norm in norms)
    if not normalized:
        observations.append(label + ':not_unit_normalized')
    emit('embedding', probe=label, count=len(vectors), dimension=dimension, norms=norms,
         normalized=normalized, usage=usage, measured_token_counts=token_counts,
         elapsed_ms=(time.perf_counter() - started) * 1000)
    return vectors, dimension


async def memory(worker, emit, observations, phase):
    await worker.refresh_memory()
    telemetry = worker.telemetry.snapshot()
    emit('native-memory', phase=phase, telemetry=telemetry)
    measurement = telemetry.get('measurement') or {}
    report = measurement.get('report') or {}
    if report.get('status') != 'known':
        observations.append(phase + ':memory_not_known')
    emit('memory', phase=phase, source=telemetry.get('source'), status=report.get('status'),
         totals=measurement.get('totals'), owners=report.get('owners'))


async def error_probe(worker, body, emit, observations, label):
    emit('request', probe=label, path='/v1/embeddings', body=body)
    try:
        payload = await worker.embeddings(body)
    except WorkerRequestError as exc:
        emit('native-error', probe=label, status=exc.http_status, payload=exc.payload)
        if not 400 <= exc.http_status < 500:
            observations.append(label + ':not_client_error')
    else:
        emit('native-response', probe=label, payload=payload)
        observations.append(label + ':unexpectedly_accepted')


async def exercise_model(args, model, emit):
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    workers, observations, ok = [], [], True

    def factory(spec):
        emit('diagnostic-overrides', configured_projector=str(spec.mmproj) if spec.mmproj else None,
             effective_projector=None, mode='text_embeddings', pooling=args.pooling,
             cache_type_k='f16', cache_type_v='f16', flash_attn=False, spec_type='none')
        worker = ProbeWorker(probe_launch(spec, args))
        workers.append(worker)
        return worker

    manager = ModelManager(config, native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    try:
        started = time.perf_counter()
        async with manager.acquire_native_model(model, num_ctx=args.ctx, keep_alive=0) as lease:
            worker, pid = lease.worker, lease.worker.pid
            emit('ready', generation=lease.generation, pid=pid, argv=worker.command,
                 fingerprint=worker.runtime_fingerprint, props=worker.props,
                 metadata=_plain(lease.definition.metadata.get('metadata_raw')),
                 model_path=str(worker.spec.model), model_size_bytes=worker.spec.model.stat().st_size,
                 load_elapsed_ms=(time.perf_counter() - started) * 1000)
            await memory(worker, emit, observations, 'after-load')
            controls = []
            dimension = None
            for index, text in enumerate(TEXTS):
                values, dimension = await collect(worker, text, args, emit, observations,
                                                  'single-' + str(index), dimension)
                controls.extend(values)
            batched, _ = await collect(worker, list(TEXTS), args, emit, observations, 'batch', dimension)
            compare_vectors(controls, batched, args, emit, observations, 'single-vs-batch')
            repeated, _ = await collect(worker, list(TEXTS), args, emit, observations, 'repeat', dimension)
            compare_vectors(batched, repeated, args, emit, observations, 'repeat')
            permutation = [2, 0, 3, 1, 0]
            reordered, _ = await collect(worker, [TEXTS[i] for i in permutation], args, emit,
                                          observations, 'reordered-duplicates', dimension)
            compare_vectors([controls[i] for i in permutation], reordered, args, emit,
                            observations, 'reordered-duplicates')
            await error_probe(worker, {'input': []}, emit, observations, 'empty-array')
            await error_probe(worker, {'input': TEXTS[0], 'encoding_format': 'invalid'},
                              emit, observations, 'invalid-format')
            # Empty text is model/tokenizer dependent at the native layer.
            try:
                await collect(worker, '', args, emit, observations, 'empty-text', dimension)
            except WorkerRequestError as exc:
                require(400 <= exc.http_status < 500, 'Empty text caused a native server error')
                emit('native-error', probe='empty-text', status=exc.http_status, payload=exc.payload)
            tokens = await worker.tokenize(TEXTS[0])
            require(tokens, 'Tokenizer returned no tokens for the control text')
            context = worker.props['default_generation_settings']['n_ctx']
            require(context <= 16384, 'Effective context is too large for the bounded limit probe')
            boundary = {'input': [tokens[-1]] * (context - 1), 'encoding_format': 'float'}
            emit('request', probe='near-context', path='/v1/embeddings', body=boundary)
            payload = await worker.embeddings(boundary)
            inspect_vectors(payload, 1, dimension)
            emit('native-response', probe='near-context', payload=payload)
            await error_probe(worker, {'input': [tokens[-1]] * (context + 1)},
                              emit, observations, 'over-context')
            recovered, _ = await collect(worker, list(TEXTS), args, emit, observations, 'after-errors', dimension)
            compare_vectors(batched, recovered, args, emit, observations, 'after-errors')
            # Non-streaming responses provide no reliable inference-start event.
            # Closing this task is transport cancellation, not proof of an early native stop.
            cancel_body = {'input': list(TEXTS) * 8, 'encoding_format': 'float'}
            emit('request', probe='cancel', path='/v1/embeddings', body=cancel_body)
            task = asyncio.create_task(worker.embeddings(cancel_body))
            try:
                done, _ = await asyncio.wait({task}, timeout=args.cancel_after_ms / 1000)
                if done:
                    payload = await task
                    emit('native-response', probe='cancel-completed', payload=payload)
                    inspect_vectors(payload, len(cancel_body['input']), dimension)
                else:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                emit('transport-cancel', pending_at_cancel=not bool(done),
                     native_early_stop_observed=None,
                     note='Review worker.log; transport closure/reuse does not prove native cancellation')
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            recovered, _ = await collect(worker, list(TEXTS), args, emit, observations, 'after-cancel', dimension)
            compare_vectors(batched, recovered, args, emit, observations, 'after-cancel')
            require(worker.pid == pid, 'Recovery did not use the same native process')
            emit('recovery', pid=pid, generation=lease.generation)
            await memory(worker, emit, observations, 'after-requests')
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            await manager.shutdown()
        finally:
            for index, worker in enumerate(workers):
                code = worker.process.returncode if worker.process else None
                removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
                clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
                ok &= clean
                emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
                (args.output / f'{args.profile}-{index}-load.log').write_text(worker.load_log, encoding='utf-8')
        require(not manager.native_snapshot(), 'Embedding diagnostic retained a generation')
    emit('model-finished', protocol_ok=ok, review_required=not ok or bool(observations),
         observations=observations, residency=manager.native_snapshot())
    return ok, observations


async def exercise(args, emit):
    emit('diagnostic-version', version=1, models=args.model, public_api='not_exercised',
         mode='text_only', native_cancellation='requires_log_review')
    ok, observations = True, []
    for index, model in enumerate(args.model):
        args.profile = f'model-{index}'
        def record(stage, **values):
            emit(stage, profile=args.profile, model=model, **values)
        passed, notes = await exercise_model(args, model, record)
        ok &= passed
        observations.extend(f'{model}:{note}' for note in notes)
    emit('finished', protocol_ok=ok, review_required=not ok or bool(observations),
         observations=observations, public_api='not_exercised')
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Registered model; repeat for sequential runs')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=256)
    parser.add_argument('--pooling', choices=['auto', 'mean', 'cls', 'last'], default='auto')
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--cancel-after-ms', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--norm-tolerance', type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.model = args.model or list(MODELS)
    if not 64 <= args.ctx <= 16384:
        parser.error('--ctx must be 64..16384')
    for name in ('load_timeout', 'request_timeout', 'cancel_after_ms', 'atol', 'rtol', 'norm_tolerance'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + ' must be finite and positive')
    args.output = args.output.expanduser().resolve()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('--output already exists; choose a new directory')
    logging.basicConfig(level=logging.DEBUG, filename=args.output / 'worker.log', encoding='utf-8')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8') as raw, \
         (args.output / 'summary.jsonl').open('w', encoding='utf-8') as summary:
        def emit(stage, **values):
            row = {'stage': stage, **values}
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            raw.flush()
            if stage not in {'request', 'native-response', 'native-memory'}:
                concise = {key: value for key, value in row.items() if key not in {'argv', 'props', 'metadata'}}
                line = json.dumps(concise, ensure_ascii=False, allow_nan=False)
                summary.write(line + '\n')
                summary.flush()
                print(line, flush=True)
        try:
            return 0 if asyncio.run(exercise(args, emit)) else 1
        except Exception as exc:
            emit('fatal', error=f'{type(exc).__name__}: {exc}')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
