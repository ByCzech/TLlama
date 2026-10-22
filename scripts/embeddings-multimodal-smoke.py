#!/usr/bin/env python3
"""Audit native image embeddings before integrating multimodal public APIs."""

import argparse
import asyncio
import base64
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import signal
import struct
import sys
import zlib

import tomlkit

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.embedding_contract import embedding_contract
from tllama.native_config import _plain
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError
from tllama.worker_config import WorkerLaunchSpec


MODEL = 'mradermacher/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B.Q8_0'
TEXT = 'První řádek\nDruhý řádek; literal \\n. 日本語 🐈'


def require(condition, message):
    if not condition:
        raise WorkerError(message)


@dataclass(frozen=True)
class ProbeLaunchSpec(WorkerLaunchSpec):
    """Diagnostic-only projector launch; production embedding guards stay intact."""

    def command(self, api_key_file, template_file=None, socket_file=None):
        return super().command(api_key_file, template_file, socket_file) + ('--embeddings', '--pooling', 'last')


def probe_launch(spec, args):
    require(spec.mmproj is not None, 'The registered model needs a resolved projector')
    values = {name: getattr(spec, name) for name in WorkerLaunchSpec.__dataclass_fields__}
    options = {key: value for key, value in spec.runtime_options.items()
               if key not in {'draft_n_max', 'spm_infill'}}
    options.update(spec_type='none', n_batch=args.ctx, n_ubatch=args.ctx)
    values.update(n_ctx=args.ctx, runtime_options=options, template=None,
                  embedding=False, pooling_type=-1, type_k='f16', type_v='f16',
                  flash_attn=False, memory_source='native', memory_compare=False)
    return ProbeLaunchSpec(**values)


class ProbeWorker(NativeWorker):
    async def embeddings(self, body):
        # Use the owned authentication, slot, timeout, body bound and cleanup.
        async with self._response('/v1/embeddings', {**body, 'model': self.spec.alias}) as response:
            return await self._json_body(response)


def managed_definition(source, ctx):
    """Explicit production mode without editing the registered model definition."""
    require(source is not None and source.llm_model is not None and source.mmproj_model is not None,
            'Managed diagnostic needs an imported model and registered projector')
    runtime = _plain(source.runtime)
    for key in ('draft_n_max', 'spm_infill'):
        runtime.pop(key, None)
    runtime.update(embedding=True, pooling_type=3, n_ctx=ctx, n_batch=ctx, n_ubatch=ctx,
                   type_k='f16', type_v='f16', flash_attn=False, spec_type='none')
    return tomlkit.dumps({'llm': {'model': source.llm_model}, 'runtime': runtime,
                         'mmproj': {'model': source.mmproj_model, **_plain(source.mmproj_context)}})


class ManagedProbeManager(ModelManager):
    def __init__(self, config, model, definition, **kwargs):
        self.probe_model, self.probe_definition = model, definition
        super().__init__(config, **kwargs)

    def _toml_path_for_reference(self, model_ref):
        return self.probe_definition if model_ref == self.probe_model else None


def png(swapped=False):
    width, height = 224, 112
    left, right = (b'\xff\0\0', b'\0\0\xff')
    if swapped:
        left, right = right, left

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    rows = (b'\0' + left * (width // 2) + right * (width // 2)) * height
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    return data + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


def text(value):
    return {'type': 'text', 'text': value}


def image(data):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}}


def raw_entry(entry, marker):
    """An independent pinned-native representation of the identical content."""
    prompt, media = [], []
    for part in entry['content']:
        if part['type'] == 'text':
            prompt.append(part['text'])
        else:
            prompt.append(marker)
            media.append(part['image_url']['url'].split(',', 1)[1])
    return {'prompt_string': ''.join(prompt), 'multimodal_data': media}


def vectors(payload, count, dimension, encoding, args, emit, notes, label):
    require(isinstance(payload, dict) and payload.get('object') == 'list', 'Invalid embedding envelope')
    rows = payload.get('data')
    require(isinstance(rows, list) and len(rows) == count, 'Wrong embedding count')
    result, norms = [], []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and row.get('object') == 'embedding'
                and type(row.get('index')) is int and row['index'] == index, 'Invalid embedding index/object')
        vector = row.get('embedding')
        if encoding == 'base64':
            require(isinstance(vector, str), 'Expected base64 vector')
            data = base64.b64decode(vector, validate=True)
            require(len(data) == dimension * 4, 'Wrong base64 dimension')
            vector = list(struct.unpack('<' + 'f' * dimension, data))
        require(isinstance(vector, list) and len(vector) == dimension, 'Wrong embedding dimension')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in vector), 'Invalid embedding value')
        norm = math.sqrt(math.fsum(v * v for v in vector))
        require(math.isfinite(norm) and norm > 0, 'Zero or invalid embedding norm')
        norms.append(norm)
        result.append(vector)
    if any(abs(norm - 1) > args.norm_tolerance for norm in norms):
        notes.append(label + ':not_unit_norm')
    usage = payload.get('usage')
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0
                                        for k in ('prompt_tokens', 'total_tokens')):
        notes.append(label + ':missing_or_invalid_usage')
    elif usage['prompt_tokens'] != usage['total_tokens']:
        notes.append(label + ':inconsistent_usage')
    emit('embedding', probe=label, count=count, dimension=dimension, norms=norms, usage=usage)
    return result


def compare(reference, actual, args, emit, notes, label):
    require(len(reference) == len(actual)
            and all(len(a) == len(b) for a, b in zip(reference, actual)), 'Different embedding shapes')
    pairs = [(a, b) for va, vb in zip(reference, actual) for a, b in zip(va, vb)]
    close = all(math.isclose(a, b, abs_tol=args.atol, rel_tol=args.rtol) for a, b in pairs)
    emit('comparison', probe=label, within_tolerance=close,
         max_abs_difference=max(abs(a - b) for a, b in pairs), atol=args.atol, rtol=args.rtol)
    if not close:
        notes.append(label + ':vector_difference')


async def memory(worker, emit, notes, phase):
    await worker.refresh_memory()
    telemetry = worker.telemetry.snapshot()
    emit('native-memory', phase=phase, telemetry=telemetry)
    measurement = telemetry.get('measurement') or {}
    report = measurement.get('report') or {}
    emit('memory', phase=phase, source=telemetry.get('source'), status=report.get('status'),
         totals=measurement.get('totals'), owners=report.get('owners'))
    if report.get('status') != 'known':
        notes.append(phase + ':memory_not_known')


async def exercise(args, emit):
    emit('diagnostic-version', version=3 if args.public else 2 if args.managed else 1,
         checkpoint='0064' if args.public else '0063' if args.managed else '0062', model=args.model,
         launch_path='production' if args.managed else 'independent_reference',
         public_api='requested' if args.public else 'not_exercised', mode='native_multimodal_embeddings', pooling='last',
         prompt_policy='literal_content_no_implicit_template_or_instruction',
         native_cancellation='early_compute_stop_not_proven')
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    workers, notes, ok = [], [], True

    def factory(spec):
        if args.managed:
            # The resolved production launch and transport are used unchanged.
            worker = NativeWorker(spec)
            workers.append(worker)
            return worker
        launch = probe_launch(spec, args)
        emit('diagnostic-overrides', configured_context=spec.n_ctx, effective_context=launch.n_ctx,
             projector=str(launch.mmproj), projector_options=_plain(launch.mmproj_options),
             pooling='last', runtime=_plain(launch.runtime_options), cache_type='f16', flash_attn=False)
        worker = ProbeWorker(launch)
        workers.append(worker)
        return worker

    source = ModelManager(config)
    definition = source._toml_path_for_reference(args.model)
    before = definition.read_bytes() if definition is not None and definition.is_file() else None
    options = dict(native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    if args.managed:
        probe = args.output / 'managed-model.toml'
        contents = managed_definition(source._resolve_virtual_model_spec(args.model), args.ctx)
        probe.write_text(contents, encoding='utf-8')
        emit('managed-definition', path=probe.name, toml=contents,
             omitted_sections=['template', 'sampling', 'system', 'fim'])
        globals_ = {k: v for k, v in config.runtime_overrides.items() if k not in {'draft_n_max', 'spm_infill'}}
        config = replace(config, runtime_overrides=globals_, memory_source='native', memory_compare=False)
        manager = ManagedProbeManager(config, args.model, probe, **options)
    else:
        manager = ModelManager(config, **options)
    manager.metadata_cache_dir = args.output / 'metadata-cache'
    try:
        require(before is not None, 'Registered model definition not found')
        emit('registered-definition', phase='before', path=str(definition), toml=before.decode('utf-8'),
             sha256=hashlib.sha256(before).hexdigest())
        async with manager.acquire_native_model(args.model, num_ctx=None if args.managed else args.ctx, keep_alive=-1,
                                                operation='text_embedding' if args.managed else None) as lease:
            worker, pid, generation = lease.worker, lease.worker.pid, lease.generation
            contract = embedding_contract(lease.definition.metadata, 3)
            require((worker.props.get('modalities') or {}).get('vision') is True,
                    'Loaded native worker does not advertise vision')
            marker = worker.props.get('media_marker')
            require(isinstance(marker, str) and marker, 'Native props lack the media marker')
            emit('ready', pid=pid, generation=generation, argv=worker.command, props=worker.props,
                 fingerprint=worker.runtime_fingerprint, metadata=_plain(lease.definition.metadata),
                 dimension=contract.dimension, model_path=str(worker.spec.model),
                 model_size_bytes=worker.spec.model.stat().st_size,
                 projector_path=str(worker.spec.mmproj), projector_size_bytes=worker.spec.mmproj.stat().st_size)
            await memory(worker, emit, notes, 'after-load')
            images = []
            for index in range(2):
                data = png(bool(index))
                name = f'image-{index}.png'
                (args.output / name).write_bytes(data)
                emit('fixture', path=name, sha256=hashlib.sha256(data).hexdigest(), width=224, height=112)
                images.append(base64.b64encode(data).decode())
            entries = [
                {'content': [text(TEXT)]},
                {'content': [image(images[0])]},
                {'content': [image(images[1])]},
                {'content': [text(TEXT), image(images[0]), text('\nEnd.')]},
                {'content': [text(TEXT), image(images[1]), text('\nEnd.')]},
                {'content': [image(images[0]), text('\nBetween.\n'), image(images[1]), text('\nEnd.')]},
                {'content': [image(images[1]), text('\nBetween.\n'), image(images[0]), text('\nEnd.')]},
            ]

            async def collect(value, count, label, encoding='float'):
                body = {'input': value, 'encoding_format': encoding}
                emit('request', probe=label, path='/v1/embeddings', body=body)
                payload = await worker.embeddings(body)
                emit('native-response', probe=label, payload=payload)
                return vectors(payload, count, contract.dimension, encoding, args, emit, notes, label)

            controls = []
            for index, entry in enumerate(entries):
                single = await collect(entry, 1, f'typed-single-{index}')
                controls.extend(single)
                other = await collect(raw_entry(entry, marker), 1, f'raw-single-{index}')
                compare(single, other, args, emit, notes, f'typed-vs-raw-{index}')
            plain = await collect(TEXT, 1, 'plain-text')
            compare(controls[:1], plain, args, emit, notes, 'typed-vs-plain-text')
            tokens = await worker.tokenize(TEXT)
            emit('tokenization', probe='plain-text', tokens=tokens)
            exact = await collect(tokens, 1, 'exact-text-tokens')
            compare(plain, exact, args, emit, notes, 'plain-vs-exact-text-tokens')
            batched = await collect(entries, len(entries), 'typed-batch')
            compare(controls, batched, args, emit, notes, 'single-vs-batch')
            raw_batch = await collect([raw_entry(entry, marker) for entry in entries], len(entries), 'raw-batch')
            compare(batched, raw_batch, args, emit, notes, 'typed-vs-raw-batch')
            repeated = await collect(entries, len(entries), 'typed-repeat')
            compare(batched, repeated, args, emit, notes, 'repeat')
            order = [6, 2, 0, 3, 1, 5, 4, 1]
            reordered = await collect([entries[i] for i in order], len(order), 'reordered-duplicates')
            compare([controls[i] for i in order], reordered, args, emit, notes, 'reordered-duplicates')
            encoded = await collect(entries, len(entries), 'base64', 'base64')
            compare(controls, encoded, args, emit, notes, 'float-vs-base64')
            for left, right, label in ((1, 2, 'changed-image'), (3, 4, 'changed-image-with-same-text'),
                                       (5, 6, 'swapped-image-order')):
                difference = max(abs(a - b) for a, b in zip(controls[left], controls[right]))
                distinct = not all(math.isclose(a, b, abs_tol=args.atol, rel_tol=args.rtol)
                                   for a, b in zip(controls[left], controls[right]))
                emit('image-sensitivity', probe=label, distinct=distinct, max_abs_difference=difference)
                if not distinct:
                    notes.append(label + ':no_measurable_image_effect')
            for label, body in (
                ('empty-array', {'input': []}),
                ('invalid-format', {'input': entries[1], 'encoding_format': 'invalid'}),
                ('unsupported-content', {'input': {'content': [{'type': 'unsupported'}]}}),
                ('invalid-image', {'input': {'content': [image(base64.b64encode(b'not a PNG').decode())]}}),
            ):
                emit('request', probe=label, path='/v1/embeddings', body=body)
                try:
                    payload = await worker.embeddings(body)
                except WorkerRequestError as exc:
                    emit('native-error', probe=label, status=exc.http_status, payload=exc.payload)
                    error = exc.payload.get('error') if isinstance(exc.payload, dict) else None
                    decoder_error = (label == 'invalid-image' and exc.http_status == 500
                                     and isinstance(error, dict)
                                     and error.get('message') == 'Failed to load image or audio file')
                    if decoder_error:
                        emit('known-native-limitation', probe=label, status=500,
                             reason='Pinned media decoder throws runtime_error for invalid image bytes')
                    elif not 400 <= exc.http_status < 500:
                        notes.append(label + ':not_client_error')
                else:
                    emit('native-response', probe=label, payload=payload)
                    notes.append(label + ':unexpectedly_accepted')
            recovered = await collect(entries, len(entries), 'after-errors')
            compare(controls, recovered, args, emit, notes, 'after-errors')
            cancel_body = {'input': [entries[3]] * 8, 'encoding_format': 'float'}
            emit('request', probe='cancel', path='/v1/embeddings', body=cancel_body)
            task = asyncio.create_task(worker.embeddings(cancel_body))
            try:
                done, _ = await asyncio.wait({task}, timeout=args.cancel_after_ms / 1000)
                if done:
                    payload = await task
                    emit('native-response', probe='cancel-completed', payload=payload)
                    vectors(payload, 8, contract.dimension, 'float', args, emit, notes, 'cancel-completed')
                else:
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                emit('transport-cancel', pending_at_cancel=not bool(done),
                     native_early_stop_observed=None, inference_started_observed=None)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            recovered = await collect(entries, len(entries), 'after-cancel')
            compare(controls, recovered, args, emit, notes, 'after-cancel')
            require(worker.pid == pid, 'Recovery changed the native process')
            emit('recovery', pid=pid, generation=generation)
            await memory(worker, emit, notes, 'after-requests')
            if args.public:
                spec = importlib.util.spec_from_file_location('embeddings_media_public',
                    Path(__file__).with_name('embeddings-media-public.py'))
                public = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = public
                spec.loader.exec_module(public)
                await public.exercise(args, manager, worker, entries, controls, contract.dimension,
                                      emit, notes, sys.modules[__name__])
                require(worker.pid == pid and len(workers) == 1, 'Public requests changed the reference worker')
                await memory(worker, emit, notes, 'after-public-requests')
        await manager.unload_native_model(args.model)
        await manager.wait_native_closed(generation)
        require(not manager.native_snapshot(), 'Unload retained a generation')
        emit('unloaded', generation=generation, residency=manager.native_snapshot())
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        await manager.shutdown()
        for index, worker in enumerate(workers):
            code = worker.process.returncode if worker.process else None
            removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
            clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
            ok &= clean
            emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
            (args.output / f'worker-{index}-load.log').write_text(worker.load_log, encoding='utf-8')
        unchanged = before is not None and definition.is_file() and definition.read_bytes() == before
        ok &= unchanged
        emit('registered-definition', phase='after', unchanged=unchanged)
        ok &= not bool(manager.native_snapshot())
    emit('finished', protocol_ok=ok, review_required=not ok or bool(notes), observations=notes,
         residency=manager.native_snapshot(), public_api='exercised' if args.public and ok else
         'incomplete' if args.public else 'not_exercised')
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--managed', action='store_true', help='Use production resolution and NativeWorker unchanged')
    parser.add_argument('--public', action='store_true', help='Add public TCP/SDK comparison; implies --managed')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=1024)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--cancel-after-ms', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--norm-tolerance', type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.managed |= args.public
    if not 512 <= args.ctx <= 4096:
        parser.error('--ctx must be 512..4096')
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
                concise = {key: value for key, value in row.items()
                           if key not in {'argv', 'props', 'metadata', 'toml', 'tokens'}}
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
