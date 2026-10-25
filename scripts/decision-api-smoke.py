#!/usr/bin/env python3
"""Compare independent native, managed and public text Decision responses."""

import argparse
import asyncio
import base64
from dataclasses import replace
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import signal
import socket
import sys

import httpx
import tomlkit
import uvicorn
from starlette.requests import ClientDisconnect

from tllama.backend import ModelManager
from tllama.daemon_config import load_daemon_config
from tllama.native_config import _plain
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError


# The oracle uses the accepted independent 0065/0066 measurements, never the
# production request preparation or typed response validator.
spec = importlib.util.spec_from_file_location('decision_reference_0069', Path(__file__).with_name('decision-smoke.py'))
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
require = reference.require
PATH = '/v1/systemone'
MODELS = (reference.MODEL, reference.MODELS[1], reference.MODELS[2])


async def wait_until(predicate, timeout):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


def diagnostic_definition(source, ctx):
    """Select the accepted native recipe in a separate reviewable definition."""
    require(source is not None and source.llm_model is not None, 'Registered imported GGUF required')
    runtime = _plain(source.runtime)
    for key in ('draft_n_max', 'spm_infill'):
        runtime.pop(key, None)
    runtime.update(n_ctx=ctx, n_batch=ctx, n_ubatch=ctx, type_k='f16', type_v='f16',
                   flash_attn=False, spec_type='none', embedding=False, pooling_type=-1)
    return tomlkit.dumps({'llm': {'model': source.llm_model}, 'runtime': runtime})


class DiagnosticManager(ModelManager):
    def __init__(self, config, model, definition, **kwargs):
        self.probe_model, self.probe_definition = model, definition
        super().__init__(config, **kwargs)

    def _toml_path_for_reference(self, model_ref):
        return self.probe_definition if model_ref == self.probe_model else None


def stable_command(argv):
    """Ignore owned listener and credential locations when comparing launches."""
    result, skip = [], False
    for value in argv:
        if skip:
            skip = False
        elif value in ('--api-key-file', '--port', '--host'):
            skip = True
        else:
            result.append(value)
    return result


def cases(kind, item):
    state, questions = reference.STATE, reference.QUESTIONS
    controls = [('single-' + key, state, {key: q}) for key, q in questions.items()]
    controls += [('combined', state, questions), ('repeat', state, questions),
                 ('reversed-questions', state, dict(reversed(list(questions.items()))))]
    if kind != 'laya':
        controls.append(('reversed-questions-repeat', state, dict(reversed(list(questions.items())))))
    structured = {'ticket': state, 'nested': {'plan': 'pro', 'attempts': [1, 2]}}
    special = {'řádek\nliteral\\n 🐈': {'type': 'choice', 'instructions': 'Which team?', 'criteria': {
        'platba\nbilling': 'payments', 'literal\\nshipping': 'parcels', '日本語': 'software'}}}
    controls += [('json-state', structured, questions), ('json-state-repeat', structured, questions),
                 ('literal-keys-unicode', 'První řádek\nDruhý; literal \\n. 日本語 🐈 ' + state, special),
                 ('one-option', state, {'one': {'type': 'choice', 'instructions': 'Choose the only option', 'criteria': {'only': None}}}),
                 ('ten-level-score', state, {'scale': {'type': 'score', 'instructions': 'How urgent?', 'criteria': [str(i) for i in range(10)]}})]
    if kind != 'laya':
        reverse = {'route': {**questions['route'], 'criteria': dict(reversed(list(questions['route']['criteria'].items())))}}
        controls += [('reversed-options', state, reverse), ('reversed-options-repeat', state, reverse),
                     ('twenty-seven-options', state, {'many': {'type': 'choice', 'instructions': 'Choose a category.',
                         'criteria': {f'category-{i:02}': None for i in reversed(range(27))}}}),
                     ('structured-instructions', {'ticket': state}, {'typed': {'type': 'noul',
                         'instructions': {'question': 'Is the customer angry?', 'tags': ['support', 1]},
                         'criteria': {'true': {'meaning': 'angry'}, 'false': ['calm', 'happy']}}})]
    controls.append(('changed-state', 'Customer: I am happy with my purchase. My parcel arrived on time.', questions))
    if kind == 'laya':
        head = item['metadata']['modern-bert.decision.max_head_tokens']
        require(head <= 4096, 'Head budget exceeds this bounded reference probe')
        controls.append(('native-laya-head-truncation', state,
                         {'long': {**questions['route'], 'instructions': 'Please decide. ' * (head * 2)}}))
    return [(label, {'state': s, 'questions': q}) for label, s, q in controls]


def validate(payload, body, kind, model):
    require(isinstance(payload, dict) and set(payload) == {'model', 'answers', 'usage'}
            and payload['model'] == model, 'Decision envelope/model differs')
    reference.inspect(payload, body['questions'], kind)
    return {'answers': payload['answers'], 'usage': payload['usage']}


def compare(a, b, args, emit, notes, label):
    # Usage is an exact integer contract, irrespective of numerical tolerances.
    require(a['usage'] == b['usage'], label + ': native usage differs')
    reference.compare(a, b, args, emit, notes, label)


def reference_observations(results, kind, args, emit, notes):
    """Keep native composition effects distinct from identical-request checks."""
    def check(a, b, label, required=True):
        def record(stage, **values):
            emit('reference-comparison' if stage == 'comparison' else stage, **values)
        reference.compare(a, b, args, record, notes, label, required=required)
    for key in reference.QUESTIONS:
        check(results['single-' + key]['answers'][key], results['combined']['answers'][key],
              'single-vs-combined-' + key, required=kind == 'laya')
    singles = sum(results['single-' + key]['usage']['input_tokens'] for key in reference.QUESTIONS)
    combined = results['combined']['usage']['input_tokens']
    emit('usage-composition', single_input_tokens=singles, combined_input_tokens=combined,
         additive_required=kind == 'laya')
    require(kind != 'laya' or singles == combined, 'Laya independent question usage is not additive')
    for label in ('repeat', 'json-state-repeat', 'after-errors', 'after-cancel'):
        initial = 'json-state' if label == 'json-state-repeat' else 'combined'
        compare(results[initial], results[label], args,
                lambda stage, **values: emit('reference-comparison', **values), notes, label)
    check(results['combined'], results['reversed-questions'], 'reversed-questions', required=kind == 'laya')
    if kind != 'laya':
        check(results['reversed-questions'], results['reversed-questions-repeat'], 'reversed-questions-repeat')
        check(results['single-route'], results['reversed-options'], 'reversed-options', required=kind == 'clef')
        check(results['reversed-options'], results['reversed-options-repeat'], 'reversed-options-repeat')
    effect = max(abs(results['combined']['answers']['route']['probabilities'][k]
                     - results['changed-state']['answers']['route']['probabilities'][k])
                 for k in reference.QUESTIONS['route']['criteria'])
    emit('state-sensitivity', max_probability_difference=effect, distinct=effect > args.atol)
    if effect <= args.atol:
        notes.append('changed_state:no_measurable_effect')


async def expect_public_error(client, emit, label, status, *, body=None, content=None, path=PATH):
    emit('request', source='public', probe=label, path=path, body=body, content=content)
    response = await client.post(path, **({'json': body} if content is None else {
        'content': content, 'headers': {'Content-Type': 'application/json'}}))
    payload = response.json()
    emit('public-error', probe=label, status=response.status_code, payload=payload)
    require(response.status_code == status and isinstance(payload, dict) and 'error' in payload
            and (path != PATH or isinstance(payload['error'], dict) and isinstance(payload['error'].get('message'), str)),
            label + ': unexpected public error/status')


async def public_errors(client, model, emit):
    valid = {'model': model, 'state': reference.STATE, 'questions': reference.QUESTIONS}
    for label, extra in [('stream', {'stream': True}), ('sampling', {'temperature': 0.5}),
                         ('options', {'options': {}}), ('remote-image', {'images': ['https://example.invalid/a.png']}),
                         ('inline-image', {'images': ['data:image/png;base64,' + base64.b64encode(b'image').decode()]})]:
        await expect_public_error(client, emit, label, 400, body={**valid, **extra})
    for wrapped in (False, True):
        for kind in ('image_url', 'input_audio'):
            messages = [{'role': 'user', 'content': [{'type': kind, kind: {'url': 'https://example.invalid/a'}}]}]
            await expect_public_error(client, emit, f'chat-media:{wrapped}:{kind}', 400,
                                      body={**valid, 'state': {'messages': messages} if wrapped else messages})
    for label, body in [('null-state', {**valid, 'state': None}), ('empty-questions', {**valid, 'questions': {}}),
                        ('unknown-type', {**valid, 'questions': {'q': {'type': 'invalid', 'instructions': 'x'}}})]:
        await expect_public_error(client, emit, label, 400, body=body)
    for label, content in [('duplicate-key', '{"model":"a","model":"b"}'),
                           ('nonfinite', '{"state":NaN}'), ('invalid-json', '{')]:
        await expect_public_error(client, emit, label, 400, content=content)
    await expect_public_error(client, emit, 'missing-model', 404, body={**valid, 'model': 'not-a-registered-decision-model'})


async def native_error(worker, emit, label, body, status):
    emit('request', source='raw-native', probe=label, path=PATH, body=body)
    try:
        await reference.ProbeWorker.systemone(worker, body)
    except WorkerRequestError as exc:
        emit('native-error', probe=label, status=exc.http_status, payload=exc.payload)
        require(exc.http_status == status, label + ': unexpected native status')
    else:
        raise WorkerError(label + ': native accepted invalid/unsupported input')


async def cancel_request(call, manager, args, emit, kind, source, validate_completed):
    body = {'state': reference.STATE, 'questions': {f'q{i}': reference.QUESTIONS['route']
                                                  for i in range(32 if kind == 'laya' else 2)}}
    emit('request', source=source, probe='cancel', path=PATH, body=body)
    task = asyncio.create_task(call(body))
    try:
        done, _ = await asyncio.wait({task}, timeout=args.cancel_after_ms / 1000)
        leases = sum(row['leases'] for row in manager.native_snapshot())
        if done:
            validate_completed(await task, body)
        else:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await wait_until(lambda: all(row['leases'] == 0 for row in manager.native_snapshot()), args.request_timeout)
        emit('transport-cancel', source=source, pending_at_cancel=not bool(done), leases_at_cancel=leases,
             inference_started_observed=None, native_early_stop_observed=None)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def phase(args, config, model, definition, item, controls, emit, notes, *, native, expected=None, argv=None):
    from tllama import main
    from tllama.routers import ollama, openai
    workers, results, ok, command = [], {}, True, None
    kind = item['decision_type']
    def factory(launch):
        worker = reference.ProbeWorker(reference.probe_launch(launch, args)) if native else NativeWorker(launch)
        workers.append(worker)
        return worker
    manager = DiagnosticManager(config, model, definition, native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    manager.metadata_cache_dir = args.output / ('reference-cache' if native else 'managed-cache')
    previous = main.model_manager, ollama.model_manager, openai.model_manager
    sock, server, serving = None, None, None
    try:
        if not native:
            main.model_manager = ollama.model_manager = openai.model_manager = manager
            async def public_app(scope, receive, send):
                try:
                    await main.app(scope, receive, send)
                except ClientDisconnect:
                    pass
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))
            sock.listen(128)
            server = uvicorn.Server(uvicorn.Config(public_app, http='h11', ws='none', loop='asyncio', lifespan='off',
                access_log=False, log_level='warning', timeout_graceful_shutdown=5))
            serving = asyncio.create_task(server.serve(sockets=[sock]))
            await wait_until(lambda: server.started or serving.done(), 15)
            require(server.started, 'Private public listener did not start')
        address = f'http://127.0.0.1:{sock.getsockname()[1]}' if sock else 'http://127.0.0.1'
        async with httpx.AsyncClient(base_url=address, trust_env=False, timeout=args.request_timeout) as client:
            if not native:
                shown = await client.post('/api/show', json={'model': model})
                emit('public-response', probe='show', status=shown.status_code, payload=shown.json())
                require(shown.status_code == 200 and shown.json().get('capabilities') == ['decision'], 'Wrong Decision capability')
                await public_errors(client, model, emit)
                require(not workers and not manager.native_snapshot(), 'Invalid requests admitted a worker')
                emit('pre-admission', workers=0, residency=manager.native_snapshot())
                # Exercise public cold admission too, before taking any private
                # lease. All later calls must reuse this production worker.
                body = {'model': model, 'state': reference.STATE, 'questions': reference.QUESTIONS}
                emit('request', source='public', probe='cold-load', path=PATH, body=body)
                response = await client.post(PATH, json=body)
                emit('public-response', probe='cold-load', status=response.status_code, payload=response.json())
                require(response.status_code == 200, 'Public cold Decision load failed')
                value = validate(response.json(), body, kind, model)
                compare(expected['combined'], value, args,
                        lambda stage, **values: emit('cold-start-comparison', **values), notes, 'native-vs-public:cold-load')
            async with manager.acquire_native_model(model, keep_alive=-1, operation='decision') as lease:
                worker, pid, generation = lease.worker, lease.worker.pid, lease.generation
                require(worker.runtime_fingerprint == 'b11552-3144c167ef', 'Audited native b11552-3144c167ef required')
                models = await reference.ProbeWorker.models(worker)
                require(isinstance(models, dict) and len(models.get('data', [])) == 1
                        and models['data'][0].get('architecture', {}).get('output_modalities') == ['decisions'],
                        'Native Decision modality missing')
                command = stable_command(worker.command)
                if argv is not None:
                    require(command == argv, 'Managed native launch differs from independent reference recipe')
                emit('ready', pid=pid, generation=generation, argv=worker.command, props=worker.props, models=models,
                     fingerprint=worker.runtime_fingerprint, runtime=_plain(lease.resolved.runtime),
                     metadata=_plain(lease.definition.metadata), launch_path='independent' if native else 'production')
            await reference.memory(worker, emit, notes, 'after-load')

            async def direct(body):
                async with manager.acquire_native_model(model, operation='decision') as owned:
                    require(owned.worker is worker and owned.generation == generation, 'Managed worker changed')
                    if native:
                        return await worker.systemone(body)
                    return await worker.decision({'model': model, **body})

            async def public(body):
                response = await client.post(PATH, json={'model': model, **body})
                payload = response.json()
                require(response.status_code == 200, f'Public Decision failed: {response.status_code}: {payload}')
                return payload

            async def collect(label, body):
                emit('request', source='raw-native' if native else 'managed', probe=label, path=PATH, body=body)
                payload = await direct(body)
                emit('native-response' if native else 'managed-response', probe=label, payload=payload)
                value = validate(payload, body, kind, worker.spec.alias)
                if native:
                    results[label] = value
                else:
                    compare(expected[label], value, args, emit, notes, 'native-vs-managed:' + label)
                    emit('request', source='public', probe=label, path=PATH, body={'model': model, **body})
                    payload = await public(body)
                    emit('public-response', probe=label, status=200, payload=payload)
                    public_value = validate(payload, body, kind, model)
                    compare(expected[label], public_value, args, emit, notes, 'native-vs-public:' + label)
                    compare(value, public_value, args, emit, notes, 'managed-vs-public:' + label)

            for label, body in controls:
                await collect(label, body)
            await native_error(worker, emit, 'native-missing-state', {'questions': reference.QUESTIONS}, 400)
            if native:
                await native_error(worker, emit, 'native-unsupported-image', {'state': reference.STATE,
                    'questions': reference.QUESTIONS, 'images': ['data:image/png;base64,' + base64.b64encode(b'image').decode()]}, 501)
            else:
                await expect_public_error(client, emit, 'generation-mode', 400, path='/api/generate',
                    body={'model': model, 'prompt': 'hello', 'stream': False})
                await expect_public_error(client, emit, 'embedding-mode', 400, path='/v1/embeddings', body={'model': model, 'input': 'hello'})
            baseline = {'state': reference.STATE, 'questions': reference.QUESTIONS}
            await collect('after-errors', baseline)
            def completed(payload, body, source):
                emit(source + '-response', probe='cancel-completed', payload=payload)
                validate(payload, body, kind, model if source == 'public' else worker.spec.alias)
            await cancel_request(direct, manager, args, emit, kind, 'raw-native' if native else 'managed',
                                 lambda p, b: completed(p, b, 'native' if native else 'managed'))
            if not native:
                await cancel_request(public, manager, args, emit, kind, 'public', lambda p, b: completed(p, b, 'public'))
            await collect('after-cancel', baseline)
            if native:
                reference_observations(results, kind, args, emit, notes)
            require(len(workers) == 1 and worker.pid == pid and manager.native_snapshot()[0]['generation'] == generation,
                    'Recovery replaced the worker')
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0, args.request_timeout)
            emit('recovery', pid=pid, generation=generation, leases=0)
            await reference.memory(worker, emit, notes, 'after-requests')
            if native:
                await manager.unload_native_model(model)
            else:
                response = await client.post('/api/generate', json={'model': model, 'keep_alive': 0})
                emit('public-response', probe='unload', status=response.status_code, payload=response.json())
                require(response.status_code == 200 and response.json().get('done_reason') == 'unload', 'Public unload failed')
            await asyncio.wait_for(manager.wait_native_closed(generation), args.request_timeout)
            require(not manager.native_snapshot(), 'Unload retained residency')
            emit('unloaded', generation=generation, residency=manager.native_snapshot())
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            if server:
                server.should_exit = True
                await asyncio.wait_for(serving, 15)
        finally:
            if sock:
                sock.close()
            await manager.shutdown()
            main.model_manager, ollama.model_manager, openai.model_manager = previous
        for index, worker in enumerate(workers):
            code = worker.process.returncode if worker.process else None
            removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
            clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
            ok &= clean
            (args.output / f'{"reference" if native else "managed"}-{index}-load.log').write_text(worker.load_log, encoding='utf-8')
            emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
    emit('phase-finished', protocol_ok=ok, residency=manager.native_snapshot())
    return ok, results, command


async def exercise_model(args, model, emit):
    config = load_daemon_config(False, discover=False).backend
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    source = ModelManager(config)
    source.metadata_cache_dir = args.output / 'inventory-cache'
    original, original_path, item, notes, ok = None, None, None, [], True
    try:
        item = await asyncio.to_thread(reference.inventory, source, model)
        emit('inventory-detail', **item)
        contract = reference.reference_contract(item)
        emit('decision-contract', **contract)
        original_path = source._toml_path_for_reference(model)
        original = original_path.read_bytes()
        emit('registered-definition', phase='before', path=str(original_path), toml=original.decode('utf-8'),
             sha256=hashlib.sha256(original).hexdigest())
        ctx = args.ctx or (1024 if item['decision_type'] == 'laya' else 2048)
        args = replace_args(args, ctx=ctx)
        text = diagnostic_definition(source._resolve_virtual_model_spec(model), ctx)
        definition = args.output / 'managed-model.toml'
        definition.write_text(text, encoding='utf-8')
        emit('diagnostic-definition', path=str(definition), toml=text, context=ctx,
             omitted_sections=['mmproj', 'template', 'sampling', 'system', 'fim'])
        overrides = {k: v for k, v in config.runtime_overrides.items() if k not in {'draft_n_max', 'spm_infill'}}
        config = replace(config, keep_alive=-1, max_loaded_models=1, memory_source='native', memory_compare=False,
                         runtime_overrides=overrides)
        controls = cases(item['decision_type'], item)
        def scoped(name):
            def record(stage, **values):
                if 'phase' in values:
                    values['memory_phase'] = values.pop('phase')
                emit(stage, phase=name, **values)
            return record
        ok, expected, argv = await phase(args, config, model, definition, item, controls, scoped('independent'), notes, native=True)
        if ok:
            ok, _, _ = await phase(args, config, model, definition, item, controls, scoped('managed-public'), notes,
                                   native=False, expected=expected, argv=argv)
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        await source.shutdown()
        if original is not None:
            unchanged = original_path.is_file() and original_path.read_bytes() == original
            ok &= unchanged
            emit('registered-definition', phase='after', unchanged=unchanged)
        if item:
            path = Path(item['path'])
            stat = path.stat() if path.is_file() else None
            unchanged = stat is not None and stat.st_size == item['size_bytes'] and stat.st_mtime_ns == item['mtime_ns']
            ok &= unchanged
            emit('gguf-fingerprint', unchanged=unchanged)
    emit('model-finished', protocol_ok=ok, review_required=not ok or bool(notes), observations=notes)
    return ok, notes


def replace_args(args, **values):
    return argparse.Namespace(**{**vars(args), **values})


async def exercise(args, emit):
    emit('diagnostic-version', version=1, checkpoint='0069', models=args.model, public_api='tcp',
         mode='text_decision', sdk_method='not_standard', native_early_compute_stop='unknown')
    ok, observations = True, []
    for index, model in enumerate(args.model):
        output = args.output / f'model-{index}'
        output.mkdir()
        def record(stage, **values):
            emit(stage, profile=output.name, **{'model': model, **values})
        try:
            passed, notes = await exercise_model(replace_args(args, output=output), model, record)
        except Exception as exc:
            passed, notes = False, [f'{type(exc).__name__}: {exc}']
            record('error', error=notes[0])
        ok &= passed
        observations.extend(f'{model}:{note}' for note in notes)
    emit('finished', protocol_ok=ok, review_required=not ok or bool(observations), observations=observations)
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Existing registered reference; repeat for sequential runs')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, help='Default: Laya 1024, Nimble/Clef 2048')
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--cancel-after-ms', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.model = args.model or list(MODELS)
    if any(not model.strip() for model in args.model) or len(set(args.model)) != len(args.model):
        parser.error('--model must be nonblank and unique')
    if args.ctx is not None and not 512 <= args.ctx <= 4096:
        parser.error('--ctx must be 512..4096')
    for name in ('load_timeout', 'request_timeout', 'cancel_after_ms', 'atol', 'rtol'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + ' must be finite and positive')
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error('--output already exists; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.DEBUG, filename=args.output / 'worker.log', encoding='utf-8')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8') as raw, (args.output / 'summary.jsonl').open('w', encoding='utf-8') as summary:
        def emit(stage, **values):
            row = {'stage': stage, **values}
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            raw.flush()
            if stage not in {'request', 'native-response', 'managed-response', 'public-response', 'native-memory', 'inventory-detail'}:
                concise = {k: v for k, v in row.items() if k not in {'argv', 'props', 'models', 'metadata', 'toml'}}
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
