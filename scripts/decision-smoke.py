#!/usr/bin/env python3
"""Inventory local Decision GGUFs and audit native Laya, Nimble or Clef."""

import argparse
import asyncio
import base64
from dataclasses import replace
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import time

import gguf

from tllama.backend import ModelManager
from tllama.decision_contract import DECISION_RUNTIME
from tllama.daemon_config import load_daemon_config
from tllama.helpers.gguf_metadata import _field_to_python_value
from tllama.helpers.model_toml import resolve_repo_relative_path
from tllama.native_config import _plain
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError


MODEL = 'ggml-org/Laya-GGUF/Laya-Q8_0'
MODELS = (
    'Jakevin/pplx-decider-flash-v1.2-GGUF/pplx-decider-flash-v1.2-Q4_K_M',
    'ggml-org/Bespoke-Nimble-9B-v3-GGUF/Bespoke-Nimble-9B-v3-Q4_K_M',
    'ggml-org/Clef-Flash-GGUF/Clef-Flash-Q4_K_M',
    'gguf/Tev1-4B-experimental-GGUF/tev1-4b-experimental-Q4_K_M',
    MODEL,
)
TYPES = {'openjev', 'lev', 'kev', 'nimble', 'laya', 'clef', 'pplx-decider', 'lfm2-d1', 'lfm2-d1-omni'}
REFERENCE_TYPES = {'laya', 'nimble', 'clef'}
STATE = 'Customer: I am furious! I was charged twice. Please refund the duplicate payment now.'
QUESTIONS = {
    'route': {'type': 'choice', 'instructions': 'Which team should handle this?',
              'criteria': {'billing': 'payments and refunds', 'shipping': 'delivery and parcels', 'technical': 'software faults'}},
    'urgency': {'type': 'score', 'instructions': 'How urgent is this?',
                'criteria': ['can wait', 'this week', 'today', 'right now']},
    'angry': {'type': 'noul', 'instructions': 'Is the customer angry?'},
}


def require(condition, message):
    if not condition:
        raise WorkerError(message)


def inventory(manager, model):
    """Read local metadata/tensor descriptors only; never fetch or load weights."""
    definition = manager._toml_path_for_reference(model)
    require(definition is not None and definition.is_file(), 'Registered model definition not found')
    source = manager._resolve_virtual_model_spec(model)
    require(source is not None and source.llm_model is not None, 'Imported GGUF reference required')
    path = resolve_repo_relative_path(source.llm_model, manager.models_dir).resolve()
    require(path.is_file(), 'Registered GGUF not found')
    before = path.stat()
    reader = gguf.GGUFReader(str(path), 'r')
    arch = _field_to_python_value(reader.fields.get('general.architecture'))
    require(isinstance(arch, str) and arch, 'Missing GGUF architecture')
    selected = {k: _field_to_python_value(v) for k, v in reader.fields.items()
                if k.startswith(arch + '.decision.') or k.startswith('tokenizer.chat_template')
                or k in {'general.architecture', 'general.name', 'general.file_type',
                         arch + '.context_length', arch + '.embedding_length', arch + '.embedding_length_out',
                         arch + '.block_count', arch + '.attention.causal', arch + '.pooling_type',
                         'tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.ggml.mask_token_id',
                         'tokenizer.ggml.seperator_token_id'}}
    # Descriptors only: accessing tensor.data would read model weights.
    tensors = [{'name': t.name, 'shape': [int(n) for n in t.shape], 'type': int(t.tensor_type),
                'bytes': int(t.n_bytes)} for t in reader.tensors]
    vocab = reader.fields.get('tokenizer.ggml.tokens')
    vocab_size = len(vocab.data) if vocab is not None else None
    special_tokens = {}
    for suffix in ('mask_token_id', 'seperator_token_id'):
        token = selected.get('tokenizer.ggml.' + suffix)
        if type(token) is int and vocab_size is not None and 0 <= token < vocab_size:
            special_tokens[suffix] = {'id': token, 'piece': bytes(vocab.parts[vocab.data[token]]).decode('utf-8')}
    del reader
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'GGUF changed during inventory')
    kind = selected.get(arch + '.decision.type')
    template = selected.get('tokenizer.chat_template.systemone')
    issues = []
    if kind not in TYPES:
        issues.append('missing_or_unknown_decision_type')
    if not isinstance(template, str) or not template:
        issues.append('missing_systemone_template')
    for key, value in selected.items():
        if key.startswith(arch + '.decision.temperature.') and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            issues.append('invalid_temperature:' + key)
    if kind == 'laya':
        blocks, head = selected.get(arch + '.block_count'), selected.get(arch + '.decision.block_count')
        if arch != 'modern-bert' or type(blocks) is not int or type(head) is not int or not 0 < head < blocks:
            issues.append('invalid_laya_head_blocks')
        budget = selected.get(arch + '.decision.max_head_tokens')
        if type(budget) is not int or budget <= 0:
            issues.append('missing_laya_head_budget')
        for suffix in ('mask_token_id', 'seperator_token_id'):
            token = selected.get('tokenizer.ggml.' + suffix)
            if type(token) is not int or vocab_size is None or not 0 <= token < vocab_size:
                issues.append('invalid_laya_' + suffix)
        names = {t['name'] for t in tensors}
        scorer = {'token_types.weight', 'cls.weight', 'cls.bias', 'cls.norm.weight',
                  'cls.norm.bias', 'cls.output.weight', 'cls.output.bias'}
        if scorer - names:
            issues.append('missing_laya_scorer:' + ','.join(sorted(scorer - names)))
    caution = ('Published v1.2 requires a custom noncausal runner/readout; pinned native pplx-decider targets an older prompt'
               if model == MODELS[0] else None)
    return {'model': model, 'path': str(path), 'size_bytes': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'registered_definition': str(definition),
            'registered_sha256': hashlib.sha256(definition.read_bytes()).hexdigest(),
            'projector_reference': source.mmproj_model, 'decision_type': kind,
            'header_candidate': not issues, 'issues': issues, 'source_caution': caution,
            'template_sha256': hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None,
            'vocab_size': vocab_size, 'special_tokens': special_tokens, 'metadata': selected, 'tensors': tensors}


def probe_launch(spec, args):
    options = {k: v for k, v in spec.runtime_options.items() if k not in {'draft_n_max', 'spm_infill'}}
    options.update(spec_type='none', n_batch=args.ctx, n_ubatch=args.ctx)
    # Native selects the Decision computation from its GGUF. Never force the
    # semantic embedding API's mode or pooled output onto a trained head.
    return replace(spec, n_ctx=args.ctx, mmproj=None, mmproj_options={}, template=None,
                   embedding=False, pooling_type=-1, runtime_options=options,
                   type_k='f16', type_v='f16', flash_attn=False,
                   memory_source='native', memory_compare=False)


class ProbeWorker(NativeWorker):
    async def systemone(self, body):
        async with self._response('/v1/systemone', {**body, 'model': self.spec.alias}) as response:
            return await self._json_body(response)

    async def models(self):
        async with self._slot:
            self._check_ready()
            async with self._client.stream('GET', '/v1/models') as response:
                return await self._json_body(response)


def reference_contract(selected):
    require(selected is not None and selected['header_candidate']
            and selected['decision_type'] in REFERENCE_TYPES and not selected['source_caution'],
            'Reference requires a complete local Laya, Nimble or Clef Decision GGUF; inspect inventory first')
    kind = selected['decision_type']
    return {'decision_type': kind,
            'prompt_composition': {'laya': 'independent_questions', 'nimble': 'whole_schema_per_question',
                                   'clef': 'joint_schema'}[kind],
            'choice_key_order': 'sorted' if kind == 'clef' else 'request',
            'single_combined_equality_required': kind == 'laya',
            'reversed_questions_equality_required': kind == 'laya',
            'additive_single_usage_required': kind == 'laya',
            'cancel_questions': 32 if kind == 'laya' else 2}


def inspect(payload, questions, kind='laya'):
    require(isinstance(payload, dict) and isinstance(payload.get('model'), str), 'Invalid Decision envelope')
    answers, usage = payload.get('answers'), payload.get('usage')
    require(isinstance(answers, dict) and list(answers) == list(questions), 'Question IDs/order differ from the request')
    require(isinstance(usage, dict) and type(usage.get('input_tokens')) is int and usage['input_tokens'] > 0
            and type(usage.get('output_tokens')) is int and usage['output_tokens'] == 0, 'Invalid Decision usage')

    def number(value, low, high):
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                'Invalid Decision numeric value')
        return value

    for qid, q in questions.items():
        answer = answers[qid]
        require(isinstance(answer, dict) and answer.get('type') == q['type'], 'Wrong Decision answer type')
        if q['type'] == 'noul':
            number(answer.get('noul'), 0, 1)
            continue
        keys = list(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]
        if kind == 'clef' and q['type'] == 'choice':
            keys.sort()
        probs = answer.get('probabilities')
        require(isinstance(probs, dict) and list(probs) == keys, 'Decision probability keys/order differ')
        p = [number(probs[k], 0, 1) for k in keys]
        require(abs(math.fsum(p) - 1) < 1e-4, 'Decision probabilities do not sum to one')
        n = len(p)
        if q['type'] == 'choice':
            require(answer.get('choice') in keys and probs[answer['choice']] >= max(p) - 1e-7,
                    'Decision choice is not an argmax')
            confidence = 1 if n == 1 else max(0, (max(p) - 1 / n) / (1 - 1 / n))
        else:
            require(answer.get('legend') == dict(zip(keys, q['criteria'])), 'Decision score legend differs')
            expected = math.fsum(i * v for i, v in enumerate(p))
            require(abs(number(answer.get('score'), 0, n - 1) - expected) < 1e-4, 'Decision score differs from expected level')
            mode = p.index(max(p))
            dist = math.fsum(v * abs(i - mode) for i, v in enumerate(p))
            uniform = math.fsum(abs(i - (n - 1) / 2) / n for i in range(n))
            confidence = max(0, 1 - dist / uniform)
        require(abs(number(answer.get('confidence'), 0, 1) - confidence) < 1e-4, 'Decision confidence formula differs')
    return payload


def compare(a, b, args, emit, notes, label, *, required=True):
    differences = []
    close = True
    def walk(x, y):
        nonlocal close
        if type(x) in (int, float) and type(y) in (int, float):
            differences.append(abs(x - y))
            close &= (x == y if type(x) is int and type(y) is int else
                      math.isclose(x, y, abs_tol=args.atol, rel_tol=args.rtol))
        elif isinstance(x, dict) and isinstance(y, dict) and set(x) == set(y):
            for key in x:
                walk(x[key], y[key])
        else:
            close &= x == y
    walk(a, b)
    emit('comparison' if required else 'composition-observation', probe=label,
         within_tolerance=close, max_abs_difference=max(differences, default=0))
    if required and not close:
        notes.append(label + ':difference')


async def memory(worker, emit, notes, phase):
    await worker.refresh_memory()
    telemetry = worker.telemetry.snapshot()
    emit('native-memory', phase=phase, telemetry=telemetry)
    measurement = telemetry.get('measurement') or {}
    report = measurement.get('report') or {}
    emit('memory', phase=phase, source=telemetry.get('source'), status=report.get('status'), totals=measurement.get('totals'))
    if report.get('status') != 'known':
        notes.append(phase + ':memory_not_known')


async def exercise(args, emit):
    emit('diagnostic-version', version=2, checkpoint='0066', model=args.model,
         mode='native_decision_reference', public_api='not_exercised',
         prompt_policy='embedded_systemone_template_native_head_and_calibration',
         semantic_quality='observations_only', native_early_compute_stop='unknown')
    config = load_daemon_config(False, discover=False).backend
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    workers, notes, inventory_rows, originals = [], [], {}, {}
    ok = True
    def factory(spec):
        launch = replace(probe_launch(spec, args), version_policy="strict",
                         runtime_fingerprint=DECISION_RUNTIME, expected_runtime_fingerprint=DECISION_RUNTIME)
        emit('diagnostic-overrides', configured_context=spec.n_ctx, effective_context=launch.n_ctx,
             configured_projector=str(spec.mmproj) if spec.mmproj else None, effective_projector=None,
             configured_runtime=_plain(spec.runtime_options), effective_runtime=_plain(launch.runtime_options),
             configured_embedding=spec.embedding, configured_pooling=spec.pooling_type,
             configured_cache_k=spec.type_k, configured_cache_v=spec.type_v,
             native_auto_head=True, cache_type='f16', flash_attn=False, spec_type='none')
        worker = ProbeWorker(launch)
        workers.append(worker)
        return worker
    manager = ModelManager(config, native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    manager.metadata_cache_dir = args.output / 'metadata-cache'
    try:
        for model in dict.fromkeys([*(args.inventory_model or MODELS), args.model]):
            path = manager._toml_path_for_reference(model)
            if path is not None and path.is_file():
                originals[path] = path.read_bytes()
                emit('registered-definition', phase='before', path=str(path), toml=originals[path].decode('utf-8'),
                     sha256=hashlib.sha256(originals[path]).hexdigest())
            try:
                item = await asyncio.to_thread(inventory, manager, model)
                inventory_rows[model] = item
                emit('inventory-detail', **item)
                emit('inventory', **{k: v for k, v in item.items() if k not in {'metadata', 'tensors'}})
            except Exception as exc:
                emit('inventory-error', model=model, error=f'{type(exc).__name__}: {exc}')
                notes.append(model + ':inventory_error')
        if not args.inventory_only:
            selected = inventory_rows.get(args.model)
            contract = reference_contract(selected)
            kind = contract['decision_type']
            independent = kind == 'laya'
            emit('decision-contract', **contract)
            async with manager.acquire_native_model(args.model, keep_alive=-1) as lease:
                worker, pid, generation = lease.worker, lease.worker.pid, lease.generation
                require(worker.spec.model.stat().st_size == selected['size_bytes']
                        and worker.spec.model.stat().st_mtime_ns == selected['mtime_ns'], 'Reference GGUF changed after inventory')
                models = await worker.models()
                require(isinstance(models, dict) and isinstance(models.get('data'), list) and len(models['data']) == 1
                        and models['data'][0].get('architecture', {}).get('output_modalities') == ['decisions'],
                        'Loaded native model does not advertise Decision output')
                emit('ready', pid=pid, generation=generation, argv=worker.command, props=worker.props,
                     fingerprint=worker.runtime_fingerprint, models=models)
                await memory(worker, emit, notes, 'after-load')
                async def collect(state, questions, label):
                    body = {'state': state, 'questions': questions}
                    emit('request', probe=label, path='/v1/systemone', body=body)
                    started = time.monotonic()
                    payload = await worker.systemone(body)
                    emit('native-response', probe=label, payload=payload)
                    inspect(payload, questions, kind)
                    emit('decision', probe=label, answers=payload['answers'], usage=payload['usage'],
                         elapsed_ms=(time.monotonic() - started) * 1000)
                    return payload
                singles = {key: await collect(STATE, {key: q}, 'single-' + key) for key, q in QUESTIONS.items()}
                together = await collect(STATE, QUESTIONS, 'combined')
                for key, single in singles.items():
                    compare(single['answers'][key], together['answers'][key], args, emit, notes,
                            'single-vs-combined-' + key, required=independent)
                single_tokens = sum(p['usage']['input_tokens'] for p in singles.values())
                emit('usage-composition', single_input_tokens=single_tokens,
                     combined_input_tokens=together['usage']['input_tokens'], additive_required=independent)
                if independent:
                    require(together['usage']['input_tokens'] == single_tokens, 'Laya independent question usage is not additive')
                repeat = await collect(STATE, QUESTIONS, 'repeat')
                compare(together, repeat, args, emit, notes, 'repeat')
                reverse = await collect(STATE, dict(reversed(list(QUESTIONS.items()))), 'reversed-questions')
                compare(together, reverse, args, emit, notes, 'reversed-questions', required=independent)
                if not independent:
                    reversed_repeat = await collect(STATE, dict(reversed(list(QUESTIONS.items()))), 'reversed-questions-repeat')
                    compare(reverse, reversed_repeat, args, emit, notes, 'reversed-questions-repeat')
                structured = await collect({'ticket': STATE, 'nested': {'plan': 'pro', 'attempts': [1, 2]}}, QUESTIONS, 'json-state')
                repeated = await collect({'ticket': STATE, 'nested': {'plan': 'pro', 'attempts': [1, 2]}}, QUESTIONS, 'json-state-repeat')
                compare(structured, repeated, args, emit, notes, 'json-state-repeat')
                special = {'řádek\nliteral\\n 🐈': {'type': 'choice', 'instructions': 'Which team?', 'criteria': {
                    'platba\nbilling': 'payments', 'literal\\nshipping': 'parcels', '日本語': 'software'}}}
                await collect('První řádek\nDruhý; literal \\n. 日本語 🐈 ' + STATE, special, 'literal-keys-unicode')
                await collect(STATE, {'one': {'type': 'choice', 'instructions': 'Choose the only option', 'criteria': {'only': None}}}, 'one-option')
                await collect(STATE, {'scale': {'type': 'score', 'instructions': 'How urgent?', 'criteria': [str(i) for i in range(10)]}}, 'ten-level-score')
                if not independent:
                    permuted = {'route': {**QUESTIONS['route'], 'criteria': dict(reversed(list(QUESTIONS['route']['criteria'].items())))}}
                    option_reverse = await collect(STATE, permuted, 'reversed-options')
                    compare(singles['route'], option_reverse, args, emit, notes, 'reversed-options', required=kind == 'clef')
                    option_repeat = await collect(STATE, permuted, 'reversed-options-repeat')
                    compare(option_reverse, option_repeat, args, emit, notes, 'reversed-options-repeat')
                    # Nimble switches from one-letter to short label codes above
                    # 26 options; Clef exercises its sorted joint option spans.
                    many = {'many': {'type': 'choice', 'instructions': 'Choose a category.',
                                     'criteria': {f'category-{i:02}': None for i in reversed(range(27))}}}
                    await collect(STATE, many, 'twenty-seven-options')
                    typed = {'typed': {'type': 'noul', 'instructions': {'question': 'Is the customer angry?', 'tags': ['support', 1]},
                                       'criteria': {'true': {'meaning': 'angry'}, 'false': ['calm', 'happy']}}}
                    await collect({'ticket': STATE}, typed, 'structured-instructions')
                changed = await collect('Customer: I am happy with my purchase. My parcel arrived on time.', QUESTIONS, 'changed-state')
                effect = max(abs(together['answers']['route']['probabilities'][k] - changed['answers']['route']['probabilities'][k])
                             for k in QUESTIONS['route']['criteria'])
                emit('state-sensitivity', max_probability_difference=effect, distinct=effect > args.atol,
                     baseline_choice=together['answers']['route']['choice'], changed_choice=changed['answers']['route']['choice'])
                if effect <= args.atol:
                    notes.append('changed_state:no_measurable_effect')
                if independent:
                    # Only Laya has this audited training-budget truncation.
                    head = selected['metadata']['modern-bert.decision.max_head_tokens']
                    require(head <= 4096, 'Head budget exceeds this bounded reference probe')
                    longq = {'long': {**QUESTIONS['route'], 'instructions': 'Please decide. ' * (head * 2)}}
                    await collect(STATE, longq, 'native-laya-head-truncation')
                    emit('native-budget', max_head_tokens=head, option_text_cap=48, state_policy='not_truncated_by_this_diagnostic')
                else:
                    emit('native-budget', max_head_tokens=None, option_text_cap=None,
                         state_policy='not_truncated_by_this_diagnostic', laya_truncation_probe='not_applicable')
                errors = [
                    ('missing-state', {'questions': QUESTIONS}, 400), ('missing-questions', {'state': STATE}, 400),
                    ('empty-questions', {'state': STATE, 'questions': {}}, 400),
                    ('unknown-type', {'state': STATE, 'questions': {'q': {'type': 'unknown', 'instructions': 'x'}}}, 400),
                    ('missing-instructions', {'state': STATE, 'questions': {'q': {'type': 'noul'}}}, 400),
                    ('empty-options', {'state': STATE, 'questions': {'q': {'type': 'choice', 'instructions': 'x', 'criteria': {}}}}, 400),
                    ('one-score-level', {'state': STATE, 'questions': {'q': {'type': 'score', 'instructions': 'x', 'criteria': ['x']}}}, 400),
                    ('too-many-options', {'state': STATE, 'questions': {'q': {**QUESTIONS['route'], 'criteria': {str(i): None for i in range(256)}}}}, 400),
                    ('remote-image', {'state': STATE, 'questions': QUESTIONS, 'images': ['https://example.invalid/a.png']}, 400),
                    ('unsupported-image', {'state': STATE, 'questions': QUESTIONS, 'images': ['data:image/png;base64,' + base64.b64encode(b'image').decode()]}, 501),
                ]
                for label, body, expected in errors:
                    emit('request', probe=label, path='/v1/systemone', body=body)
                    try:
                        await worker.systemone(body)
                    except WorkerRequestError as exc:
                        emit('native-error', probe=label, status=exc.http_status, payload=exc.payload)
                        require(exc.http_status == expected, label + ': unexpected native error status')
                    else:
                        raise WorkerError(label + ': native accepted invalid/unsupported input')
                recovered = await collect(STATE, QUESTIONS, 'after-errors')
                compare(together, recovered, args, emit, notes, 'after-errors')
                # Whole-schema prompts grow with question count. Do not turn
                # the cancellation control into an accidental context overflow.
                cancel = {'state': STATE, 'questions': {f'q{i}': QUESTIONS['route'] for i in range(contract['cancel_questions'])}}
                emit('request', probe='cancel', path='/v1/systemone', body=cancel)
                task = asyncio.create_task(worker.systemone(cancel))
                try:
                    done, _ = await asyncio.wait({task}, timeout=args.cancel_after_ms / 1000)
                    if done:
                        payload = await task
                        emit('native-response', probe='cancel-completed', payload=payload)
                        inspect(payload, cancel['questions'], kind)
                    else:
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    emit('transport-cancel', pending_at_cancel=not bool(done),
                         inference_started_observed=None, native_early_stop_observed=None)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                recovered = await collect(STATE, QUESTIONS, 'after-cancel')
                compare(together, recovered, args, emit, notes, 'after-cancel')
                require(worker.pid == pid and len(workers) == 1, 'Recovery changed the reference process')
                emit('recovery', pid=pid, generation=generation)
                await memory(worker, emit, notes, 'after-requests')
            await manager.unload_native_model(args.model)
            await manager.wait_native_closed(generation)
            require(not manager.native_snapshot(), 'Unload retained a native generation')
            emit('unloaded', generation=generation, residency=manager.native_snapshot())
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        await manager.shutdown()
        for i, worker in enumerate(workers):
            code = worker.process.returncode if worker.process else None
            removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
            clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
            ok &= clean
            emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
            (args.output / f'worker-{i}-load.log').write_text(worker.load_log, encoding='utf-8')
        for path, before in originals.items():
            unchanged = path.is_file() and path.read_bytes() == before
            ok &= unchanged
            emit('registered-definition', phase='after', path=str(path), unchanged=unchanged, sha256=hashlib.sha256(before).hexdigest())
        for item in inventory_rows.values():
            path = Path(item['path'])
            unchanged = path.is_file() and path.stat().st_size == item['size_bytes'] and path.stat().st_mtime_ns == item['mtime_ns']
            ok &= unchanged
            emit('gguf-fingerprint', model=item['model'], unchanged=unchanged)
    emit('finished', protocol_ok=ok, review_required=not ok or bool(notes), observations=notes,
         inventory_count=len(inventory_rows), inventory_only=args.inventory_only,
         inventory_followup=[{'model': item['model'], 'issues': item['issues'], 'source_caution': item['source_caution']}
                             for item in inventory_rows.values() if item['issues'] or item['source_caution']],
         residency=manager.native_snapshot(), public_api='not_exercised')
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--inventory-model', action='append', help='Override inventory references; repeat as needed')
    parser.add_argument('--inventory-only', action='store_true', help='Read local headers only; start no worker')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=1024)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--cancel-after-ms', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args(argv)
    if not 512 <= args.ctx <= 4096:
        parser.error('--ctx must be 512..4096')
    for name in ('load_timeout', 'request_timeout', 'cancel_after_ms', 'atol', 'rtol'):
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
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8') as raw, (args.output / 'summary.jsonl').open('w', encoding='utf-8') as summary:
        def emit(stage, **values):
            row = {'stage': stage, **values}
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            raw.flush()
            if stage not in {'request', 'native-response', 'native-memory', 'inventory-detail'}:
                concise = {k: v for k, v in row.items() if k not in {'argv', 'props', 'models', 'toml'}}
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
