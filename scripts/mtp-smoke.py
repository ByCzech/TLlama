#!/usr/bin/env python3
"""Compare native embedded MTP off/on and audit cached request state."""

import argparse
import asyncio
from dataclasses import dataclass, fields, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import time

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.worker import NativeWorker, WorkerError, WorkerStartupError, WorkerStartupTimeout
from tllama.worker_config import WorkerLaunchSpec


MODEL = 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S'
NO_MTP_MODEL = 'unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S'
MISSING_HEAD_REASON = "context type MTP requested but model doesn't contain MTP layers"
PROMPTS = {
    'copy': 'Copy this sequence, then continue to 100. Output only the comma-separated numbers: 1, 2, 3, 4, 5',
    'code': 'Write a Python function merge_sorted(a, b) merging two sorted lists without sorted() or sort(). '
            'Include type hints and a short example. Output only Python code.',
}
CANCEL = 'Write the integers from 1 to 10000, one per line. Do not summarize or stop early.'
STATE_PREFIX = [{'role': 'system', 'content': 'Follow the last user request. Background records:\n' + '\n'.join(
    f'Record {n:02d}: label=sample_{n:02d}; value={n * 17}; status=available.' for n in range(12))}]


def state_request(messages, predict, cache):
    body = request('', predict)
    return {**body, 'messages': messages, 'cache_prompt': cache}


def compare_state(reference, actual, emit, observations, label):
    before, after = reference['metrics']['predicted_n'], actual['metrics']['predicted_n']
    checks = {'exact_text_match': reference['text_sha256'] == actual['text_sha256'],
              'finish_match': reference['finish_reason'] == actual['finish_reason'],
              'output_tokens_match': None if before is None or after is None else before == after}
    emit('cache-comparison', probe=label, **checks,
         reference_hash=reference['text_sha256'], actual_hash=actual['text_sha256'],
         cached_tokens=actual['metrics']['cache_n'])
    if not all(checks.values()):
        observations.append(f'{label}:cached_result_differs')


async def state_checks(worker, args, emit, observations):
    """Compare each cached path with its own cache-disabled control."""
    results = {}
    base = STATE_PREFIX + [{'role': 'user', 'content': PROMPTS['copy']}]
    async def run(label, messages, cache, stream=False):
        result = await collect(worker, state_request(messages, args.predict, cache), stream, emit, probe=label)
        results[label, stream] = result
        if not result['content']:
            observations.append(f'{label}:empty_content')
        if not cache and result['metrics']['cache_n'] != 0:
            observations.append(f'{label}:cache_disabled_not_measured_zero')
        return result
    cold = await run('base-cold', base, False)
    repeat = await run('base-repeat', base, True, True)
    compare_state(cold, repeat, emit, observations, 'repeat')
    # Replay exactly the actual first answer, even when its token budget ended.
    extended = base + [{'role': 'assistant', 'content': cold['content']},
                       {'role': 'user', 'content': PROMPTS['code']}]
    extension = await run('extension-cold', extended, False)
    await run('base-prime', base, True)
    continued = await run('extension-cached', extended, True, True)
    compare_state(extension, continued, emit, observations, 'extension')
    branch = STATE_PREFIX + [{'role': 'user', 'content': PROMPTS['code']}]
    alternate = await run('branch-cold', branch, False)
    await run('extension-prime', extended, True)
    changed = await run('branch-cached', branch, True, True)
    compare_state(alternate, changed, emit, observations, 'branch')
    shortened = await run('base-return', base, True, True)
    compare_state(cold, shortened, emit, observations, 'shortened')
    if not any(r['metrics']['cache_n'] and r['metrics']['cache_n'] > 0
               for r in (repeat, continued, changed, shortened)):
        observations.append('state:no_measured_prefix_reuse')
    emit('state-evidence', measured_prefix_reuse=any(
        (r['metrics']['cache_n'] or 0) > 0 for r in (repeat, continued, changed, shortened)),
        rollback_proof='Inspect worker.log: checkpoint restore or native rollback is not implied by cache_n')
    return results, base, cold


def require(condition, message):
    if not condition:
        raise WorkerError(message)


@dataclass(frozen=True)
class ProbeLaunchSpec(WorkerLaunchSpec):
    """Diagnostic-only CLI extension, deliberately outside production config."""
    speculation: str = 'none'
    draft_tokens: int = 3

    def command(self, api_key_file, template_file=None, socket_file=None):
        require(self.speculation in {'none', 'draft-mtp'}, 'Invalid diagnostic speculation type')
        require(type(self.draft_tokens) is int and 1 <= self.draft_tokens <= 32, 'Invalid draft token limit')
        return super().command(api_key_file, template_file, socket_file) + (
            '--spec-type', self.speculation, '--spec-draft-n-max', str(self.draft_tokens))


def probe_launch(spec, args, speculation):
    values = {field.name: getattr(spec, field.name) for field in fields(WorkerLaunchSpec)}
    # This diagnostic owns off/on selection even when TOML enables MTP.
    # Emit each switch once, with no last-wins ambiguity.
    runtime = {k: v for k, v in spec.runtime_options.items() if k not in {'spec_type', 'draft_n_max'}}
    values.update(runtime_options={**runtime, 'n_batch': 512, 'n_ubatch': 512},
                  memory_source='native', memory_compare=False)
    return ProbeLaunchSpec(**values, speculation=speculation, draft_tokens=args.draft)


def metrics(timings):
    require(isinstance(timings, dict), 'Invalid native timings')
    result = {}
    for name in ('draft_n', 'draft_n_accepted', 'predicted_n', 'prompt_n', 'cache_n'):
        value = timings.get(name)
        require(value is None or type(value) is int and value >= 0, f'Invalid native {name}')
        result[name] = value
    for name in ('prompt_ms', 'predicted_ms', 'predicted_per_second'):
        value = timings.get(name)
        require(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f'Invalid native {name}')
        result[name] = value
    drafted, accepted = result['draft_n'], result['draft_n_accepted']
    require(accepted is None or drafted is not None and accepted <= drafted, 'Inconsistent draft counters')
    result['acceptance'] = accepted / drafted if drafted and accepted is not None else None
    result['mtp_observed'] = drafted is not None and drafted > 0
    return result


def request(prompt, predict):
    return {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'seed': 42,
            'max_tokens': predict, 'cache_prompt': False, 'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False}}


async def collect(worker, body, stream, emit, **labels):
    emit('request', stream=stream, body=body, **labels)
    started = time.perf_counter()
    content, reasoning, finish, timings, usage, ttft = '', '', None, {}, None, None
    if stream:
        async with worker.chat_stream(body) as events:
            async for payload in events:
                emit('native-event', payload=payload, **labels)
                if 'timings' in payload:
                    timings = payload['timings']
                if 'usage' in payload:
                    usage = payload['usage']
                choices = payload.get('choices', [])
                require(isinstance(choices, list) and len(choices) <= 1, 'Invalid native choices')
                if not choices:
                    continue
                choice = choices[0]
                require(finish is None and choice.get('index', 0) == 0, 'Data after native finish or wrong index')
                delta = choice.get('delta', {})
                require(isinstance(delta, dict) and not delta.get('tool_calls'), 'Unexpected native tool calls/delta')
                text, thought = delta.get('content') or '', delta.get('reasoning_content') or ''
                require(isinstance(text, str) and isinstance(thought, str), 'Invalid native text delta')
                if ttft is None and (text or thought):
                    ttft = (time.perf_counter() - started) * 1000
                content += text
                reasoning += thought
                finish = choice.get('finish_reason')
    else:
        payload = await worker.chat(body)
        emit('native-response', payload=payload, **labels)
        choices = payload.get('choices', [])
        require(isinstance(choices, list) and len(choices) == 1, 'Missing native choice')
        choice = choices[0]
        message = choice.get('message', {})
        require(isinstance(message, dict) and not message.get('tool_calls'), 'Unexpected native tool calls/message')
        content, reasoning = message.get('content') or '', message.get('reasoning_content') or ''
        finish, timings, usage = choice.get('finish_reason'), payload.get('timings', {}), payload.get('usage')
    require(finish in {'stop', 'length'}, 'Missing or invalid native finish')
    require(isinstance(content, str) and isinstance(reasoning, str), 'Invalid native text')
    result = {'stream': stream, 'finish_reason': finish, 'content': content, 'reasoning': reasoning,
              'usage': usage, 'timings': timings, 'metrics': metrics(timings), 'ttft_ms': ttft,
              'elapsed_ms': (time.perf_counter() - started) * 1000,
              'text_sha256': hashlib.sha256((content + '\0' + reasoning).encode()).hexdigest()}
    emit('assembled', **result, **labels)
    return result


async def memory(worker, emit, phase):
    await worker.refresh_memory()
    telemetry = worker.telemetry.snapshot()
    emit('native-memory', phase=phase, telemetry=telemetry)
    measurement = telemetry.get('measurement') or {}
    report = measurement.get('report') or {}
    totals = measurement.get('totals') or {}
    owners = [owner['id'] for owner in report.get('owners', []) if 'mtp' in owner.get('roles', [])]
    emit('memory', phase=phase, source=telemetry.get('source'), status=report.get('status'),
         snapshot_sequence=report.get('snapshot_sequence'), totals=totals, mtp_owner_ids=owners)
    return owners


async def exercise(args, emit):
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    profiles, observations, all_ok = {}, [], True
    baseline_definition, baseline_runtime = None, None
    emit('diagnostic-version', version=2, model=args.model, public_api='not_exercised',
         mode='state' if args.state_only else 'reference',
         native_contract='b11552-3144c167ef', note='Same GGUF, sequential fresh processes; diagnostic CLI flags only')
    for profile, speculation in [('off', 'none'), ('mtp', 'draft-mtp')]:
        workers, results = [], {}
        def factory(spec):
            worker = NativeWorker(probe_launch(spec, args, speculation))
            workers.append(worker)
            return worker
        manager = ModelManager(config, native_worker_factory=factory, native_launch_options={
            'executable': args.executable.expanduser().resolve() if args.executable else None,
            'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
        def record(stage, **values):
            emit(stage, profile=profile, **values)
        ok = True
        try:
            started = time.perf_counter()
            async with manager.acquire_native_model(args.model, num_ctx=args.ctx, keep_alive=0) as lease:
                worker = lease.worker
                if baseline_definition is None:
                    baseline_definition, baseline_runtime = lease.definition, worker.runtime_fingerprint
                require(baseline_definition.same_version(lease.definition), 'Model definition/files changed between profiles')
                require(worker.runtime_fingerprint == baseline_runtime, 'Native build changed between profiles')
                record('ready', pid=worker.pid, generation=lease.generation, argv=worker.command,
                       fingerprint=worker.runtime_fingerprint, props=worker.props,
                       model_path=str(worker.spec.model), model_size_bytes=worker.spec.model.stat().st_size,
                       load_elapsed_ms=(time.perf_counter() - started) * 1000)
                await memory(worker, record, 'after-load')
                control = await collect(worker, request('Reply exactly READY.', 16), False, record, probe='warmup')
                if args.state_only:
                    state_observations = []
                    results, state_base, state_control = await state_checks(worker, args, record, state_observations)
                    observations.extend(f'{profile}:{item}' for item in state_observations)
                for probe, prompt in (() if args.state_only else PROMPTS.items()):
                    for stream in (False, True):
                        result = await collect(worker, request(prompt, args.predict), stream, record, probe=probe)
                        results[probe, stream] = result
                        if not result['content']:
                            observations.append(f'{profile}:{probe}:{stream}:empty_content')
                    same = results[probe, False]['text_sha256'] == results[probe, True]['text_sha256']
                    record('stream-comparison', probe=probe, exact_text_match=same)
                    if not same:
                        observations.append(f'{profile}:{probe}:stream_text_difference')
                owners = await memory(worker, record, 'after-requests')
                if profile == 'mtp':
                    if not any(r['metrics']['mtp_observed'] for r in results.values()):
                        observations.append('mtp:no_measured_drafting')
                    if not owners:
                        observations.append('mtp:no_memory_owner_evidence')
                    live = False
                    cancel_body = (state_request(state_base + [{'role': 'user', 'content': CANCEL}], 512, True)
                                   if args.state_only else request(CANCEL, 512))
                    async with worker.chat_stream(cancel_body) as events:
                        async for payload in events:
                            record('native-event', probe='cancel', payload=payload)
                            choices = payload.get('choices', [])
                            if choices:
                                delta = choices[0].get('delta', {})
                                if delta.get('content') or delta.get('reasoning_content'):
                                    live = choices[0].get('finish_reason') is None
                                    break
                    record('stream-cancel', live_delta_seen=live, pid=worker.pid,
                           native_cancellation_proof='Review worker.log for actual early native stop; closure/reuse alone is not proof')
                    if not live:
                        observations.append('mtp:no_live_cancel_delta')
                    await memory(worker, record, 'after-cancel')
                    if args.state_only:
                        recovered = await collect(worker, state_request(state_base, args.predict, True),
                                                  True, record, probe='state-after-cancel')
                        compare_state(state_control, recovered, record, observations, 'after-cancel')
                        fresh = await collect(worker, state_request(state_base, args.predict, False),
                                              False, record, probe='state-after-cancel-cold')
                        compare_state(state_control, fresh, record, observations, 'after-cancel-cold')
                    answer = await collect(worker, request('Reply exactly READY.', 16), False, record, probe='after-cancel')
                    # Compare identical requests before/after cancellation;
                    # punctuation chosen consistently by the model is harmless.
                    if (not control['content'].strip() or control['finish_reason'] != 'stop'
                            or answer['finish_reason'] != 'stop' or answer['text_sha256'] != control['text_sha256']):
                        observations.append('mtp:after_cancel_reply_differs')
                    record('after-cancel', pid=worker.pid, generation=lease.generation)
        except Exception as exc:
            ok = False
            record('error', error=f'{type(exc).__name__}: {exc}')
        finally:
            await manager.shutdown()
            for worker in workers:
                code = worker.process.returncode if worker.process else None
                removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
                clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
                ok &= clean
                record('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
                (args.output / f'{profile}-load.log').write_text(worker.load_log, encoding='utf-8')
            require(not manager.native_snapshot(), 'MTP diagnostic retained a generation after shutdown')
            record('profile-finished', protocol_ok=ok, residency=manager.native_snapshot())
        profiles[profile] = results
        all_ok &= ok
    for key in profiles['off']:
        if key not in profiles['mtp']:
            continue
        off, mtp = profiles['off'][key], profiles['mtp'][key]
        same = off['text_sha256'] == mtp['text_sha256']
        if not same:
            observations.append(f'comparison:{key[0]}:{key[1]}:text_difference')
        a, b = off['metrics']['predicted_per_second'], mtp['metrics']['predicted_per_second']
        emit('comparison', probe=key[0], stream=key[1], exact_text_match=same,
             generation_tps_ratio=b / a if a and b is not None else None,
             off_metrics=off['metrics'], mtp_metrics=mtp['metrics'],
             note='Short diagnostic, not a statistical benchmark; inspect token counts and content')
    emit('finished', protocol_ok=all_ok, review_required=bool(observations) or not all_ok,
         observations=observations, public_api='not_exercised')
    return all_ok


async def exercise_negative(args, emit):
    """Control -> missing-head startup rejection -> fresh working control."""
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    observations, controls = [], []
    baseline_definition = baseline_runtime = baseline_binary = None
    all_ok = True
    emit('diagnostic-version', version=3, model=args.model, mode='negative', public_api='not_exercised',
         expected_native_reason=MISSING_HEAD_REASON,
         note='Separate non-MTP GGUF; diagnostic flags only, no TOML edits or automatic fallback')
    for profile, speculation in [('off-before', 'none'), ('mtp-reject', 'draft-mtp'), ('off-after', 'none')]:
        workers, rejected, ok = [], False, True
        def record(stage, **values):
            emit(stage, profile=profile, **values)
        def factory(spec):
            nonlocal baseline_binary
            spec = probe_launch(spec, args, speculation)
            stat = spec.executable.stat()
            identity = (str(spec.executable), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if baseline_binary is None:
                baseline_binary = identity
            require(identity == baseline_binary, 'Native executable changed between profiles')
            worker = NativeWorker(spec)
            workers.append(worker)
            return worker
        manager = ModelManager(config, native_worker_factory=factory, native_launch_options={
            'executable': args.executable.expanduser().resolve() if args.executable else None,
            'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
        try:
            definition = await manager._resolve_native_definition(args.model, args.ctx)
            if baseline_definition is None:
                baseline_definition = definition
            require(baseline_definition.same_version(definition), 'Model definition/files changed between profiles')
            try:
                async with manager.acquire_native_model(args.model, num_ctx=args.ctx, keep_alive=0) as lease:
                    worker = lease.worker
                    require(definition.same_version(lease.definition), 'Definition changed during admission')
                    record('ready', pid=worker.pid, argv=worker.command, fingerprint=worker.runtime_fingerprint,
                           model_path=str(worker.spec.model), model_size_bytes=worker.spec.model.stat().st_size)
                    if profile == 'mtp-reject':
                        observations.append('mtp-reject:unexpected_ready_or_native_fallback')
                        ok = False
                    else:
                        if baseline_runtime is None:
                            baseline_runtime = worker.runtime_fingerprint
                        require(worker.runtime_fingerprint == baseline_runtime, 'Native build changed between controls')
                        result = await collect(worker, request('Reply exactly READY.', 16), False, record, probe='control')
                        require(result['content'].strip() and result['finish_reason'] == 'stop', 'Control failed to answer')
                        require(not result['metrics']['mtp_observed'], 'Disabled control unexpectedly drafted tokens')
                        controls.append(result)
            except WorkerStartupError as exc:
                reason = any(MISSING_HEAD_REASON in worker.load_log for worker in workers)
                # A generic error, timeout, or a process killed by our cleanup
                # does not prove rejection because of a missing MTP head.
                exited = (len(workers) == 1 and workers[0].process is not None
                          and workers[0].process.returncode not in (None, 0, -signal.SIGTERM, -signal.SIGKILL))
                rejected = profile == 'mtp-reject' and reason and exited and not isinstance(exc, WorkerStartupTimeout)
                record('startup-rejection', expected=rejected, missing_head_reason=reason,
                       error=f'{type(exc).__name__}: {exc}')
                if not rejected:
                    observations.append(f'{profile}:unrelated_or_unproven_startup_failure')
                    ok = False
            if profile == 'mtp-reject' and not rejected:
                ok = False
            if len(workers) != 1:
                observations.append(f'{profile}:expected_one_launch_without_fallback')
                ok = False
        except Exception as exc:
            ok = False
            record('error', error=f'{type(exc).__name__}: {exc}')
        finally:
            await manager.shutdown()
            for worker in workers:
                code = worker.process.returncode if worker.process else None
                removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
                clean = worker.state == 'closed' and removed and code is not None
                if profile != 'mtp-reject':
                    clean &= code in (0, -signal.SIGTERM)
                ok &= clean
                record('worker-closed', pid=worker.pid, argv=worker.command, returncode=code,
                       directory_removed=removed, clean=clean)
                (args.output / f'{profile}-load.log').write_text(worker.load_log, encoding='utf-8')
            require(not manager.native_snapshot(), 'Negative MTP diagnostic retained a generation')
            record('profile-finished', protocol_ok=ok, residency=manager.native_snapshot())
        all_ok &= ok
    if len(controls) == 2:
        compare_state(controls[0], controls[1], emit, observations, 'off-recovery')
    else:
        observations.append('off-recovery:missing_control')
    emit('finished', protocol_ok=all_ok, review_required=bool(observations) or not all_ok,
         observations=observations, public_api='not_exercised')
    return all_ok and not observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Registered model; negative mode defaults to the existing non-MTP Qwen')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--predict', type=int, default=128)
    parser.add_argument('--draft', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--state-only', action='store_true', help='Replace the reference pair with prefix/branch/recovery probes')
    mode.add_argument('--negative-only', action='store_true', help='Verify missing-head rejection between two MTP-disabled controls')
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    args = parser.parse_args(argv)
    args.model = args.model or (NO_MTP_MODEL if args.negative_only else MODEL)
    if not (1 <= args.draft <= 32 and 16 <= args.predict < args.ctx and args.ctx >= 2048):
        parser.error('Need draft 1..32, predict >= 16, ctx >= 2048 and predict < ctx')
    if any(not math.isfinite(v) or v <= 0 for v in (args.load_timeout, args.request_timeout)):
        parser.error('Timeouts must be finite and positive')
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
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); raw.flush()
            if stage not in {'request', 'native-event', 'native-response', 'native-memory'}:
                concise = {k: v for k, v in row.items() if k not in {'props', 'argv', 'content', 'reasoning'}}
                line = json.dumps(concise, ensure_ascii=False, allow_nan=False)
                summary.write(line + '\n'); summary.flush()
                print(line, flush=True)
        try:
            exercise_mode = exercise_negative if args.negative_only else exercise
            return 0 if asyncio.run(exercise_mode(args, emit)) else 1
        except Exception as exc:
            emit('fatal', error=f'{type(exc).__name__}: {exc}')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
