#!/usr/bin/env python3
"""Compare native embedded MTP off/on before adding production configuration."""

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
from tllama.worker import NativeWorker, WorkerError
from tllama.worker_config import WorkerLaunchSpec


MODEL = 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S'
PROMPTS = {
    'copy': 'Copy this sequence, then continue to 100. Output only the comma-separated numbers: 1, 2, 3, 4, 5',
    'code': 'Write a Python function merge_sorted(a, b) merging two sorted lists without sorted() or sort(). '
            'Include type hints and a short example. Output only Python code.',
}
CANCEL = 'Write the integers from 1 to 10000, one per line. Do not summarize or stop early.'


def require(condition, message):
    if not condition:
        raise WorkerError(message)


@dataclass(frozen=True)
class ProbeLaunchSpec(WorkerLaunchSpec):
    """Diagnostic-only CLI extension, deliberately outside production config."""
    speculation: str = 'none'
    draft_tokens: int = 3

    def command(self, api_key_file, template_file=None):
        require(self.speculation in {'none', 'draft-mtp'}, 'Invalid diagnostic speculation type')
        require(type(self.draft_tokens) is int and 1 <= self.draft_tokens <= 32, 'Invalid draft token limit')
        return super().command(api_key_file, template_file) + (
            '--spec-type', self.speculation, '--spec-draft-n-max', str(self.draft_tokens))


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
    emit('diagnostic-version', version=1, model=args.model, public_api='not_exercised',
         native_contract='b11552-3144c167ef', note='Same GGUF, sequential fresh processes; diagnostic CLI flags only')
    for profile, speculation in [('off', 'none'), ('mtp', 'draft-mtp')]:
        workers, results = [], {}
        def factory(spec):
            values = {field.name: getattr(spec, field.name) for field in fields(WorkerLaunchSpec)}
            values.update(runtime_options={**spec.runtime_options, 'n_batch': 512, 'n_ubatch': 512},
                          memory_source='native', memory_compare=False)
            worker = NativeWorker(ProbeLaunchSpec(**values, speculation=speculation, draft_tokens=args.draft))
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
                for probe, prompt in PROMPTS.items():
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
                    async with worker.chat_stream(request(CANCEL, 512)) as events:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL, help='Registered model containing embedded MTP tensors')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--predict', type=int, default=128)
    parser.add_argument('--draft', type=int, default=3)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    args = parser.parse_args(argv)
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
            return 0 if asyncio.run(exercise(args, emit)) else 1
        except Exception as exc:
            emit('fatal', error=f'{type(exc).__name__}: {exc}')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
