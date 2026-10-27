#!/usr/bin/env python3
"""Check public Tools against independent client assembly over loopback HTTP."""

import argparse
import asyncio
from copy import deepcopy
from contextlib import aclosing
from dataclasses import replace
import importlib.util
import json
import logging
import math
from pathlib import Path
import signal
import socket
import time

import httpx
import uvicorn

from tllama.backend import ModelManager
from tllama.daemon_config import load_daemon_config
from tllama.worker import NativeWorker, WorkerError


# Share only test declarations/prompts and semantic inspection. The public
# consumer below neither imports production tool state nor uses native Assembly.
spec = importlib.util.spec_from_file_location('tools_reference', Path(__file__).with_name('tools-smoke.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def require(condition, message):
    if not condition:
        raise WorkerError(message)


async def wait_until(predicate, timeout=15):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


class Consumer:
    """A client appends OpenAI deltas and receives whole Ollama objects."""

    def __init__(self, api):
        self.api, self.calls = api, {}
        self.message = {'role': 'assistant', 'content': ''}
        self.finish = None

    def push(self, payload, stream):
        require('error' not in payload, f'Public error: {payload}')
        if self.api == 'openai':
            choices = payload['choices']
            if not choices:
                return
            require(len(choices) == 1 and self.finish is None, 'Invalid public choice/terminal order')
            choice = choices[0]
            message = choice['delta' if stream else 'message']
            finish = choice.get('finish_reason')
        else:
            require(self.finish is None, 'Ollama response after done')
            message = payload['message']
            finish = payload.get('done_reason') if payload.get('done') else None
            require(not payload.get('done') or finish in {'stop', 'length'}, 'Invalid Ollama terminal')
        self.message['content'] += message.get('content') or ''
        reasoning = message.get('thinking' if self.api == 'ollama' else 'reasoning_content') or ''
        if reasoning:
            field = 'thinking' if self.api == 'ollama' else 'reasoning_content'
            self.message[field] = self.message.get(field, '') + reasoning
        for ordinal, call in enumerate(message.get('tool_calls', [])):
            if self.api == 'ollama':
                require(isinstance(call['function']['arguments'], dict), 'Ollama arguments are not an object')
                index = call['function']['index']
                require(index not in self.calls, 'Ollama replayed a call')
                self.calls[index] = call
            elif not stream:
                self.calls[ordinal] = call
            else:
                index = call['index']
                current = self.calls.setdefault(index, {'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if 'id' in call:
                    require('id' not in current, 'Repeated public call ID header')
                    current['id'] = call['id']
                for field, text in call.get('function', {}).items():
                    current['function'][field] += text
        if finish:
            self.finish = finish

    def result(self):
        require(self.finish is not None, 'Public response lacks its terminal event')
        calls = [self.calls[i] for i in sorted(self.calls)]
        if self.api == 'openai':
            require(not calls or self.finish in {'tool_calls', 'length'}, 'Wrong OpenAI tool finish reason')
            require(self.finish != 'tool_calls' or calls, 'OpenAI tool finish without calls')
        require(all(c.get('id') for c in calls) and len({c['id'] for c in calls}) == len(calls),
                'Missing/duplicate public IDs')
        args = []
        for call in calls:
            value = call['function']['arguments']
            if self.api == 'openai':
                require(isinstance(value, str), 'OpenAI arguments are not a string')
                try:
                    value = reference.json_arguments(value)
                except ValueError as exc:
                    require(self.finish == 'length', 'Invalid completed public tool arguments')
                    args.append({'valid': False, 'error': str(exc), 'arguments': None})
                    continue
            args.append({'valid': True, 'arguments': value})
        if calls:
            self.message['tool_calls'] = calls
        return {'message': self.message, 'arguments': args,
                'finish_reason': 'tool_calls' if calls and self.finish == 'stop' else self.finish}


async def collect(client, path, body, api, emit, **labels):
    emit('request', api=api, body=body, **labels)
    consumer = Consumer(api)
    started = time.monotonic()
    async with client.stream('POST', path, json=body) as response:
        if response.status_code != 200:
            await response.aread()
            raise WorkerError(f'HTTP {response.status_code}: {response.text}')
        if not body['stream']:
            await response.aread()
            payload = response.json()
            emit('public-response', api=api, payload=payload, **labels)
            consumer.push(payload, False)
        else:
            done = False
            frames = 0
            async for line in response.aiter_lines():
                if not line or api == 'openai' and not line.startswith('data: '):
                    continue
                frames += 1
                require(frames <= 8192, 'Too many public diagnostic frames')
                require(not done, 'Public event after terminal sentinel')
                if api == 'openai':
                    line = line[6:]
                    if line == '[DONE]':
                        done = True
                        continue
                payload = json.loads(line)
                emit('public-event', api=api, payload=payload,
                     elapsed_ms=round((time.monotonic() - started) * 1000, 3), **labels)
                consumer.push(payload, True)
            require(api != 'openai' or done, 'OpenAI stream lacks [DONE]')
    return consumer.result()


def body_for(name, api, probe, native, stream, predict):
    common = {'model': name, 'messages': native['messages'], 'tools': native['tools'], 'stream': stream}
    if api == 'ollama':
        common.update(think=probe == 'thinking-auto', keep_alive=-1,
                      options={'temperature': 0, 'seed': 42, 'num_predict': predict})
    else:
        common.update(temperature=0, max_tokens=predict, tool_choice=native['tool_choice'],
                      parallel_tool_calls=native['parallel_tool_calls'],
                      reasoning_effort='medium' if probe == 'thinking-auto' else 'none')
        if probe == 'named-required':
            common['tool_choice'] = {'type': 'function', 'function': {'name': 'lookup_record'}}
    return common


async def exercise_clients(args, address, emit):
    """Use SDK objects directly, including Ollama's ID-less history shape."""
    try:
        import openai
        import ollama
    except ImportError as exc:
        raise WorkerError('SDK smoke needs the optional openai and ollama Python packages: ' + str(exc)) from exc
    from importlib.metadata import version
    emit('sdk-versions', openai=version('openai'), ollama=version('ollama'))
    review = []
    native = reference.probes(args.predict)[3][1]
    async with openai.AsyncOpenAI(base_url=address + '/v1', api_key='local-test', max_retries=0,
            timeout=args.request_timeout, http_client=openai.DefaultAsyncHttpxClient(trust_env=False)) as oa:
        ol = ollama.AsyncClient(host=address, trust_env=False, timeout=args.request_timeout,
                                headers={'authorization': 'Bearer local-test'})
        try:
            for api, client in (('openai', oa), ('ollama', ol)):
                for stream in (False, True):
                    body = body_for(args.model, api, 'parallel-auto', native, stream, args.predict)
                    if api == 'openai':
                        body.pop('model')
                        response = await client.chat.completions.create(model=args.model, **body)
                        consumer = Consumer('openai')
                        if stream:
                            async with response:
                                async for part in response:
                                    payload = part.model_dump(exclude_none=True)
                                    emit('sdk-event', api=api, stream=True, payload=payload)
                                    consumer.push(payload, True)
                        else:
                            payload = response.model_dump(exclude_none=True)
                            emit('sdk-event', api=api, stream=False, payload=payload)
                            consumer.push(payload, False)
                        result = consumer.result()
                    else:
                        response = await client.chat(**body)
                        message = {'role': 'assistant', 'content': ''}
                        calls = []
                        if stream:
                            async with aclosing(response):
                                async for part in response:
                                    emit('sdk-event', api=api, stream=True, payload=part.model_dump(exclude_none=True))
                                    message['content'] += part.message.content or ''
                                    calls.extend(part.message.tool_calls or [])
                            require(part.done, 'Ollama SDK stream lacks done')
                        else:
                            emit('sdk-event', api=api, stream=False, payload=response.model_dump(exclude_none=True))
                            message['content'] = response.message.content or ''
                            calls = response.message.tool_calls or []
                        message['tool_calls'] = [call.model_dump(exclude_none=True) for call in calls]
                        require(all(isinstance(c.function.arguments, dict) for c in calls), 'SDK arguments are not objects')
                        result = {'message': message, 'arguments': [{'valid': True, 'arguments': dict(c.function.arguments)} for c in calls],
                                  'finish_reason': 'tool_calls' if calls else 'stop'}
                    match = reference.semantics(result, ['alpha', 'beta'])
                    emit('sdk-assembled', api=api, stream=stream, semantic_match=match, **result)
                    if not match:
                        review.append(f'sdk:{api}:{stream}:semantic_mismatch')
                        continue
                    markers = [f'SDK_RESULT_7f4a_{i}' for i in range(2)]
                    if api == 'openai':
                        history = native['messages'] + [result['message']] + [
                            {'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps({'marker': marker})}
                            for c, marker in zip(result['message']['tool_calls'], markers)]
                        answer = await client.chat.completions.create(model=args.model, messages=history,
                            tools=reference.TOOLS, tool_choice='none', temperature=0, max_tokens=args.predict,
                            reasoning_effort='none')
                        content = answer.choices[0].message.content or ''
                        payload = answer.model_dump(exclude_none=True)
                        complete = answer.choices[0].finish_reason == 'stop' and not answer.choices[0].message.tool_calls
                    else:
                        history = native['messages'] + [ollama.Message(role='assistant', content=message['content'], tool_calls=calls)] + [
                            ollama.Message(role='tool', tool_name=c.function.name, content=json.dumps({'marker': marker}))
                            for c, marker in zip(calls, markers)]
                        answer = await client.chat(model=args.model, messages=history, tools=reference.TOOLS,
                            stream=False, think=False, options={'temperature': 0, 'num_predict': args.predict})
                        content = answer.message.content or ''
                        payload = answer.model_dump(exclude_none=True)
                        complete = answer.done and answer.done_reason == 'stop' and not answer.message.tool_calls
                    match = complete and all(marker in content for marker in markers)
                    emit('sdk-followup', api=api, stream=stream, semantic_match=match, response=payload)
                    if not match:
                        review.append(f'sdk:{api}:{stream}:followup_mismatch')
        finally:
            await ol.close()
    return review


def escape_probes(predict):
    original = next(case for case in reference.probes(predict) if case[0] == 'escaped-required')
    literal = deepcopy(original[1])
    key = reference.ESCAPED.replace('\n', '\\n')
    literal['messages'][-1]['content'] = (
        'Retrieve these record keys using lookup_record, one call per key: ' + json.dumps([key], ensure_ascii=False))
    return [original, ('literal-backslash-n', literal, [key])]


def inspect_escape(body, payload, result, expected):
    debug = payload.get('__verbose')
    require(isinstance(debug, dict), 'Native response lacks __verbose evidence')
    raw, prompt, settings = (debug.get(key) for key in ('content', 'prompt', 'generation_settings'))
    require(isinstance(raw, str) and bool(raw) and isinstance(prompt, str) and bool(prompt),
            'Native debug content/prompt must be nonempty strings')
    require(isinstance(settings, dict) and isinstance(settings.get('grammar'), str) and bool(settings['grammar']),
            'Native debug evidence lacks the active tool grammar')
    native = reference.full_result(payload)
    require(len(native['arguments']) == 1 and native['arguments'][0]['valid'], 'Expected one complete native tool argument')
    native_args = native['arguments'][0]['arguments']
    require(set(native_args) == {'key'} and isinstance(native_args['key'], str), 'Expected one native string key')
    require(reference.signatures(native) == reference.signatures(result), 'Native/public tool argument mismatch')
    # Inspect only the exact single-parameter XML envelope. Unknown formats stay
    # reviewable; never guess an escape convention or repair the native value.
    opener, closer = '<parameter=key>\n', '\n</parameter>'
    recognized = (raw.count('<tool_call>') == raw.count('</tool_call>') == 1
                  and raw.count('<function=lookup_record>') == raw.count('</function>') == 1
                  and raw.count(opener) == raw.count(closer) == 1)
    raw_key = raw.split(opener, 1)[1].split(closer, 1)[0] if recognized else None
    input_preserved = body['messages'][-1]['content'] in prompt
    native_key = native_args['key']
    diagnosis = ('input-needs-review' if not input_preserved else
                 'raw-format-needs-review' if not recognized else
                 'native-parser-difference' if raw_key != native_key else
                 'generated-value-mismatch' if raw_key != expected else 'exact')
    return {'diagnosis': diagnosis, 'input_text_preserved': input_preserved,
            'raw_xml_recognized': recognized, 'expected_key': expected,
            'raw_xml_key': raw_key, 'native_key': native_key,
            'native_public_match': True, 'raw_native_match': recognized and raw_key == native_key,
            'semantic_match': reference.semantics(result, [expected]),
            'note': 'Raw debug text is from the same nonstream inference; no parser/grammar override.'}


async def exercise_escapes(args, client, worker, emit):
    review = []
    for api, path in (('ollama', '/api/chat'), ('openai', '/v1/chat/completions')):
        for probe, native, keys in escape_probes(args.predict):
            body = body_for(args.model, api, probe, native, False, args.predict)
            before = worker.escape_serial
            result = await collect(client, path, body, api, emit, probe=probe, stream=False, phase='call')
            require(worker.escape_serial == before + 1, 'Missing or ambiguous native debug response')
            evidence = inspect_escape(body, worker.escape_response, result, keys[0])
            emit('escape-comparison', api=api, probe=probe, stream=False, **evidence)
            if evidence['diagnosis'] != 'exact' or not evidence['semantic_match']:
                review.append(f"{api}:{probe}:{evidence['diagnosis']}")
    return review


async def exercise(args, emit):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_daemon_config(False, discover=False).backend
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    workers = []
    escape_only = getattr(args, 'escape_only', False)
    class EscapeWorker(NativeWorker):
        escape_serial = 0
        escape_response = None

        async def chat(self, payload):
            # verbose only adds raw debug fields; keep parsing and generation
            # grammar enabled, and return the original parsed response as well.
            body = self._body({**payload, 'verbose': True}, chat=True, stream=False)
            response = await super().chat(body)
            self.escape_serial += 1
            self.escape_response = deepcopy(response)
            emit('native-debug-response', serial=self.escape_serial, pid=self.pid, request=body, payload=response)
            return response

    def factory(spec):
        worker_type = EscapeWorker if escape_only else NativeWorker
        worker = worker_type(replace(spec, runtime_options={**spec.runtime_options, 'n_batch': 512, 'n_ubatch': 512}))
        workers.append(worker)
        return worker
    class SmokeManager(ModelManager):
        def _resolve_native_definition_sync(self, name, num_ctx, *, runtime_overrides=None):
            # OpenAI has no context-size request field. Pin every admission in
            # this isolated diagnostic, without changing registered TOML files.
            return super()._resolve_native_definition_sync(name, args.ctx,
                runtime_overrides={**(runtime_overrides or {}), "n_ctx": args.ctx})
    manager = SmokeManager(config, native_worker_factory=factory, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout, 'request_timeout': args.request_timeout})
    main.model_manager = ollama.model_manager = openai.model_manager = manager
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(main.app, log_level='warning', access_log=False,
        lifespan='on', timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    ok, review = True, []
    try:
        await wait_until(lambda: server.started or task.done())
        require(server.started, 'Private TLlama server failed to start')
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}',
                trust_env=False, timeout=args.request_timeout) as client:
            async with manager.acquire_native_model(args.model, num_ctx=args.ctx, keep_alive=-1) as lease:
                worker = lease.worker
                emit('ready', pid=worker.pid, generation=lease.generation, argv=worker.command,
                     fingerprint=worker.runtime_fingerprint, context=worker.props['default_generation_settings']['n_ctx'])
            if escape_only:
                review.extend(await exercise_escapes(args, client, worker, emit))
            for api, path in (() if args.clients_only or escape_only else (('ollama', '/api/chat'), ('openai', '/v1/chat/completions'))):
                cases = reference.probes(args.predict)
                if api == 'ollama':
                    # Ollama has no public tool_choice / parallel switch. Test
                    # its auto behavior instead of pretending those fields work.
                    cases = [('parallel-auto' if c[0] == 'parallel-required' else c[0], c[1], c[2])
                             for c in cases if c[0] in {'single-auto', 'parallel-required', 'escaped-required', 'thinking-auto'}]
                else:
                    cases.append(('named-required', cases[1][1], ['alpha']))
                for probe, native, keys in cases:
                    for stream in (False, True):
                        labels = {'probe': probe, 'stream': stream}
                        body = body_for(args.model, api, probe, native, stream, args.predict)
                        try:
                            result = await collect(client, path, body, api, emit, phase='call', **labels)
                            match = reference.semantics(result, keys)
                            emit('assembled', api=api, semantic_match=match, **result, **labels)
                            if not match:
                                review.append(f'{api}:{probe}:{stream}:semantic_mismatch')
                                continue
                            if keys:
                                history, markers = list(body['messages']) + [result['message']], []
                                for i, (call, value) in enumerate(zip(result['message']['tool_calls'], result['arguments'])):
                                    marker = f'TOOL_RESULT_7f4a_{i}'
                                    markers.append(marker)
                                    history.append({'role': 'tool', 'tool_call_id': call['id'], 'name': call['function']['name'],
                                        'content': json.dumps({'key': value['arguments']['key'], 'marker': marker}, ensure_ascii=False)})
                                follow = {**body, 'messages': history}
                                if api == 'openai':
                                    follow['tool_choice'] = 'none'
                                answer = await collect(client, path, follow, api, emit, phase='followup', **labels)
                                match = not answer['message'].get('tool_calls') and all(m in answer['message']['content'] for m in markers)
                                emit('followup', api=api, semantic_match=match, **answer, **labels)
                                if not match:
                                    review.append(f'{api}:{probe}:{stream}:followup_mismatch')
                        except Exception as exc:
                            ok = False
                            emit('error', api=api, error=f'{type(exc).__name__}: {exc}', **labels)
            if args.clients_only:
                review.extend(await exercise_clients(args, str(client.base_url).rstrip('/'), emit))
            elif not escape_only:
                # Cancel while OpenAI still exposes only partial string arguments.
                native = reference.probes(args.predict)[1][1]
                body = body_for(args.model, 'openai', 'single-required', native, True, 512)
                body['messages'] = [native['messages'][0], {'role': 'user', 'content':
                    'Call lookup_record with this key: ' + json.dumps('cancel_' + 'x' * 1024)}]
                seen = False
                async with client.stream('POST', '/v1/chat/completions', json=body) as response:
                    require(response.status_code == 200, 'Public cancellation request failed')
                    async for line in response.aiter_lines():
                        if line.startswith('data: ') and line[6:] != '[DONE]':
                            payload = json.loads(line[6:])
                            emit('public-event', api='openai', probe='cancel', payload=payload)
                            require('error' not in payload, f'Cancellation stream error: {payload}')
                            for choice in payload.get('choices', []):
                                if any(c.get('function', {}).get('arguments') for c in choice['delta'].get('tool_calls', [])):
                                    seen = choice.get('finish_reason') is None
                            if seen:
                                break
                await wait_until(lambda: not manager.native_snapshot()[0]['leases'])
                emit('stream-cancel', partial_arguments_seen=seen, tool_dispatched=False,
                     native_cancellation_proof='Review cancel task in worker.log; reuse alone is not proof')
                if not seen:
                    review.append('cancel:no_live_arguments')
                body = body_for(args.model, 'openai', 'single-required', native, False, args.predict)
                result = await collect(client, '/v1/chat/completions', body, 'openai', emit, probe='after-cancel')
                match = reference.semantics(result, ['alpha'])
                emit('after-cancel', pid=worker.pid, semantic_match=match, **result)
                if not match:
                    review.append('after-cancel:semantic_mismatch')
            require(len(workers) == 1 and workers[0] is worker, 'Public requests changed the owned worker')
    except Exception as exc:
        ok = False
        emit('error', error=f'{type(exc).__name__}: {exc}')
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=15)
        finally:
            sock.close()
            await manager.shutdown()
        for index, worker in enumerate(workers):
            code = worker.process.returncode if worker.process else None
            removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
            clean = worker.state == 'closed' and removed and code in (None, 0, -signal.SIGTERM)
            ok &= clean
            (args.output / f'{index:02d}-load.log').write_text(worker.load_log, encoding='utf-8')
            emit('worker-closed', pid=worker.pid, returncode=code, directory_removed=removed, clean=clean)
        ok &= not manager.native_snapshot()
    emit('finished', protocol_ok=ok, review_required=bool(review) or not ok,
         observations=review, residency=manager.native_snapshot(), public_api='exercised')
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=reference.MODEL)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clients-only', action='store_true', help='Only actual Python SDK cycles; needs optional openai/ollama packages')
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--predict', type=int, default=256)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=180)
    args = parser.parse_args(argv)
    if args.ctx < 2048 or not 16 <= args.predict < args.ctx:
        parser.error('Need --ctx >= 2048 and 16 <= --predict < --ctx')
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
    with (args.output / 'raw.jsonl').open('w', encoding='utf-8') as raw, (args.output / 'summary.jsonl').open('w', encoding='utf-8') as summary:
        def emit(stage, **values):
            record = {'stage': stage, **values}
            raw.write(json.dumps(record, ensure_ascii=False) + '\n'); raw.flush()
            if stage not in {'request', 'public-event', 'public-response', 'sdk-event'}:
                line = json.dumps({k: v for k, v in record.items() if k != 'argv'}, ensure_ascii=False)
                summary.write(line + '\n'); summary.flush()
                print(line, flush=True)
        return 0 if asyncio.run(exercise(args, emit)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
