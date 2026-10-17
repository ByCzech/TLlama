#!/usr/bin/env python3
"""Exercise public native text APIs over a private loopback TLlama instance."""

import argparse
import asyncio
from dataclasses import replace
import json
import logging
from pathlib import Path
import socket

import httpx
import uvicorn

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.native_memory import buffer_location
from tllama.worker import WorkerError


def report(stage, **values):
    print(json.dumps({'stage': stage, **values}, ensure_ascii=False), flush=True)


async def wait_until(predicate, timeout=15):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


def require(condition, message):
    if not condition:
        raise WorkerError(message)


def public_mtp_stats(payload):
    stats = payload.get('tllama', {}).get('speculative', {})
    drafted, accepted = stats.get('draft_tokens'), stats.get('accepted_tokens')
    require(type(drafted) is int and drafted > 0 and type(accepted) is int
            and 0 <= accepted <= drafted, 'Public MTP drafting was not measured')
    require(stats.get('acceptance') == accepted / drafted, 'Public MTP acceptance disagrees')
    return stats


async def exercise(args):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    class SmokeManager(ModelManager):
        def acquire_native_model(self, model_name, num_ctx=None, keep_alive=None):
            return super().acquire_native_model(model_name, num_ctx=args.ctx if args.ctx is not None else num_ctx,
                                                keep_alive=keep_alive)

        async def preload_native_model(self, model_name, num_ctx=None, keep_alive=None):
            return await super().preload_native_model(model_name, num_ctx=args.ctx if args.ctx is not None else num_ctx,
                                                      keep_alive=keep_alive)

    manager = SmokeManager(config, native_launch_options={
        'executable': args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout': args.load_timeout,
        'request_timeout': args.request_timeout})
    main.model_manager = ollama.model_manager = openai.model_manager = manager
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=0, log_level='warning',
                           access_log=False, lifespan='on', timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    worker = None
    try:
        await wait_until(lambda: server.started or task.done())
        require(server.started, 'Public TLlama server failed to start')
        address = f'http://127.0.0.1:{sock.getsockname()[1]}'
        async with httpx.AsyncClient(base_url=address, trust_env=False,
                                     timeout=httpx.Timeout(args.request_timeout, connect=10)) as client:
            async def post(path, body):
                response = await client.post(path, json=body)
                require(response.status_code == 200, f'{path}: HTTP {response.status_code}: {response.text}')
                return response

            # No model loads for the agreed checkpoint limitations.
            empty_ps = await client.get('/api/ps')
            require(empty_ps.status_code == 200 and empty_ps.json() == {'models':[]}, 'Expected an empty native /api/ps')
            require((await client.post('/api/generate', json={'model': args.model, 'prompt':'x', 'suffix':'x', 'raw':True})).status_code == 400,
                    'Expected explicit raw/suffix incompatibility')
            require(not manager.native_snapshot(), 'Validation unexpectedly loaded a model')

            loaded = (await post('/api/chat', {'model': args.model, 'messages': [], 'keep_alive': -1})).json()
            require(loaded['done_reason'] == 'load', 'Preload did not return load')
            snapshot = manager.native_snapshot()
            require(len(snapshot) == 1, 'Preload did not create exactly one generation')
            generation = snapshot[0]['generation']
            async with manager.acquire_native_model(args.model, keep_alive=-1) as lease:
                worker = lease.worker
                report('ready', pid=worker.pid, generation=generation, public_address=address,
                       fingerprint=worker.props['build_info'], context=worker.props['default_generation_settings']['n_ctx'],
                       residency=snapshot, argv=worker.command)
                if args.require_mtp:
                    require(worker.spec.runtime_options.get('spec_type') == 'draft-mtp',
                            'MTP check needs effective spec_type=draft-mtp; TOML overrides environment')
                    probe = await worker.chat({'messages': [{'role': 'user', 'content':
                        'Write the integers from 1 to 100, separated by commas.'}], 'temperature': 0,
                        'seed': 42, 'max_tokens': 32, 'reasoning_effort': 'none',
                        'chat_template_kwargs': {'enable_thinking': False}})
                    timings = probe.get('timings', {})
                    drafted, accepted = timings.get('draft_n'), timings.get('draft_n_accepted')
                    require(type(drafted) is int and drafted > 0 and type(accepted) is int
                            and 0 <= accepted <= drafted, 'Native MTP drafting was not measured')
                    report('native-mtp-check', response=probe, draft_n=drafted, draft_n_accepted=accepted,
                           acceptance=accepted / drafted)

            def same_generation():
                current = manager.native_snapshot()
                require(len(current) == 1 and current[0]['generation'] == generation,
                        'Public endpoint reloaded a different configuration')

            async def ps(stage, *, state='ready', leases=None):
                response = await client.get('/api/ps')
                require(response.status_code == 200, 'Native /api/ps failed')
                models = response.json()['models']
                require(len(models) == 1, 'Expected one owned native generation in /api/ps')
                model = models[0]
                require(model['generation'] == generation and model['pid'] == worker.pid and model['model'] == args.model,
                        '/api/ps reported a different generation')
                require(model['state'] == state and (leases is None or model['leases'] == leases), '/api/ps state/leases disagree')
                memory = model['memory']
                compatibility = memory['ollama_compatibility']
                require(model['size'] == compatibility['size'] and model['size_vram'] == compatibility['size_vram'],
                        'Processor sizes disagree with their declared basis')
                require((model['size'] is None) == (model['size_vram'] is None), 'Incomplete processor split')
                if memory['host_bytes'] is not None and memory['device_bytes'] is not None:
                    require(memory['host_bytes'] + memory['device_bytes'] == memory['allocated_bytes'], 'Allocation split disagrees')
                if model['size'] is not None:
                    require(model['size_vram'] == memory['device_bytes']
                            and model['size_vram'] <= model['size'] <= memory['allocated_bytes'],
                            '/api/ps processor summary exceeds its allocation report')
                    require(memory['allocated_bytes'] - model['size'] == compatibility['excluded_host_bytes'],
                            'Excluded helpers disagree with the complete footprint')
                if memory['source'] == 'native_memory_api':
                    require(memory['schema_version'] == 1 and memory['precision'] == 'exact_bytes'
                            and memory['scope'] == 'backend_buffers', 'Unexpected native memory provenance')
                elif memory['source'] == 'native_load_log' and memory['status'] == 'known':
                    offload = memory['layer_offload']
                    full = offload is not None and offload['offloaded_layers'] == offload['total_layers']
                    # A full device load with only input/staging buffers on
                    # the host must not look like partial CPU execution.
                    host_work = sum(size for backend, kinds in memory['buffers'].items()
                                    for kind, size in kinds.items() if buffer_location(backend) == 'host'
                                    and kind not in {'model', 'output'}
                                    and not (backend.upper().endswith('_HOST') and kind == 'compute'))
                    if full and memory['runtime_verified'] and not host_work:
                        require(model['size'] == model['size_vram'], 'Full offload is not reported as 100% GPU')
                require(memory['physical_residency_bytes'] is None, 'Buffer capacities were reported as physical residency')
                require(model['context_length'] == worker.props['default_generation_settings']['n_ctx'], 'Effective context disagrees')
                report(stage, models=models)

            await worker.refresh_memory()
            await wait_until(lambda: worker.memory_snapshot()['source'] != 'native_load_log'
                             or worker.memory.snapshot()['reason'] != 'load_log_pending')
            await ps('ps-ready', leases=0)

            messages = [{'role':'user', 'content':'Napiš krátký pozdrav česky.'}]
            chat = {'model': args.model, 'messages': messages, 'stream': False, 'think': False,
                    'options': {'temperature': 0, 'num_predict': 32}}
            body = (await post('/api/chat', chat)).json()
            require(body['done'] and body['message']['content'], 'Ollama chat produced no answer')
            report('ollama-chat', response=body)

            async def stream(path, payload):
                content, reasoning, final, frames = [], [], None, []
                async with client.stream('POST', path, json=payload) as response:
                    require(response.status_code == 200, f'{path}: HTTP {response.status_code}')
                    done = False
                    async for line in response.aiter_lines():
                        if not line or line.startswith(':'):
                            continue
                        if path.startswith('/v1/'):
                            if not line.startswith('data: '):
                                continue
                            line = line[6:]
                            if line == '[DONE]':
                                done = True
                                continue
                        value = json.loads(line)
                        require('error' not in value, f'{path}: {value}')
                        frames.append(value)
                        if path.startswith('/v1/'):
                            for choice in value['choices']:
                                delta = choice.get('delta', {})
                                content.append(delta.get('content') or '')
                                reasoning.append(delta.get('reasoning_content') or '')
                                if choice.get('finish_reason'):
                                    final = value
                        else:
                            message = value.get('message', {})
                            content.append(message.get('content', value.get('response', '')))
                            reasoning.append(message.get('thinking', value.get('thinking', '')))
                            if value.get('done'):
                                done, final = True, value
                    require(done and final is not None, f'{path}: missing public terminal')
                require(''.join(content) or ''.join(reasoning), f'{path}: empty stream')
                report(path.replace('/', '-').strip('-') + '-stream', content=''.join(content),
                       reasoning=''.join(reasoning), final=final)
                same_generation()
                return final, frames

            await stream('/api/chat', {**chat, 'stream': True, 'think': True,
                                      'options': {'temperature': 0, 'num_predict': 64}})
            oa = {'model':args.model, 'messages':messages, 'temperature':0, 'max_tokens':32, 'reasoning_effort':'none'}
            body = (await post('/v1/chat/completions', oa)).json()
            require(body['choices'][0]['message']['content'], 'OpenAI chat produced no answer')
            require(body['usage']['total_tokens'] == body['usage']['prompt_tokens'] + body['usage']['completion_tokens'],
                    'OpenAI usage counts disagree')
            report('openai-chat', response=body)
            await stream('/v1/chat/completions', {**oa, 'stream':True})

            for raw in (False, True):
                generate = {'model':args.model, 'prompt':'Hello' if raw else 'Napiš krátký pozdrav česky.',
                            'raw':raw, 'think':False, 'stream':False,
                            'options':{'temperature':0, 'num_predict':8 if raw else 32}}
                body = (await post('/api/generate', generate)).json()
                require(body['done'] and body['response'], 'Generate produced no answer')
                report('generate-raw' if raw else 'generate-rendered', response=body)
                await stream('/api/generate', {**generate, 'stream':True})

            if args.require_mtp:
                # A longer public request proves the exported counters, rather
                # than inferring activation from the earlier native-only probe.
                prompt = 'Write the integers from 1 to 100, separated by commas.'
                for path, raw in [('/api/chat', False), ('/api/generate', False),
                                  ('/api/generate', True), ('/v1/chat/completions', False)]:
                    if path.startswith('/v1/'):
                        payload = {**oa, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 32}
                    elif path.endswith('chat'):
                        payload = {**chat, 'messages': [{'role': 'user', 'content': prompt}]}
                    else:
                        payload = {'model': args.model, 'prompt': prompt, 'raw': raw, 'think': False,
                                   'options': {'temperature': 0, 'num_predict': 32}}
                    for streaming in (False, True):
                        payload['stream'] = streaming
                        if streaming:
                            if path.startswith('/v1/'):
                                payload['stream_options'] = {'include_usage': True}
                            final, frames = await stream(path, payload)
                            require(sum('tllama' in frame for frame in frames) == 1,
                                    'Public MTP counters must occur once at stream completion')
                        else:
                            final = (await post(path, payload)).json()
                        stats = public_mtp_stats(final)
                        report('public-mtp-check', api=path, raw=raw, stream=streaming,
                               stats=stats, response=final)
                        same_generation()

            schema = {'type':'object', 'properties':{'answer':{'type':'integer', 'enum':[42]}},
                      'required':['answer'], 'additionalProperties':False}
            prompt = [{'role':'user', 'content':'Vrať JSON objekt s jediným klíčem answer a hodnotou 42.'}]
            body = (await post('/api/chat', {**chat, 'messages':prompt, 'format':schema,
                'options':{'temperature':0, 'num_predict':64}})).json()
            require(json.loads(body['message']['content']) == {'answer':42}, 'Ollama schema did not constrain output')
            report('ollama-schema', response=body)
            body = (await post('/v1/chat/completions', {**oa, 'messages':prompt, 'max_tokens':64,
                'response_format':{'type':'json_schema', 'json_schema':{'name':'answer','schema':schema,'strict':True}}})).json()
            require(json.loads(body['choices'][0]['message']['content']) == {'answer':42}, 'OpenAI schema did not constrain output')
            report('openai-schema', response=body)
            same_generation()

            # Close after actual output, rather than the initial assistant role.
            async with client.stream('POST', '/api/chat', json={**chat, 'stream':True,
                'messages':[{'role':'user','content':'Napiš dlouhý souvislý příběh.'}],
                'options':{'temperature':0,'num_predict':512}}) as response:
                require(response.status_code == 200, 'Cancellation stream was refused')
                async for line in response.aiter_lines():
                    if line:
                        value = json.loads(line)
                        require('error' not in value, f'Cancellation stream failed: {value}')
                        if value.get('message', {}).get('content') or value.get('message', {}).get('thinking'):
                            report('stream-cancel', first=value)
                            await ps('ps-during-stream', leases=1)
                            break
                else:
                    raise WorkerError('Cancellation stream produced no actual output')
            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
            body = (await asyncio.wait_for(post('/api/chat', chat), timeout=15)).json()
            require(body['done'], 'Request after stream cancellation failed')
            report('after-stream-cancel', response=body)
            same_generation()
            await ps('ps-after-stream-cancel', leases=0)

            # Hold an admission lease so retirement is observable even on a
            # fast device; /api/ps itself must not acquire or extend a lease.
            async with manager.acquire_native_model(args.model, keep_alive=-1):
                unloaded = (await post('/api/generate', {'model':args.model,'prompt':'','keep_alive':0})).json()
                require(unloaded['done_reason'] == 'unload', 'Unload did not return unload')
                await ps('ps-draining', state='draining', leases=1)
                require(worker.process.returncode is None, 'Retired worker exited with a protected lease')
            await asyncio.wait_for(manager.wait_native_closed(generation), timeout=30)
            require(worker.process.returncode is not None and not worker.runtime_directory.exists(), 'Native process/directory survived unload')
            report('worker-closed', pid=worker.pid, returncode=worker.process.returncode,
                   directory_removed=True, residency=manager.native_snapshot())
            require((await client.get('/api/ps')).json() == {'models':[]}, 'Closed generation survived in /api/ps')
            report('ps-empty', models=[])
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=30)
        finally:
            sock.close()
            await manager.shutdown()
            report('manager-closed', residency=manager.native_snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Registered TLlama reference; context comes from its TOML')
    parser.add_argument('--ctx', type=int, help='Override context for every diagnostic admission (0 = native model context)')
    parser.add_argument('--require-mtp', action='store_true',
                        help='Verify measured drafting through native and public result/stream probes')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    if args.ctx is not None and args.ctx < 0:
        parser.error('--ctx must be nonnegative')
    logging.basicConfig(level=logging.DEBUG if args.log else logging.WARNING, filename=args.log,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    try:
        asyncio.run(exercise(args))
    except (OSError, ValueError, WorkerError, asyncio.TimeoutError) as exc:
        report('error', error=str(exc))
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
