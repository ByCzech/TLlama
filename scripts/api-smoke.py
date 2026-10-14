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
from tllama.worker import WorkerError
from tllama.worker_config import DEFAULT_EXECUTABLE


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


async def exercise(args):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    manager = ModelManager(config, native_launch_options={
        'executable': args.executable.expanduser().resolve(), 'load_timeout': args.load_timeout,
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
            require((await client.post('/api/generate', json={'model': args.model, 'prompt':'x', 'suffix':'x'})).status_code == 501,
                    'Expected explicit suffix checkpoint 501')
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
                require(memory['status'] == 'known', f'Incomplete text-model memory report: {memory}')
                require(model['size'] == memory['allocated_bytes'] and model['size_vram'] == memory['device_bytes'],
                        '/api/ps summary differs from its allocation report')
                require(memory['host_bytes'] + memory['device_bytes'] == model['size'], 'Allocation split disagrees')
                require(memory['physical_residency_bytes'] is None, 'Load logs were reported as physical residency')
                require(model['context_length'] == worker.props['default_generation_settings']['n_ctx'], 'Effective context disagrees')
                report(stage, models=models)

            await wait_until(lambda: worker.memory.snapshot()['reason'] != 'load_log_pending')
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
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
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
