#!/usr/bin/env python3
"""Verify native vision through public chat/generate APIs on one private TLlama instance."""

import argparse
import asyncio
import base64
from dataclasses import replace
import json
import logging
from pathlib import Path
import socket
import struct
import zlib

import httpx
import uvicorn

from tllama.backend import ModelManager
from tllama.daemon_config import load_daemon_config
from tllama.worker import WorkerError


SCHEMA = {'type':'object', 'properties':{'left':{'type':'string'}, 'right':{'type':'string'}},
          'required':['left','right'], 'additionalProperties':False}
PROMPT = ('Report the color of the left panel and the color of the right panel in the image. '
          'Use English color names. Return a JSON object with keys left and right.')


def color_png(swapped=False):
    width, height = 256, 128
    left, right = (b'\xff\0\0', b'\0\0\xff')
    if swapped:
        left, right = right, left
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = (b'\0' + left * (width//2) + right * (width//2)) * height
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return base64.b64encode(data).decode()


def report(stage, **values):
    print(json.dumps({'stage':stage, **values}, ensure_ascii=False), flush=True)


def require(condition, message):
    if not condition:
        raise WorkerError(message)


async def wait_until(predicate, timeout=15):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


def payload(model, path, image, stream=False):
    message = {'role':'user','content':PROMPT}
    if path == '/api/generate':
        return {'model':model, 'prompt':PROMPT, 'images':[image], 'stream':stream,
                'think':False, 'format':SCHEMA, 'options':{'temperature':0,'num_predict':128}}
    if path == '/api/chat':
        message['images'] = [image]
        return {'model':model, 'messages':[message], 'stream':stream, 'think':False, 'format':SCHEMA,
                'options':{'temperature':0,'num_predict':128}}
    message['content'] = [{'type':'text','text':PROMPT},
                          {'type':'image_url','image_url':{'url':'data:image/png;base64,' + image}}]
    return {'model':model, 'messages':[message], 'stream':stream, 'reasoning_effort':'none',
            'temperature':0, 'max_tokens':128,
            'response_format':{'type':'json_schema','json_schema':{'name':'colors','schema':SCHEMA,'strict':True}}}


def check_colors(text, swapped):
    colors = json.loads(text)
    expected = {'left':'blue' if swapped else 'red', 'right':'red' if swapped else 'blue'}
    require(isinstance(colors, dict) and set(colors) == set(expected), f'Invalid color response: {colors}')
    # The schema accepts arbitrary strings and the prompt names no colors:
    # correctness has to come from the image, not a forced grammar enum.
    require(all(isinstance(colors[key], str) and expected[key] in colors[key].lower() for key in expected),
            f'Image recognition differs: expected {expected}, got {colors}')
    return colors


async def exercise(args):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_daemon_config(False, discover=False).backend
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    manager = ModelManager(config, native_launch_options={'executable':args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout':args.load_timeout, 'request_timeout':args.request_timeout})
    main.model_manager = ollama.model_manager = openai.model_manager = manager
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=0, lifespan='on',
                           access_log=False, log_level='warning', timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    images = [color_png(), color_png(True)]
    worker = None
    try:
        await wait_until(lambda: server.started or task.done())
        require(server.started, 'Public TLlama server failed to start')
        address = f'http://127.0.0.1:{sock.getsockname()[1]}'
        async with httpx.AsyncClient(base_url=address, trust_env=False,
                                     timeout=httpx.Timeout(args.request_timeout, connect=10)) as client:
            async def post(path, data):
                response = await client.post(path, json=data)
                require(response.status_code == 200, f'{path}: HTTP {response.status_code}: {response.text}')
                return response.json()

            show = await post('/api/show', {'model':args.model})
            require('vision' in show['capabilities'], 'The registered definition does not declare vision')
            require(not manager.native_snapshot(), '/api/show unexpectedly loaded a model')
            report('show-without-load', capabilities=show['capabilities'])
            for path in ('/api/chat', '/api/generate'):
                bad = payload(args.model, path, images[0])
                message = bad if path == '/api/generate' else bad['messages'][0]
                message['images'] = ['http://127.0.0.1:9/image.png']
                require((await client.post(path, json=bad)).status_code == 400, 'Remote images were not refused')
            require(not manager.native_snapshot(), 'Invalid input unexpectedly loaded a model')
            loaded = await post('/api/chat', {'model':args.model,'messages':[],'keep_alive':-1})
            require(loaded['done_reason'] == 'load', 'Native preload failed')
            generation = manager.native_snapshot()[0]['generation']
            async with manager.acquire_native_model(args.model, keep_alive=-1) as lease:
                worker = lease.worker
                require(worker.props.get('modalities', {}).get('vision') is True, 'Loaded worker has no vision capability')
                report('ready', generation=generation, pid=worker.pid, public_address=address,
                       fingerprint=worker.props['build_info'], modalities=worker.props['modalities'],
                       requested_context=worker.spec.n_ctx, context=worker.props['default_generation_settings']['n_ctx'],
                       argv=worker.command)

            def same_generation():
                rows = manager.native_snapshot()
                require(len(rows) == 1 and rows[0]['generation'] == generation and rows[0]['pid'] == worker.pid,
                        'Vision endpoint changed the native generation')

            async def stream(path, data, stage, *, colors=None):
                content, reasoning, final, done = [], [], None, False
                async with client.stream('POST', path, json=data) as response:
                    require(response.status_code == 200, f'{path}: HTTP {response.status_code}')
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
                        if path.startswith('/v1/'):
                            for choice in value['choices']:
                                delta = choice.get('delta', {})
                                content.append(delta.get('content') or '')
                                reasoning.append(delta.get('reasoning_content') or '')
                                if choice.get('finish_reason'):
                                    final = value
                        elif path == '/api/generate':
                            require('message' not in value, 'Generate stream returned a chat response')
                            content.append(value.get('response') or '')
                            reasoning.append(value.get('thinking') or '')
                            if value.get('done'):
                                done, final = True, value
                        else:
                            content.append(value.get('message', {}).get('content') or '')
                            reasoning.append(value.get('message', {}).get('thinking') or '')
                            if value.get('done'):
                                done, final = True, value
                require(done and final is not None, 'Missing public vision terminal')
                text = ''.join(content)
                if colors is not None:
                    check_colors(text, colors)
                else:
                    require(text or ''.join(reasoning), 'Empty thinking-enabled vision stream')
                report(stage, content=text, reasoning=''.join(reasoning), final=final)
                same_generation()

            for path, api in (('/api/chat', 'ollama'), ('/v1/chat/completions', 'openai'),
                              ('/api/generate', 'generate')):
                for swapped, image in enumerate(images):
                    data = payload(args.model, path, image)
                    response = await post(path, data)
                    if api == 'generate':
                        require('message' not in response and response.get('done') is True
                                and response.get('done_reason') == 'stop', 'Invalid image GenerateResponse')
                        text = response['response']
                    else:
                        text = response['message']['content'] if api == 'ollama' else response['choices'][0]['message']['content']
                    report(f'{api}-image-{swapped}', colors=check_colors(text, bool(swapped)), response=response)
                    same_generation()
                    await stream(path, {**data,'stream':True}, f'{api}-image-{swapped}-stream', colors=bool(swapped))

            for path, prefix in (('/api/chat', 'vision'), ('/api/generate', 'generate-vision')):
                thinking = payload(args.model, path, images[0], True)
                thinking.pop('format')
                thinking.update(think=True, options={'temperature':0,'num_predict':64})
                await stream(path, thinking, prefix + '-thinking')

                cancel = payload(args.model, path, images[0], True)
                cancel.pop('format')
                if path == '/api/generate':
                    cancel['prompt'] = 'Describe the image in a long, detailed story.'
                else:
                    cancel['messages'][0]['content'] = 'Describe the image in a long, detailed story.'
                cancel['options'] = {'temperature':0,'num_predict':512}
                async with client.stream('POST', path, json=cancel) as response:
                    require(response.status_code == 200, 'Vision cancellation stream was refused')
                    async for line in response.aiter_lines():
                        if line:
                            value = json.loads(line)
                            require('error' not in value, f'Vision cancellation failed: {value}')
                            output = value if path == '/api/generate' else value.get('message', {})
                            if output.get('response') or output.get('content') or output.get('thinking'):
                                report(prefix + '-stream-cancel', first=value)
                                break
                    else:
                        raise WorkerError('No image output before cancellation')
                await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 0)
                response = await asyncio.wait_for(post(path, payload(args.model, path, images[1])), timeout=args.request_timeout)
                text = response['response'] if path == '/api/generate' else response['message']['content']
                report('after-' + prefix + '-cancel', colors=check_colors(text, True))
                same_generation()
            await worker.refresh_memory()
            [process] = (await client.get('/api/ps')).json()['models']
            memory = process['memory']
            require((process['size'] is None) == (process['size_vram'] is None), 'Incomplete processor split')
            if memory['source'] == 'native_load_log':
                require(process['size'] is None and process['size_vram'] is None,
                        'Projector log summary was reported as a complete allocation')
            elif process['size'] is not None:
                require(memory['status'] == 'known' and process['size_vram'] == memory['device_bytes']
                        and process['size_vram'] <= process['size'] <= memory['allocated_bytes'],
                        'Vision processor summary exceeds the native allocation report')
            report('ps-vision', model=process)
            unloaded = await post('/api/chat', {'model':args.model,'messages':[],'keep_alive':0})
            require(unloaded['done_reason'] == 'unload', 'Vision unload failed')
            await asyncio.wait_for(manager.wait_native_closed(generation), timeout=30)
            require(worker.process.returncode is not None and not worker.runtime_directory.exists(), 'Vision resources survived unload')
            report('worker-closed', pid=worker.pid, returncode=worker.process.returncode, directory_removed=True)
            require((await client.get('/api/ps')).json() == {'models':[]}, 'Closed vision generation remained listed')
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
    parser.add_argument('--model', required=True, help='Registered vision reference; context comes from TOML/global configuration')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
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
        report('error', error=str(exc) or type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
