#!/usr/bin/env python3
"""Exercise public suffix over a private TLlama instance and real native workers."""

import argparse
import asyncio
from dataclasses import replace
import importlib.util
import json
import logging
import math
from pathlib import Path
import socket
import signal

import httpx
import uvicorn

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.native_config import build_native_sampling
from tllama.native_routes import lease_metadata
from tllama.worker import WorkerError

# Keep exactly the reference checkpoint's probes and independent formatter.
_spec = importlib.util.spec_from_file_location('fim_reference', Path(__file__).with_name('fim-smoke.py'))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
MODELS = reference.MODELS[:-1]  # Codestral awaits an accepted output-boundary recipe.
UNSUPPORTED = (reference.MODELS[-1], reference.NEGATIVE)


def require(condition, message):
    if not condition:
        raise WorkerError(message)


async def wait_until(predicate, timeout=15):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


_evidence = None


def emit(stage, **values):
    line = json.dumps({'stage':stage, **values}, ensure_ascii=False)
    print(line, flush=True)
    if _evidence is not None:
        _evidence.write(line + '\n')
        _evidence.flush()


async def exercise(args):
    from tllama import main
    from tllama.routers import ollama, openai
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    manager = ModelManager(config, native_launch_options={
        'executable':args.executable.expanduser().resolve() if args.executable else None,
        'load_timeout':args.load_timeout, 'request_timeout':args.request_timeout})
    main.model_manager = ollama.model_manager = openai.model_manager = manager
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=0,
        log_level='warning', access_log=False, lifespan='on', timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    selected = ([(name,False) for name in args.model or ()] +
                [(name,True) for name in args.unsupported_model or ()])
    if not selected:
        selected = [(name,False) for name in MODELS] + [(name,True) for name in UNSUPPORTED]
    all_ok = True
    semantic_matches = 0
    semantic_total = 0
    try:
        await wait_until(lambda: server.started or task.done())
        require(server.started, 'Private public server did not start')
        address = f'http://127.0.0.1:{sock.getsockname()[1]}'
        async with httpx.AsyncClient(base_url=address, trust_env=False, timeout=args.request_timeout) as client:
            for index, (name, unsupported) in enumerate(selected):
                worker = None
                ok = True
                try:
                    options = {'temperature':0, 'seed':0, 'top_k':0, 'top_p':1, 'min_p':0,
                               'repeat_penalty':1, 'num_predict':args.predict, 'num_ctx':args.ctx}
                    def body(probe, stream=False):
                        return {'model':name, 'prompt':probe['prefix'], 'suffix':probe['suffix'],
                                'stream':stream, 'think':False, 'options':options, 'keep_alive':-1}
                    async def collect(data, expected_tokens=None):
                        response = await client.post('/api/generate', json=data)
                        require(response.status_code == 200, f'HTTP {response.status_code}: {response.text}')
                        values = [json.loads(line) for line in response.text.splitlines()] if data['stream'] else [response.json()]
                        require(values and all('error' not in value for value in values), f'Public error: {values}')
                        require(sum(value.get('done') is True for value in values) == 1 and values[-1].get('done'),
                                'Public stream needs one terminal response')
                        require(all(value.get('model') == name for value in values), 'Wrong public model identity')
                        final = values[-1]
                        if expected_tokens is not None:
                            count = final.get('prompt_eval_count',0) + final.get('prompt_eval_cached_count',0)
                            require(count == expected_tokens, f'Public input count {count} != full prompt {expected_tokens}')
                        return ''.join(value.get('response','') for value in values), final

                    # Admission pins a generation for direct reference comparisons;
                    # every generation request itself still goes through public HTTP.
                    async with manager.acquire_native_model(name, num_ctx=args.ctx, keep_alive=-1) as lease:
                        worker = lease.worker
                        metadata = lease_metadata(lease)
                        emit('ready', model=name, generation=lease.generation, pid=worker.pid,
                             fingerprint=worker.runtime_fingerprint, order=lease.resolved.fim_order,
                             context=worker.props['default_generation_settings']['n_ctx'], argv=worker.command,
                             inventory=metadata.get('fim'), expectation='unsupported' if unsupported else 'supported')
                        if unsupported:
                            for stream in (False,True):
                                response = await client.post('/api/generate', json=body(reference.PROBES['add'], stream))
                                require(response.status_code == 501 and 'FIM' in response.json().get('error',''),
                                        f'Expected unsupported FIM 501, got {response.status_code}: {response.text}')
                                emit('unsupported', model=name, stream=stream, response=response.json())
                            recovery = await client.post('/api/generate', json={
                                'model':name, 'prompt':'Hello', 'raw':True, 'stream':False,
                                'options':{**options,'num_predict':8}})
                            require(recovery.status_code == 200 and recovery.json().get('done'), 'Ordinary completion recovery failed')
                            emit('after-unsupported', model=name, response=recovery.json())
                        else:
                            inv = metadata.get('fim')
                            require(inv, 'No verified generic FIM inventory')
                            inventory = {kind:{'id':inv['tokens'][inv[key]], 'piece':inv[key]}
                                         for kind,key in (('PRE','prefix'),('SUF','suffix'),('MID','middle'))}
                            bos = await reference.framing(worker, inventory)
                            order = {'auto':'psm', 'training_spm':'training-spm', 'native_spm':'spm'}.get(
                                lease.resolved.fim_order, lease.resolved.fim_order)
                            sampling = build_native_sampling(options, metadata, resolved=lease.resolved)
                            sampling.update(cache_prompt=False, preserved_tokens=list(inv['tokens']),
                                stop=list(dict.fromkeys([*sampling.get('stop',[]), *inv['tokens']])))
                            for case, probe in reference.PROBES.items():
                                tokens, counts = await reference.prompt_tokens(worker, inventory, bos,
                                    probe['prefix'], probe['suffix'], order, native=False, batch=lease.resolved.runtime["n_batch"])
                                direct = await worker.completion({**sampling,'prompt':tokens})
                                outputs = []
                                for stream in (False,True):
                                    text, final = await collect(body(probe, stream), len(tokens))
                                    match = reference.semantic_check(text, case)
                                    semantic_matches += match
                                    semantic_total += 1
                                    require(text == direct['content'], 'Public output differs from full direct FIM reference')
                                    require(not any(piece in text for piece in inv['tokens']), 'A FIM boundary leaked to output')
                                    outputs.append(text)
                                    emit('completion', model=name, case=case, stream=stream, content=text,
                                         semantic_match=match, reference_matches=True, counts=counts, final=final)
                                require(outputs[0] == outputs[1], 'Stream and result differ')

                            # This exceeds native /infill's batch-derived windows,
                            # but must fit the effective slot without clipping.
                            long_probe = {'prefix':'# prefix start\n' + '# context\n' * 160 + reference.PREFIX,
                                          'suffix':reference.SUFFIXES['add'] + '# suffix\n' * 80 + '# suffix end\n'}
                            tokens, counts = await reference.prompt_tokens(worker, inventory, bos,
                                long_probe['prefix'], long_probe['suffix'], order, native=False, batch=lease.resolved.runtime["n_batch"])
                            require(len(tokens) < worker.props['default_generation_settings']['n_ctx'],
                                    'Long smoke probe does not fit; select a larger --ctx')
                            for stream in (False,True):
                                text, final = await collect(body(long_probe,stream),len(tokens))
                                emit('long-input',model=name,stream=stream,counts=counts,content=text,final=final)
                            # Cancellation uses an actual public TCP stream. Even
                            # an early natural end is recorded honestly.
                            async with client.stream('POST','/api/generate',json=body(long_probe,True)) as response:
                                require(response.status_code == 200, 'Cancel stream refused')
                                async for line in response.aiter_lines():
                                    if line:
                                        value = json.loads(line)
                                        require('error' not in value, f'Cancel stream error: {value}')
                                        if value.get('response'):
                                            emit('stream-cancel',model=name,first=value,
                                                 terminal_seen=value.get('done',False))
                                            break
                            await wait_until(lambda: manager.native_snapshot()[0]['leases'] == 1)
                            text, final = await collect(body(reference.PROBES['add']))
                            emit('after-cancel',model=name,content=text,final=final)
                        require(len(manager.native_snapshot()) == 1 and manager.native_snapshot()[0]['generation'] == lease.generation,
                                'Public requests changed the admitted generation')
                    await manager.unload_native_model(name)
                    await asyncio.wait_for(manager.wait_native_closed(lease.generation),30)
                except Exception as exc:
                    ok = False
                    emit('error', model=name,error=f'{type(exc).__name__}: {exc}')
                finally:
                    await manager.unload_native_model(name)
                    if worker:
                        await asyncio.wait_for(worker.close(),30)
                        (args.output / f'{index:02d}-load.log').write_text(worker.load_log,encoding='utf-8')
                        clean = worker.process.returncode in (0,-signal.SIGTERM) and not worker.runtime_directory.exists()
                        ok &= clean
                        emit('worker-closed',model=name,pid=worker.pid,returncode=worker.process.returncode,
                             directory_removed=not worker.runtime_directory.exists(),clean=clean)
                    all_ok &= ok
            emit('finished',transport_ok=all_ok,semantic_matches=semantic_matches,semantic_total=semantic_total,
                 note='Semantic checks are narrow and require manual review; protocol/cleanup determines exit status.')
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task,30)
        finally:
            sock.close()
            await manager.shutdown()
            emit('manager-closed',residency=manager.native_snapshot())
    return all_ok


def main(argv=None):
    global _evidence
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append')
    parser.add_argument('--unsupported-model', action='append')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--predict', type=int, default=64)
    parser.add_argument('--load-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ctx <= 0 or args.predict <= 0:
        parser.error('Context and prediction limit must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in (args.load_timeout,args.request_timeout)):
        parser.error('Timeouts must be finite and positive')
    args.output = args.output.expanduser().resolve()
    try:
        args.output.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error('--output already exists; select a new evidence directory')
    logging.basicConfig(level=logging.DEBUG,filename=args.output/'worker.log',encoding='utf-8')
    with (args.output/'results.jsonl').open('w',encoding='utf-8') as evidence:
        _evidence = evidence
        try:
            return 0 if asyncio.run(exercise(args)) else 1
        except Exception as exc:
            emit('fatal',error=f'{type(exc).__name__}: {exc}')
            return 1
        finally:
            _evidence = None


if __name__ == '__main__':
    raise SystemExit(main())
