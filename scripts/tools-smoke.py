#!/usr/bin/env python3
"""Capture independent native Tools references for public adapter comparisons.

Only the diagnostic supplies deterministic in-memory tool results. No generated
code is executed. Reference assembly stays independent of production Tools.
"""

import argparse
import asyncio
from dataclasses import replace
import json
import logging
import math
from pathlib import Path
import signal
import time

from tllama.backend import ModelManager
from tllama.daemon_config import load_daemon_config
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError


MODEL = 'unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S'
ESCAPED = 'žluťoučký "kůň"\\folder\nline'
TOOLS = [{'type':'function','function':{'name':name,
    'description':'Retrieve the exact record for one key. Use one call per key.',
    'parameters':{'type':'object','properties':{'key':{'type':'string'}},
                  'required':['key'],'additionalProperties':False}}}
         for name in ('lookup_record','lookup_other')]
SYSTEM = ('Use lookup_record for requested records, never invent records. '
          'Preserve each requested key exactly. After tool results arrive, '
          'reply with their marker strings, without calling more tools.')


def require(ok, message):
    if not ok:
        raise WorkerError(message)


def json_arguments(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON argument key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON argument')
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError('Tool arguments are not an object')
    return value


class Assembly:
    """Independent diagnostic assembler for the inspected native SSE contract.

    The pinned server caches IDs, can emit growing full name revisions and
    sends argument-string deltas keyed by call index. Interleaving is legal. JSON is inspected
    only after the stream ends; incomplete/length-limited calls are not repaired.
    """

    def __init__(self):
        self.calls = {}
        self.content = self.reasoning = ''
        self.finish = None
        self.frames = self.argument_fragments = 0

    def push(self, payload):
        self.frames += 1
        require(self.frames <= 8192, 'Too many diagnostic stream frames')
        choices = payload.get('choices')
        require(isinstance(choices,list) and len(choices) <= 1, 'Expected zero/one native choices')
        if not choices:
            require('usage' in payload or 'timings' in payload, 'Empty choices without statistics')
            return
        require(self.finish is None, 'Choice after native finish_reason')
        choice = choices[0]
        require(isinstance(choice,dict), 'Expected a native choice object')
        require(type(choice.get('index',0)) is int and choice.get('index',0) == 0, 'Unexpected choice index')
        delta = choice.get('delta')
        require(isinstance(delta,dict), 'Expected a native delta object')
        for field, target in (('content','content'),('reasoning_content','reasoning')):
            value = delta.get(field)
            value = '' if value is None else value
            require(isinstance(value,str), 'Non-string native text delta')
            setattr(self,target,getattr(self,target)+value)
        calls = delta.get('tool_calls',[])
        require(isinstance(calls,list), 'Non-array native tool deltas')
        for call in calls:
            require(isinstance(call,dict), 'Non-object native tool delta')
            index = call.get('index')
            require(type(index) is int and 0 <= index < 64, 'Invalid native tool index')
            current = self.calls.setdefault(index,{'id':'','type':'function','name':'','arguments':'','fragments':0,
                                                  'name_revisions':0})
            require(call.get('type','function') == 'function', 'Unsupported native tool type')
            function = call.get('function',{})
            require(isinstance(function,dict), 'Non-object native function delta')
            for key, value in (('id',call.get('id')),('name',function.get('name'))):
                if value is not None:
                    require(isinstance(value,str), 'Non-string native tool identity')
                    if value:
                        if key == 'id':
                            require(not current[key] or current[key] == value, 'Native tool identity changed within one index')
                        else:
                            require(value.startswith(current[key]), 'Native tool name revision is not a growing prefix')
                            current['name_revisions'] += value != current[key]
                        current[key] = value
            if 'arguments' in function:
                value = function['arguments']
                require(isinstance(value,str), 'Native argument delta is not a string')
                current['arguments'] += value
                current['fragments'] += 1
                self.argument_fragments += 1
                require(len(current['arguments'].encode('utf-8')) <= 1024**2, 'Diagnostic arguments exceeded 1 MiB')
        reason = choice.get('finish_reason')
        if reason is not None:
            require(reason in {'stop','length','tool_calls'}, 'Unknown native finish_reason')
            self.finish = reason

    def result(self):
        require(self.finish is not None, 'Native result lacks finish_reason')
        calls, validity, ids = [], [], set()
        for index, call in sorted(self.calls.items()):
            require(call['id'] and call['name'], 'Incomplete native tool identity')
            require(call['id'] not in ids, 'Native call IDs are not unique')
            ids.add(call['id'])
            try:
                args = json_arguments(call['arguments'])
                validity.append({'index':index,'valid':True,'arguments':args,'fragments':call['fragments'],
                                 'name_revisions':call['name_revisions']})
            except (ValueError,RecursionError):
                validity.append({'index':index,'valid':False,'arguments':None,'fragments':call['fragments'],
                                 'name_revisions':call['name_revisions']})
            calls.append({'id':call['id'],'type':'function',
                          'function':{'name':call['name'],'arguments':call['arguments']}})
        message = {'role':'assistant','content':self.content}
        if self.reasoning:
            message['reasoning_content'] = self.reasoning
        if calls:
            message['tool_calls'] = calls
        return {'message':message,'finish_reason':self.finish,'arguments':validity,
                'frames':self.frames,'argument_fragments':self.argument_fragments}


def full_result(payload):
    choices = payload.get('choices')
    require(isinstance(choices,list) and len(choices) == 1, 'Expected one non-stream native choice')
    choice = choices[0]
    require(isinstance(choice,dict), 'Expected a native choice object')
    message = choice.get('message')
    require(isinstance(message,dict), 'Non-object native message')
    assembly = Assembly()
    calls = message.get('tool_calls',[])
    require(isinstance(calls,list) and all(isinstance(c,dict) for c in calls), 'Invalid native call array')
    assembly.push({'choices':[{'index':choice.get('index',0),'finish_reason':choice.get('finish_reason'),
        'delta':{**message,'tool_calls':[{'index':index,**call} for index,call in enumerate(calls)]}}]})
    return assembly.result()


def probes(predict):
    cases = []
    for name, choice, keys, parallel in (
        ('single-auto','auto',['alpha'],False),
        ('single-required','required',['alpha'],False),
        ('none','none',[],False),
        ('parallel-required','required',['alpha','beta'],True),
        ('escaped-required','required',[ESCAPED],False),
        ('thinking-auto','auto',['alpha'],False),
    ):
        prompt = ('Do not call tools. Reply exactly NO_TOOL.' if not keys else
                  'Retrieve these record keys using lookup_record, one call per key: '+json.dumps(keys,ensure_ascii=False))
        body = {'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                'tools':TOOLS,'tool_choice':choice,'parallel_tool_calls':parallel,
                'reasoning_format':'deepseek','temperature':0,'seed':42,'max_tokens':predict,
                'chat_template_kwargs':{'enable_thinking':name == 'thinking-auto'}}
        if name != 'thinking-auto':
            body['reasoning_effort'] = 'none'
        cases.append((name,body,keys))
    return cases


def signatures(result):
    return sorted((call['function']['name'],json.dumps(args['arguments'],sort_keys=True,ensure_ascii=False))
                  for call,args in zip(result['message'].get('tool_calls',[]),result['arguments']))


def semantics(result, keys):
    calls = result['message'].get('tool_calls',[])
    return (all(a['valid'] for a in result['arguments']) and
            signatures(result) == sorted(('lookup_record',json.dumps({'key':key},sort_keys=True,ensure_ascii=False)) for key in keys)
            and result['finish_reason'] == ('tool_calls' if keys else 'stop')
            and (bool(calls) if keys else result['message']['content'].strip() == 'NO_TOOL'))


async def collect(worker, body, stream, emit, **labels):
    emit('request',stream=stream,body=body,**labels)
    started = time.monotonic()
    if not stream:
        payload = await worker.chat(body)
        emit('native-response',stream=False,payload=payload,elapsed_ms=round((time.monotonic()-started)*1000,3),**labels)
        return full_result(payload)
    assembly = Assembly()
    async with worker.chat_stream(body) as events:
        async for payload in events:
            emit('native-event',stream=True,payload=payload,elapsed_ms=round((time.monotonic()-started)*1000,3),**labels)
            assembly.push(payload)
    # chat_stream has separately checked SSE [DONE], not just finish_reason.
    return assembly.result()


def followup(body, result):
    messages, markers = list(body['messages'])+[result['message']], []
    for index, (call,args) in enumerate(zip(result['message']['tool_calls'],result['arguments'])):
        # Fixed local lookup fixture, no arbitrary function dispatch/eval/network.
        marker = f'TOOL_RESULT_7f4a_{index}'
        markers.append(marker)
        messages.append({'role':'tool','tool_call_id':call['id'],'name':call['function']['name'],
                         'content':json.dumps({'key':args['arguments']['key'],'marker':marker},ensure_ascii=False)})
    return {**body,'messages':messages,'tool_choice':'none','parallel_tool_calls':False}, markers


async def exercise_worker(worker, args, emit):
    review, protocol_ok = [], True
    for name, body, keys in probes(args.predict):
        results = {}
        for stream in (False,True):
            try:
                result = await collect(worker,body,stream,emit,probe=name,phase='call')
                match = semantics(result,keys)
                emit('assembled',probe=name,phase='call',stream=stream,semantic_match=match,**result)
                results[stream] = result
                if not match:
                    review.append(f'{name}/{stream}:semantic_mismatch')
                if match and keys:
                    next_body, markers = followup(body,result)
                    answer = await collect(worker,next_body,stream,emit,probe=name,phase='followup')
                    found = not answer['message'].get('tool_calls') and answer['finish_reason'] == 'stop' and all(
                        marker in answer['message']['content'] for marker in markers)
                    emit('assembled',probe=name,phase='followup',stream=stream,semantic_match=found,**answer)
                    if not found:
                        review.append(f'{name}/{stream}:followup_mismatch')
            except WorkerRequestError as exc:
                emit('request-error',probe=name,stream=stream,status=exc.status_code,error=str(exc),payload=exc.payload)
                review.append(f'{name}/{stream}:http_{exc.status_code}')
                protocol_ok &= 400 <= exc.status_code < 500
            except (WorkerError,ValueError,RecursionError) as exc:
                emit('protocol-error',probe=name,stream=stream,error=f'{type(exc).__name__}: {exc}')
                protocol_ok = False
        if len(results) == 2:
            match = signatures(results[False]) == signatures(results[True])
            emit('stream-comparison',probe=name,tool_calls_match=match,
                 content_match=results[False]['message']['content'] == results[True]['message']['content'])
            if not match:
                review.append(f'{name}:stream_difference')
    # Deliberately long arguments permit closing during a call, before dispatch.
    body = probes(args.predict)[1][1]
    body = {**body,'messages':[body['messages'][0],{'role':'user','content':
        'Call lookup_record once with this exact key: '+json.dumps('cancel_'+'x'*1024)}],
        'max_tokens':512}
    seen = False
    try:
        emit('request',probe='cancel',phase='call',stream=True,body=body)
        async with worker.chat_stream(body) as events:
            async for payload in events:
                emit('native-event',probe='cancel',phase='call',stream=True,payload=payload)
                choices = payload.get('choices',[])
                if choices and choices[0].get('delta',{}).get('tool_calls'):
                    calls = choices[0]['delta']['tool_calls']
                    if any(c.get('function',{}).get('arguments') for c in calls):
                        seen = choices[0].get('finish_reason') is None
                        break
        emit('stream-cancel',partial_arguments_seen=seen,tool_dispatched=False,
             native_cancellation_proof='review worker.log; connection closure alone is not proof of native task cancellation')
        if not seen:
            review.append('cancel:no_live_arguments')
        result = await collect(worker,probes(args.predict)[1][1],False,emit,probe='after-cancel',phase='call')
        reused = semantics(result,['alpha'])
        emit('after-cancel',semantic_match=reused,pid=worker.pid,**result)
        if not reused:
            review.append('after-cancel:semantic_mismatch')
    except WorkerRequestError as exc:
        emit('request-error',probe='cancel',status=exc.status_code,error=str(exc),payload=exc.payload)
        review.append(f'cancel:http_{exc.status_code}')
        protocol_ok &= 400 <= exc.status_code < 500
    except (WorkerError,ValueError) as exc:
        emit('protocol-error',probe='cancel',error=f'{type(exc).__name__}: {exc}')
        protocol_ok = False
    return protocol_ok,review


async def exercise(args, emit):
    config = load_daemon_config(False, discover=False).backend
    if args.models_dir:
        config = replace(config,models_dir=str(args.models_dir.expanduser().resolve()))
    all_ok, all_review = True, []
    emit('diagnostic-version',version=1,public_api='not_exercised',native_contract='b11552-3144c167ef',
         order=['Tools','MTP','embeddings','decision'])
    for index, name in enumerate(args.model or [MODEL]):
        workers = []
        def factory(spec):
            options = {**spec.runtime_options,'n_batch':512,'n_ubatch':512}
            worker = NativeWorker(replace(spec,runtime_options=options))
            workers.append(worker)
            return worker
        manager = ModelManager(config,native_worker_factory=factory,native_launch_options={
            'executable':args.executable.expanduser().resolve() if args.executable else None,
            'load_timeout':args.load_timeout,'request_timeout':args.request_timeout})
        ok, review = True, []
        def record(stage, **values):
            emit(stage,model=name,**values)
        try:
            async with manager.acquire_native_model(name,num_ctx=args.ctx,keep_alive=0) as lease:
                worker = lease.worker
                record('ready',pid=worker.pid,generation=lease.generation,fingerprint=worker.runtime_fingerprint,
                       context=worker.props['default_generation_settings']['n_ctx'],argv=worker.command,
                       props=worker.props)
                ok,review = await exercise_worker(worker,args,record)
        except Exception as exc:
            ok = False
            record('error',error=f'{type(exc).__name__}: {exc}')
        finally:
            await manager.shutdown()
            for worker in workers:
                code = worker.process.returncode if worker.process else None
                removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
                clean = worker.state == 'closed' and removed and code in (None,0,-signal.SIGTERM)
                ok &= clean
                record('worker-closed',pid=worker.pid,returncode=code,directory_removed=removed,clean=clean)
                (args.output/f'{index:02d}-load.log').write_text(worker.load_log,encoding='utf-8')
            require(not manager.native_snapshot(),'Tools manager retained a generation after shutdown')
            record('profile-finished',protocol_ok=ok,review_required=bool(review) or not ok,
                   observations=review,residency=manager.native_snapshot())
        all_ok &= ok
        all_review.extend([name+':'+r for r in review] + ([] if ok else [name+':protocol_failure']))
    emit('finished',protocol_ok=all_ok,review_required=bool(all_review),observations=all_review,
         public_api='not_exercised',note='Exit status covers protocol/cleanup; model semantics and native cancellation require review.')
    return all_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',action='append',help='Registered model; repeat for another family')
    parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--executable',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='New directory; existing evidence is never overwritten')
    parser.add_argument('--ctx',type=int,default=4096)
    parser.add_argument('--predict',type=int,default=256)
    parser.add_argument('--load-timeout',type=float,default=600)
    parser.add_argument('--request-timeout',type=float,default=180)
    args = parser.parse_args(argv)
    if args.predict < 16 or args.ctx < 2048 or args.predict >= args.ctx:
        parser.error('Need --predict >= 16, --ctx >= 2048 and --predict < --ctx')
    if any(not math.isfinite(v) or v <= 0 for v in (args.load_timeout,args.request_timeout)):
        parser.error('Timeouts must be finite and positive')
    args.output = args.output.expanduser().resolve()
    try:
        args.output.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error('--output already exists; choose a new directory')
    logging.basicConfig(level=logging.DEBUG,filename=args.output/'worker.log',encoding='utf-8')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    with (args.output/'raw.jsonl').open('w',encoding='utf-8') as raw, (args.output/'summary.jsonl').open('w',encoding='utf-8') as summary:
        def emit(stage, **values):
            record = {'stage':stage,**values}
            raw.write(json.dumps(record,ensure_ascii=False)+'\n');raw.flush()
            if stage not in {'native-event','native-response','request'}:
                concise = {k:v for k,v in record.items() if k not in {'props','argv'}}
                line = json.dumps(concise,ensure_ascii=False)
                summary.write(line+'\n');summary.flush()
                print(line,flush=True)
        try:
            return 0 if asyncio.run(exercise(args,emit)) else 1
        except Exception as exc:
            emit('fatal',error=f'{type(exc).__name__}: {exc}')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
