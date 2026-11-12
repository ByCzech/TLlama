#!/usr/bin/env python3
"""Capture one exact target/external-draft pair, sequentially off and on.

Physical GGUF paths only, offline workers, no model-definition edits. Exit 0
means automated protocol/telemetry checks passed, not a model-quality certificate.
Inspect output differences and actual native cancellation in the retained logs.
"""

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import runpy
import signal

from tllama.helpers.gguf_import import local_group
from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_summary, public_memory
from tllama.worker import NativeWorker, WorkerError
from tllama.worker_config import DEFAULT_EXECUTABLE, WorkerLaunchSpec


HERE = Path(__file__).resolve().parent
REFERENCE = runpy.run_path(str(HERE / 'mtp-qualification-smoke.py'))
NATIVE = REFERENCE['MTP']
LIFECYCLE = REFERENCE['LIFECYCLE']


def layers(value):
    return value if value in {'auto', 'all'} else int(value)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', type=Path, required=True, help='Target GGUF or its first shard')
    result.add_argument('--draft-model', type=Path, required=True, help='Compatible DFlash/DFlash2 GGUF or first shard')
    result.add_argument('--device', required=True)
    result.add_argument('--expected-fingerprint', required=True)
    result.add_argument('--executable', type=Path, default=DEFAULT_EXECUTABLE)
    result.add_argument('--output', type=Path, required=True, help='New capture directory; retained on failure')
    result.add_argument('--ctx', type=int, default=4096)
    result.add_argument('--predict', type=int, default=128)
    result.add_argument('--draft-n-max', type=int, default=3)
    result.add_argument('--type-k', default='f16')
    result.add_argument('--type-v', default='f16')
    result.add_argument('--flash-attn', choices=('on', 'off'), default='on')
    result.add_argument('--draft-device')
    result.add_argument('--draft-gpu-layers', type=layers)
    result.add_argument('--draft-type-k')
    result.add_argument('--draft-type-v')
    result.add_argument('--draft-ubatch', type=int)
    result.add_argument('--memory-tensor-diagnostics', type=int, default=0)
    result.add_argument('--load-timeout', type=float, default=600)
    result.add_argument('--request-timeout', type=float, default=600)
    result.add_argument('--transport', choices=('uds', 'tcp'), default='uds')
    result.add_argument('--no-mmap', action='store_true')
    result.add_argument('--dry-run', action='store_true', help='Print argv only; no GGUF reads, output or worker')
    return result


def launch_specs(args):
    if not 16 <= args.predict < args.ctx or args.draft_n_max <= 0:
        raise ValueError('Need 16 <= --predict < --ctx and positive --draft-n-max')
    base = WorkerLaunchSpec(model=args.model.expanduser().resolve(), device=args.device,
        executable=args.executable.expanduser().resolve(), n_ctx=args.ctx,
        n_gpu_layers=0 if args.device == 'none' else -1,
        type_k=args.type_k, type_v=args.type_v, flash_attn=args.flash_attn == 'on',
        transport=args.transport, memory_source='native', memory_compare=True,
        memory_tensor_diagnostics=args.memory_tensor_diagnostics,
        runtime_fingerprint=args.expected_fingerprint, expected_runtime_fingerprint=args.expected_fingerprint,
        version_policy='strict', load_timeout=args.load_timeout, request_timeout=args.request_timeout,
        runtime_options={'spec_type':'none', 'use_mmap':not args.no_mmap, 'use_mlock':False,
                         'n_batch':512, 'n_ubatch':512, 'op_offload':args.device != 'none',
                         'offload_kqv':args.device != 'none'})
    options = {**base.runtime_options, 'spec_type':'draft-dflash', 'draft_n_max':args.draft_n_max}
    for key, value in {'draft_device':args.draft_device, 'draft_n_gpu_layers':args.draft_gpu_layers,
                       'draft_type_k':args.draft_type_k, 'draft_type_v':args.draft_type_v,
                       'draft_n_ubatch':args.draft_ubatch}.items():
        if value is not None:
            options[key] = value
    return {'off':base, 'dflash':replace(base, draft=args.draft_model.expanduser().resolve(), runtime_options=options)}


def weight_evidence(report, active, spec):
    """Count snapshot-local interval unions, never add overlapping role totals."""
    snapshot = report.snapshot()
    totals = {name: report.weight_totals(roles=roles) for name, roles in
              (('main', {'main'}), ('draft', {'draft'}), ('union', {'main', 'draft'}), ('all', None))}
    owners = {role:{owner['id'] for owner in snapshot['owners']
                    if owner['kind'] == 'model' and role in owner['roles']}
              for role in ('main', 'draft')}
    intervals = (snapshot.get('weight_storage') or {}).get('intervals', [])
    selected = owners['main'] | owners['draft']
    independent = sum(item['size_bytes'] for item in intervals
                      if any(rel['owner_id'] in selected for rel in item['relationships']))
    device_types = {device['id']:device['type'] for device in snapshot['devices']}
    def placed(role, cpu):
        # Shared target output/input weights can remain on the target device
        # even when the independent draft weights were explicitly placed on CPU.
        other = selected - owners[role]
        exclusive = [item for item in intervals
                     if any(rel['owner_id'] in owners[role] for rel in item['relationships'])
                     and not any(rel['owner_id'] in other for rel in item['relationships'])]
        assignments = [item['assignment'] for item in exclusive]
        types = [{device_types[key] for key in value['device_ids']} for value in assignments]
        if not assignments or any(value['status'] != 'known' for value in assignments):
            return False
        return all(value == {'cpu'} for value in types) if cpu else any(
            value and value <= {'gpu', 'igpu', 'accel'} for value in types)
    main, draft, union, total = (totals[name] for name in ('main', 'draft', 'union', 'all'))
    checks = {
        'main_payload': bool(owners['main']) and main is not None and main.status == 'known' and (main.payload_bytes or 0) > 0,
        'union_payload': union is not None and union.status == 'known' and union.payload_bytes == independent,
        'all_model_weights_covered': total is not None and total.status == 'known' and total.payload_bytes == independent,
        'main_placement': placed('main', spec.device == 'none'),
    }
    if active:
        checks['independent_draft_owner'] = bool(owners['draft']) and not owners['main'].intersection(owners['draft'])
        checks['draft_payload'] = draft is not None and draft.status == 'known' and (draft.payload_bytes or 0) > 0
        device = spec.runtime_options.get('draft_device', spec.device)
        count = spec.runtime_options.get('draft_n_gpu_layers', 'auto')
        if device == 'none' or count == 0:
            checks['draft_placement'] = placed('draft', True)
        elif count != 'auto':
            checks['draft_placement'] = placed('draft', False)
    else:
        checks['no_draft_owner'] = not any('draft' in owner['roles'] for owner in snapshot['owners'])
        checks['no_draft_payload'] = draft is not None and draft.status == 'known' and draft.payload_bytes == 0
    return {'checks':checks, 'unique_interval_bytes':independent,
            'totals':{name:asdict(value) if value is not None else None for name, value in totals.items()}}


async def observe(worker, capture, stage, active, baseline=None):
    raw = await REFERENCE['RAW_MEMORY'](worker)
    await worker.refresh_memory()
    telemetry = worker.memory_snapshot()
    capture.write('raw.jsonl', stage, pid=worker.pid, raw_native=raw, telemetry=telemetry)
    capture.check(stage + ':http', raw.get('http_status') == 200 and 'error' not in raw)
    selected = (telemetry.get('comparison') or {}).get('native')
    capture.check(stage + ':selected', telemetry.get('source') == 'native_memory_api' and selected is not None)
    payloads = []
    for name, data in (('raw', raw['body']), ('selected', selected['report'])):
        report = NativeMemoryReport(data)
        snapshot, total = report.snapshot(), report.totals()
        evidence = weight_evidence(report, active, worker.spec)
        checks = {'fingerprint':snapshot['runtime_fingerprint'] == worker.spec.expected_runtime_fingerprint,
                  'model_state':snapshot['model_state'] == 'loaded',
                  'inventory_known':snapshot['status'] == 'known' and total.allocated_bytes is not None,
                  'classified':total.host_bytes is not None and total.device_bytes is not None,
                  'extensions_valid':not snapshot.get('extension_errors')}
        if worker.spec.memory_tensor_diagnostics:
            checks['diagnostics_present'] = 'tensor_diagnostics' in snapshot.get('weight_storage', {})
        values = {role:value['payload_bytes'] if value else None for role, value in evidence['totals'].items()}
        if baseline is not None:
            checks['payload_unchanged'] = values == baseline
        capture.check(stage + ':' + name, all(checks.values()) and all(evidence['checks'].values()),
                      checks=checks, weight_evidence=evidence, extension_errors=snapshot.get('extension_errors', []))
        payloads.append(values)
    capture.check(stage + ':payload-parity', payloads[0] == payloads[1])
    display = ollama_memory_summary(public_memory(telemetry))
    capture.check(stage + ':display', display['size'] is not None and display['size_vram'] is not None, display=display)
    return payloads[0]


async def collect(worker, body, stream, record, **labels):
    def generic(metrics):
        return {key:value for key, value in metrics.items() if key != 'mtp_observed'}
    def emit(stage, **values):
        if 'metrics' in values:
            values['metrics'] = generic(values['metrics'])
        record(stage, **values)
    result = await NATIVE['collect'](worker, body, stream, emit, **labels)
    if result['metrics']['cache_n'] != 0:
        raise WorkerError('Cache-disabled request needs measured cache_n=0')
    return {**result, 'metrics':generic(result['metrics'])}


async def exercise(args, specs, capture, files):
    identity = tuple(REFERENCE['file_identity'](path) for path in files)
    capture.write('summary.jsonl', 'input-files', identity=identity,
                  note='Stat identity of complete shard sets, not a full content hash')
    profiles, observations = {}, []
    def compare(label, reference, actual):
        checks = REFERENCE['compare'](reference, actual)
        capture.write('summary.jsonl', label, checks=checks,
                      reference_hash=reference['text_sha256'], actual_hash=actual['text_sha256'])
        if not all(checks.values()):
            observations.append({'stage':label, 'checks':checks})
    for profile, spec in specs.items():
        worker, results, primary = NativeWorker(spec), {}, None
        active = profile == 'dflash'
        def record(stage, **values):
            capture.write('raw.jsonl' if stage in {'request', 'native-event', 'native-response'} else 'summary.jsonl',
                          stage, profile=profile, **values)
        try:
            capture.check(profile + ':same-files', identity == tuple(REFERENCE['file_identity'](path) for path in files))
            await worker.start()
            record('ready', pid=worker.pid, argv=worker.command, fingerprint=worker.runtime_fingerprint, props=worker.props)
            payloads = await observe(worker, capture, profile + ':loaded', active)
            control_body = NATIVE['request']('Reply exactly READY.', 16)
            control = await collect(worker, control_body, False, record, probe='control')
            capture.check(profile + ':control', bool(control['content'].strip()) and control['finish_reason'] == 'stop')
            for probe, prompt in NATIVE['PROMPTS'].items():
                for stream in (False, True):
                    result = await collect(worker, NATIVE['request'](prompt, args.predict), stream, record, probe=probe)
                    capture.check(f'{profile}:{probe}:{stream}:content', bool(result['content'].strip()))
                    results[probe, stream] = result
                compare(profile + ':' + probe + ':stream-comparison', results[probe, False], results[probe, True])
            checks = REFERENCE['activation'](results, active)
            capture.check(profile + ':draft-counters', all(checks.values()), checks=checks,
                          metrics=[{'probe':key[0], 'stream':key[1], **value['metrics']} for key, value in results.items()])
            await observe(worker, capture, profile + ':after-inference', active, payloads)
            pid = worker.pid
            async with worker.chat_stream(NATIVE['request'](NATIVE['CANCEL'], 512)) as events:
                first = await asyncio.wait_for(LIFECYCLE['first_content'](events), args.request_timeout)
                record('cancel-after-content', pid=pid, first=first,
                       native_stop_proof='Review worker.log; transport closure alone does not prove early native task stop')
            recovered = await collect(worker, control_body, False, record, probe='after-cancel')
            capture.check(profile + ':cancel-reuse', worker.pid == pid and bool(recovered['content'].strip())
                          and recovered['finish_reason'] == 'stop')
            compare(profile + ':recovery-comparison', control, recovered)
            await observe(worker, capture, profile + ':after-cancel', active, payloads)
            profiles[profile] = results
        except BaseException as exc:
            primary = exc
            raise
        finally:
            failures = []
            try:
                await worker.close()
                capture.check(profile + ':cleanup', worker.state == 'closed'
                    and (worker.process is None or worker.process.returncode in (0, -signal.SIGTERM))
                    and (worker.runtime_directory is None or not worker.runtime_directory.exists()),
                    returncode=worker.process.returncode if worker.process else None,
                    directory_removed=worker.runtime_directory is None or not worker.runtime_directory.exists())
            except Exception as exc:
                failures.append(str(exc))
            finally:
                (capture.output / (profile + '-load.log')).write_text(worker.load_log, encoding='utf-8')
                try:
                    capture.check(profile + ':files-unchanged', identity == tuple(REFERENCE['file_identity'](path) for path in files))
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                capture.write('summary.jsonl', 'cleanup-error', profile=profile, errors=failures)
                if primary is None:
                    raise WorkerError('; '.join(failures))
    for key, reference in profiles['off'].items():
        compare(f'off-on:{key[0]}:{key[1]}', reference, profiles['dflash'][key])
    return observations


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        specs = launch_specs(args)
        if args.dry_run:
            print(json.dumps({name:spec.command(Path('/diagnostic/key')) for name, spec in specs.items()}))
            return 0
        # Reuse the importer header checks, not its publication/storage actions.
        files = list(dict.fromkeys([*local_group(specs['off'].model),
                                   *local_group(specs['dflash'].draft), specs['off'].executable]))
        for path in files:
            REFERENCE['file_identity'](path)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        print(json.dumps({'error':str(exc)}))
        return 1
    capture = LIFECYCLE['Capture'](output)
    logging.basicConfig(level=logging.DEBUG, filename=output / 'worker.log',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    error, observations = None, []
    try:
        observations = asyncio.run(exercise(args, specs, capture, files))
    except KeyboardInterrupt:
        error = 'Interrupted'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    passed = error is None and bool(capture.checks) and all(item['ok'] for item in capture.checks)
    result = {'ok':passed, 'protocol_ok':passed, 'review_required':True,
              'manual_review':['semantic_quality', 'early_native_task_stop', 'draft_export_provenance'],
              'observations':observations, 'error':error, 'checks':capture.checks,
              'expected_fingerprint':args.expected_fingerprint, 'output':str(output),
              'public_api':'not_exercised', 'scope':'native_protocol_and_telemetry'}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
