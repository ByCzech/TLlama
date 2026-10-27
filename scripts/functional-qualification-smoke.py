#!/usr/bin/env python3
"""Run existing public feature probes against one explicitly selected native build.

Owns a private loopback TLlama instance and workers. Registered definitions are
read only; caches and evidence stay in a new capture directory. No allowlists
are changed. Decision qualification deliberately has its own contract.
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import logging
import math
from pathlib import Path
import runpy
import signal
import sys
import tempfile
import threading
from types import SimpleNamespace

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.daemon_config import load_daemon_config
from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_summary, public_memory
from tllama.worker import NativeWorker, WorkerError


HERE = Path(__file__).resolve().parent
MTP = runpy.run_path(str(HERE / 'mtp-qualification-smoke.py'))
LIFECYCLE = MTP['LIFECYCLE']
PROFILES = {'text': 'api-smoke.py', 'tools': 'tools-api-smoke.py', 'tools-escape': 'tools-api-smoke.py',
            'fim': 'fim-api-smoke.py', 'fim-unsupported': 'fim-api-smoke.py',
            'embeddings': 'embeddings-api-smoke.py'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--profile', choices=PROFILES, required=True)
    result.add_argument('--model', required=True, help='Registered TLlama model reference')
    result.add_argument('--models-dir', type=Path)
    result.add_argument('--device', required=True)
    result.add_argument('--expected-fingerprint', required=True)
    result.add_argument('--executable', type=Path)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--ctx', type=int, help='Default 256 for embeddings, 4096 otherwise')
    result.add_argument('--predict', type=int, default=256)
    result.add_argument('--transport', choices=('uds', 'tcp'), default='uds')
    result.add_argument('--load-timeout', type=float, default=600)
    result.add_argument('--request-timeout', type=float, default=600)
    result.add_argument('--type-k', default='q4_0')
    result.add_argument('--type-v', default='q4_0')
    result.add_argument('--flash-attn', choices=('on', 'off'), default='on')
    result.add_argument('--require-mtp', action='store_true', help='Text profile: verify native and public drafting')
    result.add_argument('--with-sdks', action='store_true', help='Embeddings: additionally run optional OpenAI/Ollama SDKs')
    result.add_argument('--no-mmap', action='store_true', help='Explicit workaround, recorded in every launch')
    result.add_argument('--fim-no-prompt-cache', action='store_true',
                        help='FIM diagnostic only: disable native prompt reuse for reference and public requests')
    result.add_argument('--dry-run', action='store_true', help='Resolve and print launch; no worker or output created')
    return result


def validate(args):
    args.ctx = args.ctx if args.ctx is not None else (256 if args.profile == 'embeddings' else 4096)
    if not args.expected_fingerprint.strip() or not args.device.strip() or not args.model.strip():
        raise ValueError('Model, device and expected fingerprint must be nonempty')
    if args.profile != 'embeddings' and not 16 <= args.predict < args.ctx:
        raise ValueError('Need 16 <= --predict < --ctx')
    if args.profile != 'embeddings' and args.ctx < 2048:
        raise ValueError('Generation profiles need --ctx >= 2048')
    for name in ('load_timeout', 'request_timeout'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError('Timeouts must be finite and positive')
    if args.profile == 'embeddings' and not 64 <= args.ctx <= 16384:
        raise ValueError('Embeddings need --ctx 64..16384')
    if args.require_mtp and args.profile != 'text':
        raise ValueError('--require-mtp is only supported by the text profile')
    if args.with_sdks and args.profile != 'embeddings':
        raise ValueError('--with-sdks is only supported by the embeddings profile')
    if args.fim_no_prompt_cache and args.profile != 'fim':
        raise ValueError('--fim-no-prompt-cache is only supported by the fim profile')
    args.output = args.output.expanduser().resolve()


def module_for(profile):
    spec = importlib.util.spec_from_file_location('qualification_' + profile.replace('-', '_'), HERE / PROFILES[profile])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pin_definition(definition, args):
    resolved = definition.resolved
    if resolved.launch.decision_type is not None:
        raise WorkerError('Decision requires separate qualification')
    if resolved.launch.mmproj is not None:
        raise WorkerError('Use a text-only model definition without a projector for this qualification')
    launch = replace(resolved.launch, runtime_fingerprint=args.expected_fingerprint,
                     version_policy='strict', transport=args.transport,
                     memory_source='native', memory_compare=True, memory_tensor_diagnostics=4096)
    return replace(definition, resolved=replace(resolved, launch=launch))


def qualification_runtime(args, incoming=None):
    embedding = args.profile == 'embeddings'
    return {**(incoming or {}), 'n_ctx': args.ctx, 'device': args.device,
            'n_gpu_layers': 0 if args.device == 'none' else -1,
            'offload_kqv': args.device != 'none', 'op_offload': args.device != 'none',
            'n_batch': max(512, args.ctx) if embedding else 512,
            'n_ubatch': max(512, args.ctx) if embedding else 512,
            'type_k': 'f16' if embedding else args.type_k,
            'type_v': 'f16' if embedding else args.type_v,
            'flash_attn': False if embedding else args.flash_attn == 'on',
            'use_mmap': not args.no_mmap, 'use_mlock': False,
            'spec_type': 'draft-mtp' if args.require_mtp else 'none',
            **({'draft_n_max': 3} if args.require_mtp else {})}


class Capture:
    def __init__(self, args):
        self.args, self.checks, self.events = args, [], []
        self.workers, self.managers, self.identities, self.payloads = [], [], {}, {}

    def emit(self, stage, **values):
        row = {'stage': stage, **values}
        self.events.append(row)
        with (self.args.output / 'raw.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if stage not in {'request', 'native-response', 'native-event', 'public-event', 'public-response',
                         'native-debug-response',
                         'sdk-response', 'sdk-wire-response', 'native-memory', 'tokenization',
                         'truncation-reference', 'qualification-memory'}:
            concise = {k: v for k, v in row.items() if k not in {'argv', 'props', 'metadata'}}
            with (self.args.output / 'summary.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(concise, ensure_ascii=False, allow_nan=False) + '\n')

    def check(self, stage, condition, **details):
        self.checks.append({'stage': stage, 'ok': bool(condition)})
        self.emit(stage, ok=bool(condition), **details)
        if not condition:
            raise WorkerError('Qualification check failed: ' + stage)

    def identity(self, path):
        if path is None:
            return
        path = Path(path).resolve()
        value = MTP['file_identity'](path)
        previous = self.identities.setdefault(path, value)
        self.check('file-identity', previous == value, identity=value)

    async def memory(self, worker, stage):
        raw = await MTP['RAW_MEMORY'](worker)
        await worker.refresh_memory()
        telemetry = worker.memory_snapshot()
        self.emit('qualification-memory', observation=stage, pid=worker.pid,
                  raw_native=raw, telemetry=telemetry)
        self.check(stage + ':http', raw.get('http_status') == 200 and 'error' not in raw)
        native = (telemetry.get('comparison') or {}).get('native')
        self.check(stage + ':selected', telemetry.get('source') == 'native_memory_api' and native is not None)
        evidence = []
        for name, data in (('raw', raw['body']), ('selected', native['report'])):
            report = NativeMemoryReport(data)
            checks = LIFECYCLE['evidence_checks'](report, self.args.expected_fingerprint, 'loaded',
                                                 4096, device=self.args.device)
            weights = MTP['weight_evidence'](report, self.args.require_mtp)
            snap = report.snapshot()
            targets = {d['id'] for d in snap['devices'] if d['name'] == self.args.device}
            if self.args.device != 'none':
                checks['selected_device'] = bool(targets) and any(
                    targets.intersection(item['assignment']['device_ids'])
                    for item in snap['weight_storage']['intervals'])
            payload = {k: v['payload_bytes'] if v else None for k, v in weights['totals'].items()}
            self.check(stage + ':' + name, all(checks.values()) and all(weights['checks'].values()),
                       checks=checks, weight_evidence=weights, extension_errors=snap.get('extension_errors', []))
            evidence.append(payload)
        baseline = self.payloads.setdefault(id(worker), evidence[0])
        self.check(stage + ':payload', evidence[0] == evidence[1] == baseline)
        display = ollama_memory_summary(public_memory(telemetry))
        self.check(stage + ':display', display['size'] is not None and display['size_vram'] is not None,
                   display=display)


def manager_type(args, capture, module):
    class ProbeWorker(NativeWorker):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            self._qualification_close_lock = asyncio.Lock()

        def _body(self, payload, *, chat, stream):
            body = super()._body(payload, chat=chat, stream=stream)
            if args.profile == 'fim' and not chat:
                # Diagnostic intervention at the owned worker boundary. This
                # also covers public requests without exposing a new API option
                # or changing production cache policy.
                if args.fim_no_prompt_cache:
                    body['cache_prompt'] = False
                capture.emit('fim-native-request', pid=self.pid, request=body)
            return body

        async def start(self):
            await super().start()
            self._qualification_started = True
            capture.check('exact-fingerprint', self.runtime_fingerprint == args.expected_fingerprint,
                          observed=self.runtime_fingerprint)
            await capture.memory(self, 'loaded')

        async def close(self):
            # A residency close must not tear down HTTP while another caller
            # is collecting the final snapshot. Serialize the whole operation.
            async with self._qualification_close_lock:
                if self.state == 'ready' and not getattr(self, '_qualification_closing', False):
                    self._qualification_closing = True
                    try:
                        await capture.memory(self, 'before-close')
                    finally:
                        await super().close()
                else:
                    await super().close()

    class QualificationManager(ModelManager):
        def __init__(self, config, **kwargs):
            # Existing public probes can supply a factory for bookkeeping. It
            # still receives the fully pinned definition before constructing a
            # worker, so no post-admission change or allowlist patch is needed.
            original = kwargs.pop('native_worker_factory', None)
            def factory(launch):
                if original is not None:
                    worker = original(launch)
                    if not isinstance(worker, ProbeWorker):
                        raise WorkerError('Public diagnostic did not use the owned qualification worker')
                else:
                    worker = ProbeWorker(launch)
                capture.workers.append(worker)
                return worker
            super().__init__(config, native_worker_factory=factory, **kwargs)
            self.metadata_cache_dir = getattr(args, '_cache_dir', args.output / 'metadata-cache')
            self._qualification_headers = {}
            self._qualification_headers_lock = threading.Lock()
            capture.managers.append(self)

        def _get_model_metadata_sync(self, model_path):
            # The ordinary disk cache is relative to the model repository.
            # An isolated capture elsewhere cannot use that layout. Memoize
            # successful header scans here without writing to the real store;
            # still resolve/check TOML and GGUF identities on every request.
            path = Path(model_path).resolve()
            with self._qualification_headers_lock:
                identity = MTP['file_identity'](path)
                cached = self._qualification_headers.get(path)
                if cached is not None and cached[0] == identity:
                    return deepcopy(cached[1])
                metadata = super()._get_model_metadata_sync(str(path))
                if metadata is not None and MTP['file_identity'](path) == identity:
                    self._qualification_headers[path] = (identity, deepcopy(metadata))
                return metadata

        def _resolve_virtual_model_spec(self, name):
            source = super()._resolve_virtual_model_spec(name)
            if args.profile == 'embeddings':
                from tllama.helpers.model_toml import parse_model_toml
                return parse_model_toml(module.diagnostic_definition(source, args.ctx))
            return source

        def _resolve_native_definition_sync(self, name, num_ctx, *, runtime_overrides=None):
            definition = super()._resolve_native_definition_sync(name, args.ctx,
                runtime_overrides=qualification_runtime(args, runtime_overrides))
            definition = pin_definition(definition, args)
            if not args.dry_run:
                capture.identity(self._toml_path_for_reference(name))
                capture.identity(definition.resolved.launch.model)
                capture.identity(definition.resolved.launch.executable)
                capture.emit('resolved', model=name, runtime=dict(definition.resolved.runtime),
                             fingerprint=definition.resolved.launch.runtime_fingerprint)
            return definition
    QualificationManager.Worker = ProbeWorker
    return QualificationManager


async def exercise(args, module, capture):
    from tllama import main
    from tllama.routers import ollama, openai
    previous = main.model_manager, ollama.model_manager, openai.model_manager
    primary_error = None
    try:
        source = module.ModelManager(module.load_daemon_config().backend)
        capture.identity(source._toml_path_for_reference(args.model))
        if args.profile == 'text':
            module.report = capture.emit
            await module.exercise(args)
        elif args.profile in {'fim', 'fim-unsupported'}:
            module.emit = capture.emit
            args.unsupported_model = [args.model] if args.profile == 'fim-unsupported' else []
            args.model = [] if args.unsupported_model else [args.model]
            passed = await module.exercise(args)
            capture.check('public-protocol', passed)
        elif args.profile in {'tools', 'tools-escape'}:
            args.clients_only = False
            args.escape_only = args.profile == 'tools-escape'
            capture.check('public-protocol', await module.exercise(args, capture.emit))
        else:
            args.atol, args.rtol, args.norm_tolerance, args.cancel_after_ms = 1e-4, 1e-3, 1e-3, 5
            sdks = module.load_sdks() if args.with_sdks else (None, None)
            if not args.with_sdks:
                async def skipped(*_args):
                    capture.emit('sdk-skipped', reason='Optional; rerun with --with-sdks for SDK qualification')
                module.exercise_sdks = skipped
            passed, observations = await module.exercise_model(args, args.model, 'embeddings', sdks, capture.emit)
            capture.emit('feature-review', observations=observations)
            capture.check('public-protocol', passed)
    except BaseException as exc:
        primary_error = exc
        capture.emit('exercise-error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Own cleanup also when an older diagnostic fails before entering its
        # server's try/finally, or when its semantic review asks for inspection.
        failures = []
        try:
            for manager in capture.managers:
                try:
                    await manager.shutdown()
                except Exception as exc:
                    failures.append(exc)
            for index, worker in enumerate(capture.workers):
                try:
                    await worker.close()
                    capture.check('worker-cleanup', worker.state == 'closed'
                        and (worker.process is None or worker.process.returncode is not None)
                        and (worker.runtime_directory is None or not worker.runtime_directory.exists()),
                        pid=worker.pid, returncode=worker.process.returncode if worker.process else None)
                    if getattr(worker, '_qualification_started', False):
                        capture.check('worker-exit', worker.process is not None
                            and worker.process.returncode in (0, -signal.SIGTERM),
                            pid=worker.pid, returncode=worker.process.returncode if worker.process else None)
                except Exception as exc:
                    failures.append(exc)
                finally:
                    (args.output / f'qualification-{index}-load.log').write_text(worker.load_log, encoding='utf-8')
            for path in list(capture.identities):
                try:
                    capture.identity(path)
                except Exception as exc:
                    failures.append(exc)
            if failures:
                details = '; '.join(f'{type(exc).__name__}: {exc}' for exc in failures)
                capture.emit('cleanup-error', error=details)
                if primary_error is None:
                    raise WorkerError(details)
        finally:
            main.model_manager, ollama.model_manager, openai.model_manager = previous


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate(args)
        # Read only storage/executable defaults; do not inherit daemon runtime
        # or sampling overrides into this isolated, explicit qualification.
        defaults = load_daemon_config(False, discover=False).backend
        config = BackendConfig(models_dir=str((args.models_dir or Path(defaults.models_dir)).expanduser().resolve()),
            runtime_executable=(args.executable or defaults.runtime_executable).expanduser().resolve(),
            runtime_version_policy='strict', keep_alive=-1, memory_source='native', memory_compare=True)
        args.models_dir, args.executable = Path(config.models_dir), config.runtime_executable
        capture = Capture(args)
        module = module_for(args.profile)
        module.ModelManager = manager_type(args, capture, module)
        module.NativeWorker = module.ModelManager.Worker
        if args.profile == 'embeddings':
            # DiagnosticManager was declared when the reference was imported;
            # include the exact-build resolver in its ordinary super() chain.
            module.DiagnosticManager = type('EmbeddingQualificationManager',
                (module.DiagnosticManager, module.ModelManager), {})
        module.load_daemon_config = lambda *a, **k: SimpleNamespace(backend=config)
        if args.dry_run:
            with tempfile.TemporaryDirectory(prefix='tllama-qualification-preview-') as cache:
                args._cache_dir = Path(cache)
                manager = module.ModelManager(config)
                definition = manager._resolve_native_definition_sync(args.model, args.ctx)
                print(json.dumps({'profile': args.profile, 'expected_fingerprint': args.expected_fingerprint,
                    'fim_no_prompt_cache': args.fim_no_prompt_cache,
                    'runtime': dict(definition.resolved.runtime),
                    'argv': definition.resolved.launch.command(Path('/diagnostic/key'))}))
            return 0
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, WorkerError) as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
        return 1
    logging.basicConfig(level=logging.DEBUG, filename=args.output / 'worker.log',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    error = None
    try:
        asyncio.run(exercise(args, module, capture))
    except KeyboardInterrupt:
        error = 'Interrupted'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    observations = [note for row in capture.events for note in row.get('observations', [])]
    if args.profile in {'fim', 'fim-unsupported'}:
        observations += [f"fim:{row.get('case')}:semantic_mismatch" for row in capture.events
                         if row.get('semantic_match') is False]
    failed_events = any(row.get('protocol_ok') is False or row.get('transport_ok') is False
                        or row.get('clean') is False or row['stage'] in {'error', 'fatal'} for row in capture.events)
    protocol_ok = (error is None and bool(capture.workers) and bool(capture.payloads) and not failed_events
                   and all(check['ok'] for check in capture.checks))
    result = {'ok': protocol_ok and not observations, 'protocol_ok': protocol_ok,
              'review_required': not protocol_ok or bool(observations), 'observations': observations,
              'error': error, 'checks': capture.checks, 'profile': args.profile,
              'expected_fingerprint': args.expected_fingerprint, 'device': args.device,
              'fim_no_prompt_cache': args.fim_no_prompt_cache,
              'mmap': not args.no_mmap, 'public_api': 'exercised' if protocol_ok else 'see_raw_evidence',
              'sdks': args.with_sdks, 'output': str(args.output)}
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
