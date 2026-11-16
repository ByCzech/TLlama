#!/usr/bin/env python3
"""Measure local listing/definition preparation; never start an inference worker."""

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import functools
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


class Measurements:
    def __init__(self, trace=False):
        self.trace = trace
        self.rows = []
        self.current = None
        self.lock = threading.Lock()

    def wrap(self, owner, name, label, *, cache=False):
        from unittest.mock import patch
        original = getattr(owner, name)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            row = self.current
            if row is None:
                return original(*args, **kwargs)
            with self.lock:
                counts = row['operations']
                counts[label] = counts.get(label, 0) + 1
            result = original(*args, **kwargs)
            if cache:
                key = label + ('.miss' if result is None else '.hit')
                with self.lock:
                    counts[key] = counts.get(key, 0) + 1
            return result

        return patch.object(owner, name, counted)

    def instrument(self, backend):
        stack = ExitStack()
        if self.trace:
            try:
                for name, label in (
                    ('parse_model_toml', 'toml.parse'),
                    ('read_gguf_metadata', 'gguf.header_read'),
                    ('save_metadata_cache', 'metadata.save_attempt'),
                    ('save_digest_cache', 'digest.save_attempt'),
                    ('build_model_file_fingerprint', 'file.fingerprint'),
                ):
                    stack.enter_context(self.wrap(backend, name, label))
                for name, label in (('load_metadata_cache', 'metadata.load'),
                                    ('load_digest_cache', 'digest.load')):
                    stack.enter_context(self.wrap(backend, name, label, cache=True))
                stack.enter_context(self.wrap(backend.ModelManager, '_compute_content_sha256', 'gguf.full_hash'))
                stack.enter_context(self.wrap(Path, 'stat', 'path.stat'))
                stack.enter_context(self.wrap(Path, 'resolve', 'path.resolve'))
            except BaseException:
                stack.close()
                raise
        return stack

    async def measure(self, label, call):
        row = {'phase': label}
        if self.trace:
            row['operations'] = {}
        self.current = row
        started = time.perf_counter()
        try:
            result = call()
            if hasattr(result, '__await__'):
                result = await result
            if isinstance(result, list):
                row['models'] = len(result)
            row['ok'] = True
            return result
        except Exception as exc:
            row.update(ok=False, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            row['seconds'] = time.perf_counter() - started
            self.current = None
            self.rows.append(row)


async def collect(args, report, backend):
    from tllama.config import BackendConfig
    from tllama.daemon_config import load_daemon_config

    measurements = Measurements(args.trace)
    report['phases'] = measurements.rows

    def configure():
        config = (load_daemon_config(args.config, discover=False).backend
                  if args.config else BackendConfig())
        return replace(config, models_dir=str(args.models_dir))

    with measurements.instrument(backend):
        config = await measurements.measure('configuration', configure)
        manager = await measurements.measure('manager', lambda: backend.ModelManager(config))
        try:
            for index in range(args.repeat):
                await measurements.measure(f'list:{index + 1}', manager.list_local_models_with_metadata)
            await measurements.measure('inventory', manager.list_local_models)
            for model in args.model:
                for index in range(args.repeat):
                    info = await measurements.measure(f'metadata:{model}:{index + 1}',
                                                      lambda: manager.get_model_metadata(model))
                    if info is None:
                        raise ValueError(f'No metadata for {model!r}')
                    await measurements.measure(f'prepare:{model}:{index + 1}',
                        lambda: manager._resolve_native_definition(model, args.ctx))
            if manager.native_snapshot():
                raise RuntimeError('Unexpected native residency in a local repository measurement')
        finally:
            await manager.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--model', action='append', default=[], help='Main model for metadata/load preparation; repeatable')
    parser.add_argument('--config', type=Path, help='Explicit daemon TOML; no automatic config discovery')
    parser.add_argument('--repeat', type=positive, default=3)
    parser.add_argument('--ctx', type=positive, default=4096)
    parser.add_argument('--trace', action='store_true', help='Count selected operations; timings include instrumentation overhead')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file, never overwritten')
    args = parser.parse_args(argv)
    args.models_dir = Path(os.path.abspath(args.models_dir.expanduser()))
    if not args.models_dir.is_dir():
        parser.error('--models-dir must be an existing repository directory')
    try:
        output = args.output.expanduser().open('x', encoding='utf-8')
    except OSError as exc:
        parser.error(str(exc))
    with output:
        report = {'schema_version': 1, 'instrumented': args.trace,
                  'models_dir': str(args.models_dir), 'models': args.model,
                  'config': str(args.config) if args.config else None,
                  'repeat': args.repeat, 'context': args.ctx, 'python': sys.version,
                  'platform': platform.platform(), 'os_page_cache': 'not_flushed',
                  'scope': 'local_repository_without_worker_or_http'}
        started = time.perf_counter()
        try:
            import tllama.backend as backend
            from tllama.helpers import metadata_cache
            from tllama.version import get_version
            report.update(import_seconds=time.perf_counter() - started,
                          backend_module=backend.__file__, application_version=get_version(),
                          cache_versions={'metadata': metadata_cache.SCHEMA_VERSION,
                                          'digest': metadata_cache.DIGEST_SCHEMA_VERSION,
                                          'envelope': metadata_cache.CACHE_FORMAT_VERSION})
            asyncio.run(collect(args, report, backend))
            report['ok'] = True
        except Exception as exc:
            report.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'ok': report['ok'], 'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
