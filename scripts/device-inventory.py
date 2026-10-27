#!/usr/bin/env python3
"""Capture native device enumeration and separate Linux PCI evidence without models."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import tempfile
from uuid import uuid4

from tllama.device_inventory import (AUDITED_RUNTIME, AUDITED_RUNTIME_FINGERPRINTS, VISIBILITY_ENV, discovery_environment,
    parse_device_list, pci_inventory, run_probe, runtime_fingerprint)
from tllama.worker_config import DEFAULT_EXECUTABLE
from tllama.runtime_fingerprint import evaluate_runtime_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-fingerprint', help='Capture this exact candidate build; unaudited captures still require review')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE', help='Child visibility override; repeat as needed')
    parser.add_argument('--timeout', type=float, default=30, help='Per-command deadline in seconds, plus bounded cleanup')
    parser.add_argument('--output-limit', type=int, default=4 * 1024 * 1024, help='Combined stdout/stderr byte cap per command')
    parser.add_argument('--pci-root', type=Path, default=Path('/sys/bus/pci/devices'))
    args = parser.parse_args(argv)
    if args.expected_fingerprint is not None and re.fullmatch(r'b[0-9]+-[0-9a-f]+', args.expected_fingerprint) is None:
        parser.error('--expected-fingerprint must use bBUILD-COMMIT format')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 300:
        parser.error('--timeout must be finite and in (0, 300]')
    if not 1024 <= args.output_limit <= 16 * 1024 * 1024:
        parser.error('--output-limit must be 1024..16777216 bytes')
    overrides = {}
    for item in args.env:
        name, separator, value = item.partition('=')
        if not separator or name not in VISIBILITY_ENV or name in overrides or '\0' in value:
            parser.error('--env requires unique documented visibility NAME=VALUE entries')
        overrides[name] = value
    environment = discovery_environment(os.environ, overrides)
    executable = args.executable.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('--output already exists; choose a new directory')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'checkpoint': '0070', 'capture_id': str(uuid4()),
              'captured_at': datetime.now(timezone.utc).isoformat(),
              'mode': 'read_only_discovery', 'executable': str(executable),
              'audited_runtime': AUDITED_RUNTIME, 'runtime_fingerprint': None,
              'audited_runtime_fingerprints': sorted(AUDITED_RUNTIME_FINGERPRINTS),
              'expected_runtime': args.expected_fingerprint or AUDITED_RUNTIME, 'runtime_audited': False,
              'selector_scope': 'this_executable_and_child_environment_at_capture',
              'visibility_environment': {k: environment[k] for k in sorted(VISIBILITY_ENV) if k in environment},
              'explicit_visibility_overrides': overrides, 'environment_fully_fingerprinted': False,
              'models_loaded': False, 'listener_started': False, 'placement_resolver': False,
              'physical_links_complete': False, 'placement_ready': False,
              'cpu_control': {'selector': 'none', 'source': 'worker_launch_policy', 'enumerated': False, 'kernel_verified': False},
              'native_inventory': None, 'pci_inventory': None, 'probes': {}, 'issues': []}
    def identity():
        stat = executable.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    private = None
    try:
        before = identity()
        report['executable_size_bytes'], report['executable_mtime_ns'] = before[2:]
        with tempfile.TemporaryDirectory(prefix='tllama-device-probe-') as private:
            for name, flag in (('version', '--version'), ('devices', '--list-devices')):
                probe = run_probe(executable, flag, environment, private, args.timeout, args.output_limit)
                (output / (name + '.stdout')).write_bytes(probe.stdout)
                (output / (name + '.stderr')).write_bytes(probe.stderr)
                metadata = asdict(probe)
                metadata.pop('stdout'); metadata.pop('stderr')
                report['probes'][name] = metadata
                if probe.outcome != 'exited' or probe.returncode != 0 or not probe.reaped:
                    raise ValueError(name + ': native probe failed')
                if name == 'version':
                    report['runtime_fingerprint'] = runtime_fingerprint(probe.stdout, probe.stderr)
                    report['runtime_audited'] = report['runtime_fingerprint'] in AUDITED_RUNTIME_FINGERPRINTS
                    evaluate_runtime_fingerprint(report['runtime_fingerprint'],
                        expected_fingerprints={report['expected_runtime']}, policy='strict',
                        context='device inventory capture')
                else:
                    report['native_inventory'] = parse_device_list(probe.stdout)
                    if report['native_inventory']['status'] != 'complete':
                        raise ValueError('Native device table is incomplete or ambiguous')
        if before != identity():
            raise ValueError('Native executable changed during capture')
    except (OSError, ValueError, UnicodeError) as exc:
        report['issues'].append(f'{type(exc).__name__}: {exc}')
    finally:
        report['private_directory_removed'] = private is None or not Path(private).exists()
    report['pci_inventory'] = pci_inventory(args.pci_root.expanduser().resolve())
    report['protocol_ok'] = not report['issues']
    report['review_required'] = (not report['protocol_ok'] or not report['runtime_audited']
                                 or report['pci_inventory']['status'] != 'complete')
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    (output / 'inventory.json').write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['protocol_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
