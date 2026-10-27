"""Read-only discovery evidence; not a placement resolver or stable device registry."""

from dataclasses import dataclass
from itertools import islice
import os
import math
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

from tllama.worker_config import WorkerLaunchSpec


# Discovery grammar only; this does not certify inference or physical PCI links.
AUDITED_RUNTIME = 'b11850-0e9433bfd3'
AUDITED_RUNTIME_FINGERPRINTS = frozenset({AUDITED_RUNTIME, 'b11552-3144c167ef'})
VISIBILITY_ENV = frozenset({'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                          'GGML_VK_VISIBLE_DEVICES', 'ONEAPI_DEVICE_SELECTOR', 'SYCL_DEVICE_FILTER',
                          'ZE_AFFINITY_MASK', 'GGML_DISABLE_VULKAN'})
_ROW = re.compile(r'^  (.+?): (.+) \(([0-9]{1,20}) MiB, ([0-9]{1,20}) MiB free\)$')
_BDF = re.compile(r'^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$')


def discovery_environment(inherited, overrides):
    if any(key not in VISIBILITY_ENV for key in overrides):
        raise ValueError('Only documented visibility variables can be overridden')
    # Reuse the worker's actual inherited-environment policy; no model is read
    # and command() is never called with this illustrative placeholder.
    spec = WorkerLaunchSpec(model=Path('/diagnostic/unused.gguf'), device='none', environment=overrides)
    return spec.child_environment(inherited)


@dataclass(frozen=True)
class Probe:
    argv: tuple
    outcome: str
    returncode: int | None
    stdout: bytes
    stderr: bytes
    elapsed_ms: float
    reaped: bool
    error: str | None = None


def run_probe(executable, flag, environment, cwd, timeout=30, output_limit=4 * 1024 * 1024):
    """Bound both pipes together and reap the owned POSIX process on every exit."""
    if flag not in ('--version', '--list-devices'):
        raise ValueError('Only version and device-list probes are allowed')
    if not math.isfinite(timeout) or not 0 < timeout <= 300 or type(output_limit) is not int or not 1024 <= output_limit <= 16 * 1024 * 1024:
        raise ValueError('Invalid probe deadline or byte limit')
    argv = (str(executable), flag)
    started = time.monotonic()
    output = {'stdout': bytearray(), 'stderr': bytearray()}
    process, outcome, error = None, 'exited', None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        deadline = started + timeout
        with selectors.DefaultSelector() as selector:
            for name in output:
                selector.register(getattr(process, name), selectors.EVENT_READ, name)
            while selector.get_map() or process.poll() is None:
                if time.monotonic() >= deadline:
                    outcome = 'timeout'
                    break
                for key, _ in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = output_limit - sum(map(len, output.values()))
                    output[key.data].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        outcome = 'output_limit'
                        break
                if outcome != 'exited':
                    break
    except OSError as exc:
        outcome, error = 'spawn_or_io_error', f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None:
            # A child holding a pipe after its parent exits is also owned by
            # this new process group. Never touch a running TLlama service.
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=1)
                # Also terminate pipe-inheriting descendants after their
                # parent exited. Only the direct child can be reaped here.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except (OSError, subprocess.TimeoutExpired) as exc:
                outcome, error = 'cleanup_failed', f'{type(exc).__name__}: {exc}'
            finally:
                for name in output:
                    getattr(process, name).close()
    return Probe(argv, outcome, process.returncode if process else None, bytes(output['stdout']),
                 bytes(output['stderr']), (time.monotonic() - started) * 1000,
                 process is not None and process.returncode is not None, error)


def runtime_fingerprint(stdout, stderr):
    text = (stdout + b'\n' + stderr).decode('utf-8')
    matches = re.findall(r'^version: [^\r\n]+ \(build ([0-9]+), commit ([0-9a-f]+)\)\r?$', text, re.M)
    if len(matches) != 1:
        return None
    build, commit = matches[0]
    return 'b' + build + '-' + commit


def parse_device_list(data):
    """Parse the audited stdout table, retaining unsupported/drift evidence."""
    text = data.decode('utf-8')
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line == 'Available devices:']
    if len(headers) != 1:
        return {'status': 'unavailable', 'devices': [], 'issues': ['missing_or_duplicate_table_header']}
    devices, issues, seen, empty_marker = [], [], set(), False
    for line in lines[headers[0] + 1:]:
        if not line.strip():
            continue
        if line == '  (none)':
            if empty_marker:
                issues.append('duplicate_empty_marker')
            empty_marker = True
            continue
        match = _ROW.fullmatch(line)
        if match is None:
            issues.append('unrecognized_device_line')
            continue
        selector, description, total, free = match.groups()
        if selector in seen:
            issues.append('duplicate_native_selector:' + selector)
        seen.add(selector)
        hint = re.fullmatch(r'(CUDA|ROCm|Vulkan|SYCL)([0-9]+)', selector)
        total, free = int(total), int(free)
        if free > total:
            issues.append('reported_free_exceeds_total:' + selector)
        devices.append({'observation_id': 'native-row-' + str(len(devices)), 'selector': selector,
                        'description': description, 'backend_hint': hint[1] if hint else None,
                        'backend_index_hint': int(hint[2]) if hint else None,
                        'hint_source': 'selector_spelling' if hint else None, 'device_type': None,
                        'physical_identity': None, 'physical_association': 'unavailable',
                        'kernel_verified': False,
                        'memory': {'total_mib': total, 'free_mib': free, 'precision': 'truncated_mib',
                                   'source': 'native_list_devices', 'usability': 'unverified'}})
    if empty_marker and devices:
        issues.append('empty_marker_with_devices')
    if not devices and not empty_marker:
        issues.append('missing_empty_marker')
    return {'status': 'partial' if issues else 'complete', 'devices': devices, 'issues': issues}


def pci_inventory(root):
    """Independent Linux PCI location evidence; never join by name or index."""
    rows, issues = [], []
    try:
        entries = sorted(islice(root.iterdir(), 4097))
    except OSError as exc:
        return {'status': 'unavailable', 'devices': [], 'issues': [type(exc).__name__]}
    if len(entries) > 4096:
        return {'status': 'unavailable', 'devices': [], 'issues': ['pci_entry_limit']}
    def read(entry, name, pattern):
        try:
            with (entry / name).open('r', encoding='ascii') as file:
                value = file.read(65).strip()
            if len(value) > 64 or re.fullmatch(pattern, value) is None:
                raise ValueError('invalid PCI attribute')
            return value
        except (OSError, UnicodeError, ValueError):
            issues.append(entry.name + ':' + name + ':unavailable_or_invalid')
            return None
    for entry in entries:
        if not _BDF.fullmatch(entry.name):
            continue
        cls = read(entry, 'class', r'0x[0-9a-fA-F]{6}')
        if cls is None or int(cls, 16) >> 16 not in (0x03, 0x12):
            continue
        row = {'pci_bdf': entry.name, 'identity_scope': 'host_pci_location', 'class': cls,
               'native_observation_id': None, 'native_association': 'unavailable'}
        for key in ('vendor', 'device', 'subsystem_vendor', 'subsystem_device'):
            row[key] = read(entry, key, r'0x[0-9a-fA-F]{4}')
        # The linked driver's basename is kernel evidence, not a backend ID.
        driver = entry / 'driver'
        row['driver'] = driver.resolve().name if driver.is_symlink() and driver.exists() else None
        rows.append(row)
    return {'status': 'partial' if issues else 'complete', 'devices': rows, 'issues': issues}
