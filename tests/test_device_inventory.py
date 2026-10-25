"""Discovery uses a short-lived native process and never guesses PCI links."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tllama.device_inventory import discovery_environment, parse_device_list, pci_inventory, run_probe, runtime_fingerprint


SCRIPT = Path(__file__).parents[1] / 'scripts/device-inventory.py'
TABLE = b'Available devices:\n  BLAS: OpenBLAS (0 MiB, 0 MiB free)\n  ROCm0: Same card (16384 MiB, 15000 MiB free)\n  Vulkan0: Same card (16384 MiB, 14900 MiB free)\n  Vulkan1: Same card (16384 MiB, 14000 MiB free)\n'
VERSION = b'version: 0.6.0 (build 11552, commit 3144c167ef)\nbuilt with GNU 14 for Linux x86_64\n'


def pci(root, bdf='0000:03:00.0', *, cls='0x030200', device='0x15dd'):
    entry = root / bdf
    entry.mkdir(parents=True)
    for name, value in {'class': cls, 'vendor': '0x1002', 'device': device,
        'subsystem_vendor': '0x1002', 'subsystem_device': '0x1234'}.items():
        (entry / name).write_text(value + '\n')
    return entry


def test_selectors_and_pci_devices_stay_separate_even_with_identical_names(tmp_path):
    table = parse_device_list(TABLE)
    assert table['status'] == 'complete' and table['issues'] == []
    assert [d['selector'] for d in table['devices']] == ['BLAS', 'ROCm0', 'Vulkan0', 'Vulkan1']
    assert [d['backend_hint'] for d in table['devices']] == [None, 'ROCm', 'Vulkan', 'Vulkan']
    assert table['devices'][0]['memory']['total_mib'] == 0  # Not a measured zero usable capacity.
    assert all(d['device_type'] is d['physical_identity'] is None and not d['kernel_verified'] for d in table['devices'])
    pci(tmp_path, '0000:03:00.0')
    pci(tmp_path, '0000:04:00.0')
    linux = pci_inventory(tmp_path)
    assert len(linux['devices']) == 2 and linux['status'] == 'complete'
    assert all(d['native_observation_id'] is None for d in linux['devices'])
    assert len({d['observation_id'] for d in table['devices']}) == 4


@pytest.mark.parametrize('text,status', [
    (b'Available devices:\n  (none)\n', 'complete'),
    (b'Available devices:\n', 'partial'),
    (b'not a table', 'unavailable'),
    (b'Available devices:\nAvailable devices:\n', 'unavailable'),
    (TABLE + b'  Vulkan0: Duplicate (1 MiB, 0 MiB free)\n', 'partial'),
    (TABLE + b'changed format\n', 'partial'),
    (TABLE + b'  (none)\n', 'partial'),
    (b'Available devices:\n  X0: X (1 MiB, 2 MiB free)\n', 'partial'),
])
def test_empty_discovery_and_format_drift_are_explicit(text, status):
    assert parse_device_list(text)['status'] == status


def test_unknown_selectors_and_parentheses_in_descriptions_are_retained():
    d = parse_device_list(b'noise before header\nAvailable devices:\n  RPC[host:123]: Card (revision 2) (12 MiB, 8 MiB free)\n')['devices'][0]
    assert d['selector'] == 'RPC[host:123]' and d['description'] == 'Card (revision 2)'
    assert d['backend_hint'] is None and d['memory']['precision'] == 'truncated_mib'
    with pytest.raises(UnicodeError):
        parse_device_list(b'Available devices:\n\xff')


def test_virtual_and_colon_selectors_are_preserved_literally():
    table = parse_device_list(b'Available devices:\n'
        b'  [virtgpu] CUDA0: Remote GPU (4096 MiB, 4000 MiB free)\n'
        b'  HTP0:1: Virtual HTP (4096 MiB, 4000 MiB free)\n')
    assert table['status'] == 'complete'
    assert [row['selector'] for row in table['devices']] == ['[virtgpu] CUDA0', 'HTP0:1']


def test_pci_evidence_reads_only_display_and_processing_accelerator_attributes(tmp_path):
    a = pci(tmp_path)
    pci(tmp_path, '0000:04:00.0', cls='0x120000')
    pci(tmp_path, '0000:05:00.0', cls='0x020000')
    driver = tmp_path / 'drivers/amdgpu'
    driver.mkdir(parents=True)
    (a / 'driver').symlink_to(driver)
    rows = pci_inventory(tmp_path)
    assert rows['status'] == 'complete' and len(rows['devices']) == 2
    assert rows['devices'][0]['driver'] == 'amdgpu'
    assert all(d['identity_scope'] == 'host_pci_location' for d in rows['devices'])
    (a / 'vendor').unlink()
    (a / 'device').write_text('x' * 100)
    rows = pci_inventory(tmp_path)
    assert rows['status'] == 'partial' and rows['devices'][0]['vendor'] is None and rows['devices'][0]['device'] is None
    assert pci_inventory(tmp_path / 'absent')['status'] == 'unavailable'


def test_pci_entry_limit_bounds_directory_iteration_before_attribute_reads():
    class Root:
        count = 0
        def iterdir(self):
            for _ in range(10000):
                self.count += 1
                yield Path('/unused/0000:03:00.0')
    root = Root()
    result = pci_inventory(root)
    assert root.count == 4097 and result['status'] == 'unavailable' and result['issues'] == ['pci_entry_limit']


def test_environment_matches_worker_scrubbing_and_preserves_visibility_without_global_mutation():
    inherited = {'PATH': '/bin', 'LLAMA_ARG_MODEL': 'download-me', 'LLAMA_API_KEY': 'secret',
        'GGML_BACKEND_PATH': 'unexpected-plugin', 'MTMD_BACKEND_DEVICE': 'wrong', 'AIP_HTTP_PORT': '80',
        'ROCR_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': '1', 'LD_LIBRARY_PATH': '/installed-libs'}
    before = dict(inherited)
    actual = discovery_environment(inherited, {'CUDA_VISIBLE_DEVICES': '2'})
    assert actual == {'PATH': '/bin', 'ROCR_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': '2', 'LD_LIBRARY_PATH': '/installed-libs'}
    assert inherited == before
    with pytest.raises(ValueError):
        discovery_environment({}, {'LLAMA_ARG_MODEL': 'x'})


def test_version_parser_uses_explicit_build_and_commit_not_diagnostic_noise():
    assert runtime_fingerprint(b'', b'CUDA noise\n' + VERSION) == 'b11552-3144c167ef'
    assert runtime_fingerprint(b'build 11552 (3144c167ef)', b'') is None
    assert runtime_fingerprint(VERSION, VERSION) is None


def stub(path, mode='normal', version=VERSION):
    source = f'''#!{sys.executable}
import os,sys,time,signal,json
from pathlib import Path
with Path(os.environ['CAPTURE']).open('a') as file:
    file.write(json.dumps({{'argv':sys.argv[1:],'cwd':os.getcwd(),'pid':os.getpid(),
        'has_model_env':'LLAMA_ARG_MODEL' in os.environ,'has_key':'LLAMA_API_KEY' in os.environ,
        'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')}})+'\\n')
mode={mode!r}
if mode=='hang':
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    time.sleep(30)
if mode=='flood':
    while True:os.write(2,b'x'*65536)
if mode=='pipe-child':
    if os.fork()==0:
        # procfs can be mounted from an ancestor PID namespace.
        Path(os.environ['CAPTURE']+'.child').write_text(os.readlink('/proc/self'))
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        time.sleep(30)
    sys.exit(0)
if '--version' in sys.argv:
    os.write(2,{version!r} if mode!='wrong-build' else b'version: 0.6.0 (build 1, commit abc)\\n')
else:
    if mode=='exit':sys.exit(19)
    os.write(1,{TABLE!r} if mode!='drift' else b'Changed table\\n')
'''
    path.write_text(source)
    path.chmod(0o700)
    return path


@pytest.mark.parametrize('mode,outcome', [('hang', 'timeout'), ('flood', 'output_limit'), ('pipe-child', 'timeout')])
def test_probe_bounds_capture_and_reaps_owned_processes(tmp_path, mode, outcome):
    exe = stub(tmp_path / 'native', mode)
    capture = tmp_path / 'calls'
    started = time.monotonic()
    result = run_probe(exe, '--list-devices', {**os.environ, 'CAPTURE': str(capture)}, tmp_path, timeout=0.2, output_limit=1024)
    assert result.outcome == outcome and result.reaped and result.returncode is not None
    assert len(result.stdout) + len(result.stderr) <= 1024 and time.monotonic() - started < 4
    pid = json.loads(capture.read_text().splitlines()[0])['pid']
    with pytest.raises(ProcessLookupError):os.kill(pid, 0)
    if mode == 'pipe-child':
        child = int(Path(str(capture) + '.child').read_text())
        path = Path(f'/proc/{child}/stat')
        deadline = time.monotonic() + 1
        while path.exists() and path.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not path.exists() or path.read_text().split()[2] == 'Z'


def test_external_cancellation_still_reaps_the_owned_process(monkeypatch, tmp_path):
    from tllama import device_inventory
    processes = []
    popen = subprocess.Popen
    selector = device_inventory.selectors.DefaultSelector
    class InterruptedSelector(selector):
        def select(self, timeout=None):
            raise KeyboardInterrupt()
    def owned(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(device_inventory.selectors, 'DefaultSelector', InterruptedSelector)
    monkeypatch.setattr(device_inventory.subprocess, 'Popen', owned)
    exe = stub(tmp_path / 'native', 'hang')
    with pytest.raises(KeyboardInterrupt):
        run_probe(exe, '--list-devices', {**os.environ, 'CAPTURE': str(tmp_path / 'calls')}, tmp_path)
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize('mode,ok', [('normal', True), ('wrong-build', False), ('drift', False), ('exit', False)])
def test_cli_captures_without_models_listener_or_registered_file_writes(tmp_path, mode, ok):
    exe = stub(tmp_path / 'native', mode)
    root = tmp_path / 'pci'
    pci(root)
    capture = tmp_path / 'calls'
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--executable', str(exe), '--pci-root', str(root),
        '--output', str(output), '--env', 'CUDA_VISIBLE_DEVICES=2']
    inherited = {**os.environ, 'CAPTURE': str(capture), 'LLAMA_ARG_MODEL': 'must-not-load', 'LLAMA_API_KEY': 'secret',
        'PYTHONPATH': str(SCRIPT.parents[1] / 'src')}
    run = subprocess.run(command, env=inherited, capture_output=True, text=True, timeout=8)
    assert run.returncode == (0 if ok else 1), run.stderr
    report = json.loads(run.stdout)
    assert report['protocol_ok'] is ok and report['review_required'] is (not ok)
    assert run.stdout == (output / 'inventory.json').read_text()
    assert not report['models_loaded'] and not report['listener_started'] and not report['placement_resolver']
    assert not report['physical_links_complete'] and not report['placement_ready']
    assert report['private_directory_removed']
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [r['argv'] for r in calls] == ([['--version']] if mode == 'wrong-build' else [['--version'], ['--list-devices']])
    assert all(not r['has_model_env'] and not r['has_key'] and r['cuda'] == '2' for r in calls)
    assert all(not Path(r['cwd']).exists() for r in calls)
    before = capture.read_bytes()
    repeated = subprocess.run(command, env=inherited, capture_output=True, text=True, timeout=5)
    assert repeated.returncode == 2 and capture.read_bytes() == before
    assert 'secret' not in run.stdout


@pytest.mark.parametrize('mode,expected,ok', [('normal', None, False),
    ('normal', 'b11850-0e9433bfd3', True), ('normal', 'b11850-deadbeef', False),
    ('drift', 'b11850-0e9433bfd3', False)])
def test_candidate_capture_requires_exact_opt_in_and_never_claims_audit(tmp_path, mode, expected, ok):
    version = b'version: 0.7.0 (build 11850, commit 0e9433bfd3)\nbuilt for Linux\n'
    exe = stub(tmp_path/'native', mode, version=version)
    root = tmp_path/'pci'
    pci(root)
    output = tmp_path/'capture'
    command = [sys.executable,str(SCRIPT),'--executable',str(exe),'--pci-root',str(root),
               '--output',str(output)]
    if expected is not None:
        command += ['--expected-fingerprint',expected]
    calls = tmp_path/'calls'
    run = subprocess.run(command, capture_output=True, text=True, timeout=8,
        env={**os.environ,'CAPTURE':str(calls),'PYTHONPATH':str(SCRIPT.parents[1]/'src')})
    report = json.loads(run.stdout)
    assert run.returncode == int(not ok), run.stderr
    assert report['protocol_ok'] is ok
    assert report['runtime_fingerprint'] == 'b11850-0e9433bfd3'
    assert report['audited_runtime'] == 'b11552-3144c167ef'
    assert report['expected_runtime'] == (expected or report['audited_runtime'])
    assert report['runtime_audited'] is False and report['review_required'] is True
    assert report['models_loaded'] is report['listener_started'] is False
    assert (output/'version.stderr').read_bytes() == version
    assert not report['placement_ready']
    probes = [json.loads(line)['argv'] for line in calls.read_text().splitlines()]
    assert len(probes) == (2 if expected == 'b11850-0e9433bfd3' else 1)
    if len(probes) == 2:
        assert (output/'devices.stdout').read_bytes() == (TABLE if mode == 'normal' else b'Changed table\n')


@pytest.mark.parametrize('arguments', [['--timeout', 'nan'], ['--output-limit', '0'],
    ['--expected-fingerprint', ''], ['--expected-fingerprint', 'new-runtime'],
    ['--env', 'LLAMA_ARG_MODEL=x'], ['--env', 'CUDA_VISIBLE_DEVICES=1', '--env', 'CUDA_VISIBLE_DEVICES=2']])
def test_invalid_cli_does_not_create_output_or_spawn(tmp_path, arguments):
    output = tmp_path / 'result'
    # Disable site packages so an editable install cannot mask a missing source path.
    run = subprocess.run([sys.executable, '-S', str(SCRIPT), '--output', str(output), *arguments],
        env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')},
        capture_output=True, text=True, timeout=5)
    assert run.returncode == 2 and not output.exists(), run.stderr


@pytest.mark.parametrize('missing', [True, False])
def test_unavailable_pci_evidence_does_not_invent_associations_or_hide_native_capture(tmp_path, missing):
    root = tmp_path / 'pci'
    if not missing:
        entry = pci(root)
        (entry / 'vendor').write_text('malformed')
    exe = stub(tmp_path / 'native')
    run = subprocess.run([sys.executable, str(SCRIPT), '--executable', str(exe), '--pci-root', str(root),
        '--output', str(tmp_path / 'result')], capture_output=True, text=True, timeout=5,
        env={**os.environ, 'CAPTURE': str(tmp_path / 'calls'), 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')})
    report = json.loads(run.stdout)
    assert run.returncode == 0 and report['protocol_ok'] and report['review_required']
    assert report['native_inventory']['status'] == 'complete' and not report['physical_links_complete']
    assert report['pci_inventory']['status'] == ('unavailable' if missing else 'partial')


def test_missing_executable_is_a_saved_failure_without_a_child(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), '--executable', str(tmp_path / 'missing'),
        '--pci-root', str(tmp_path / 'missing-pci'), '--output', str(tmp_path / 'result')],
        capture_output=True, text=True, timeout=5,
        env={**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src')})
    report = json.loads(run.stdout)
    assert run.returncode == 1 and not report['protocol_ok'] and report['review_required']
    assert report['probes'] == {} and report['private_directory_removed']
