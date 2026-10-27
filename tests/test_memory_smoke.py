"""Verify diagnostic profiles and saved evidence with a real loopback subprocess."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from dataclasses import replace

import pytest

from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "memory-smoke.py"
module_spec = importlib.util.spec_from_file_location("memory_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(smoke)


@pytest.mark.parametrize('limit', [0, 1, 4096])
def test_tensor_diagnostic_limit_reaches_explicit_argv(tmp_path, capsys, limit):
    args = ['--profile', 'cpu', '--model', str(tmp_path/'m.gguf'),
            '--output', str(tmp_path/'out'), '--dry-run', '--memory-tensor-diagnostics', str(limit)]
    assert smoke.main(args) == 0
    argv = json.loads(capsys.readouterr().out)['argv']
    assert ('--memory-tensor-diagnostics' in argv) == bool(limit)
    if limit:
        assert argv[argv.index('--memory-tensor-diagnostics') + 1] == str(limit)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('limit', ['-1', '4097', 'true', '1.5'])
def test_smoke_rejects_invalid_diagnostic_limit_without_output(tmp_path, limit):
    with pytest.raises(SystemExit):
        smoke.main(['--profile','cpu','--model',str(tmp_path/'m.gguf'),
            '--output',str(tmp_path/'out'),'--dry-run','--memory-tensor-diagnostics',limit])
    assert not (tmp_path/'out').exists()


def test_weight_summary_preserves_nonadditive_diagnostic_meaning(diagnosed):
    summary = smoke.weight_summary(diagnosed)
    assert summary['totals']['payload_bytes'] == 190
    assert summary['tensor_diagnostics']['record_count'] == 1
    assert summary['tensor_diagnostics']['range_count'] == 1
    assert summary['tensor_diagnostics']['name_encoding'] == 'byte_escape'
    assert 'size_bytes' not in summary['tensor_diagnostics']
    assert smoke.weight_summary({}) == {'supported': False}


@pytest.mark.parametrize('invalid', [False, True])
def test_tensor_capture_preserves_raw_and_separate_summary(tmp_path, diagnosed, monkeypatch, capsys, invalid):
    root = tmp_path/'stub'
    root.mkdir()
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    model = tmp_path/'m.gguf'
    model.touch()
    diagnosed['runtime_fingerprint'] = 'b11850-0e9433bfd3'
    for group in diagnosed['placement']['layer_groups']:
        group.update(location='cpu', device_ids=['cpu'])
        if 'assignment' in group:
            group['assignment']['device_ids'] = ['cpu']
    for buffer in diagnosed['buffers']:
        buffer.update(memory_kind='host', device_ids=['cpu'])
    for interval in diagnosed['weight_storage']['intervals']:
        interval['assignment']['device_ids'] = ['cpu']
    diagnostics = diagnosed['weight_storage']['tensor_diagnostics']
    diagnostics['records'][0]['assignment']['device_ids'] = ['cpu']
    # The full size survives a deliberately incomplete range prefix.
    diagnostics.update(status='partial', truncated=True,
        missing_components=[dict(owner_id=None,component='model_weights',layer_index=None,reason='diagnostic_range_limit')])
    diagnostics['records'][0].update(ranges_complete=False, ranges=[])
    if invalid:
        diagnostics['records'][0]['buffer_id'] = 'missing'
    (root/'memory-report').write_text(json.dumps(diagnosed))
    monkeypatch.setenv('STUB_ROOT', str(root))
    monkeypatch.setenv('STUB_BUILD', diagnosed['runtime_fingerprint'])
    worker_class = smoke.NativeWorker
    monkeypatch.setattr(smoke, 'NativeWorker', lambda spec: worker_class(replace(spec,transport='tcp')))
    output = tmp_path/'output'
    code = smoke.main(['--profile','cpu','--model',str(model),'--executable',str(executable),
        '--output',str(output),'--expected-fingerprint',diagnosed['runtime_fingerprint'],
        '--memory-tensor-diagnostics','4096','--weight-storage'])
    assert code == int(invalid), capsys.readouterr()
    raw = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
    concise = [json.loads(line) for line in (output/'summary.jsonl').read_text().splitlines()]
    captures = [r for r in raw if 'raw_native' in r]
    assert len(captures) == 2
    for record in captures:
        assert record['raw_native']['body'] == diagnosed
        summary = record['summary']['weight_storage']
        assert summary['totals']['payload_bytes'] == 190
        assert summary['tensor_diagnostics']['status'] == ('unavailable' if invalid else 'partial')
        assert record['summary']['checks']['tensor_diagnostics_adapter_valid'] is (not invalid)
        assert record['summary']['checks']['weight_storage_known'] is True
        assert record['summary']['checks']['weight_assignment_classified'] is True
    assert not any('raw_native' in r or 'telemetry' in r for r in concise)
    argv = raw[0]['argv']
    assert argv[argv.index('--memory-tensor-diagnostics')+1] == '4096'
    assert raw[-2]['directory_removed'] is True


@pytest.mark.parametrize("profile,layers,load_mode", [
    ("cpu", "0", "mmap"), ("partial", "20", "mmap"),
    ("full", "all", "mmap"),
    ("vision", "all", "mmap"), ("no-mmap", "all", "none"),
])
def test_profiles_are_explicit_and_dry_run_writes_nothing(tmp_path, capsys, monkeypatch, profile, layers, load_mode):
    monkeypatch.setenv("TLLAMA_MEMORY_SOURCE", "log")
    monkeypatch.setenv("TLLAMA_MEMORY_COMPARE", "false")
    monkeypatch.setenv("TLLAMA_RUNTIME_DEVICE", "unrelated-device")
    output = tmp_path / "output"
    args = ["--profile", profile, "--model", str(tmp_path / "model.gguf"),
            "--output", str(output), "--dry-run"]
    if profile == "vision":
        args += ["--mmproj", str(tmp_path / "projector.gguf")]
    assert smoke.main(args) == 0
    command = json.loads(capsys.readouterr().out)["argv"]
    assert command[command.index("--n-gpu-layers") + 1] == layers
    assert command[command.index("--load-mode") + 1] == load_mode
    assert command[command.index("--device") + 1] == ("none" if profile == "cpu" else "Vulkan0")
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--cache-type-k") + 1] == ("q8_0" if profile == "vision" else "q4_0")
    assert ("--no-kv-offload" in command) == (profile == "cpu")
    assert ("--no-op-offload" in command) == (profile == "cpu")
    assert not output.exists()


@pytest.mark.parametrize('profile', ['full', 'vision'])
def test_accelerator_profiles_retain_real_cpu_weights_instead_of_asserting_full_gpu(diagnosed, profile):
    from tllama.native_memory_api import NativeMemoryReport
    from tllama.native_telemetry import MemoryTelemetry
    from test_native_memory import observe
    for group in diagnosed['placement']['layer_groups']:
        if group['component'] == 'repeating_layers':
            group.update(location='accelerator', device_ids=['gpu0'])
    telemetry = MemoryTelemetry(observe(''), compare=True)
    telemetry.accept(NativeMemoryReport(diagnosed))
    summary = smoke.summarize(telemetry.snapshot(), profile)
    assert summary['checks']['accelerator_weights'] is True
    assert summary['display']['size'] == 510 and summary['display']['size_vram'] == 478


@pytest.mark.parametrize('profile', ['full', 'vision', 'no-mmap'])
@pytest.mark.parametrize('placement', ['mixed', 'cpu', 'empty'])
def test_accelerator_requirement_matches_profile_without_changing_payload(diagnosed, profile, placement):
    from tllama.native_memory_api import NativeMemoryReport
    from tllama.native_telemetry import MemoryTelemetry
    from test_native_memory import observe
    groups = diagnosed['placement']['layer_groups']
    for group in groups:
        if group['component'] != 'input_weights':
            group.update(location=placement if placement != 'empty' else 'cpu',
                         device_ids=['cpu', 'gpu0'] if placement == 'mixed' else ['cpu'])
    if placement == 'empty':
        diagnosed['placement']['layer_groups'] = []
    telemetry = MemoryTelemetry(observe(''), compare=True)
    telemetry.accept(NativeMemoryReport(diagnosed))
    summary = smoke.summarize(telemetry.snapshot(), profile, require_weights=True)
    assert summary['checks']['accelerator_weights'] is (placement == 'mixed' and profile != 'no-mmap')
    assert summary['checks']['weight_storage_known'] is True
    assert summary['weight_storage']['totals']['payload_bytes'] == 190
    assert summary['display']['size'] == 510 and summary['display']['size_vram'] == 478


@pytest.mark.parametrize('state,known,classified', [('known',True,True), ('absent',False,False),
    ('partial',False,False), ('invalid',False,False), ('unknown_device',True,False), ('http_failure',False,False)])
def test_requested_weight_qualification_requires_payload_and_assignment(weights, state, known, classified):
    from tllama.native_memory_api import NativeMemoryReport
    from tllama.native_telemetry import MemoryTelemetry
    from test_native_memory import observe
    if state == 'absent':
        del weights['weight_storage']
    elif state == 'partial':
        weights['weight_storage'].update(status='partial', missing_components=[dict(
            owner_id='model', component='other_weights', layer_index=None, reason='unsupported')])
    elif state == 'invalid':
        weights['weight_storage']['intervals'][0]['buffer_id'] = 'missing'
    elif state == 'unknown_device':
        weights['devices'][1]['type'] = 'unknown'
    telemetry = MemoryTelemetry(observe(''), compare=True)
    if state == 'http_failure':
        telemetry.fail('authentication_failed', http_status=401)
    else:
        telemetry.accept(NativeMemoryReport(weights))
    checks = smoke.summarize(telemetry.snapshot(), 'full', require_weights=True)['checks']
    assert checks['weight_storage_known'] is known
    assert checks['weight_assignment_classified'] is classified
    assert 'weight_storage_known' not in smoke.summarize(telemetry.snapshot(), 'full')['checks']


@pytest.mark.parametrize("profile,status,locality", [
    ("cpu", 200, False), ("partial", 200, False), ("vision", 200, False), ("no-mmap", 200, False),
    ("cpu", 404, False),
    ("cpu", 200, True), ("partial", 200, True), ("vision", 200, True), ("no-mmap", 200, True),
    ("cpu", 401, True), ("cpu", 503, True),
    ("cpu", 200, 'malformed'),
])
def test_saved_evidence_inference_and_cleanup(tmp_path, payload, profile, status, locality):
    root = tmp_path / "stub"
    root.mkdir()
    executable = tmp_path / "native-stub"
    executable.write_text(f"#!{sys.executable}\n" + Path(__file__).with_name("native_worker_stub.py").read_text())
    executable.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.touch()
    fingerprint = "b11552-3144c167ef" if locality else "b11552-d1570abd3d"
    payload["runtime_fingerprint"] = fingerprint
    for group in payload["placement"]["layer_groups"]:
        if profile == "cpu":
            group.update(location="cpu", device_ids=["cpu"])
        elif profile != "partial" and group["component"] != "input_weights":
            group.update(location="accelerator", device_ids=["gpu0"])
    if profile == "cpu":
        for buffer in payload["buffers"]:
            buffer.update(memory_kind="host", device_ids=["cpu"])
    if profile == "no-mmap":
        for buffer in payload["buffers"]:
            buffer["backing"] = "allocated"
    if locality:
        # Synthetic allocation span, deliberately larger than buffer capacity.
        # It must not become another memory total. CPU export is unsupported.
        payload['future_extension'] = {'value':42}
        payload['allocation_locality'] = {
            'scope':'backend_buffer_allocations', 'physical_aliasing':'unknown',
            'status':'unavailable' if profile == 'cpu' else 'partial',
            'missing_components':[{'buffer_id':b['id'], 'reason':'unsupported_backend'}
                                  for b in payload['buffers'] if profile == 'cpu' or b['id'] != 'weights'],
            'allocations':[] if profile == 'cpu' else [{'id':'a0','device_ids':['gpu0'],
                'kind':'device','origin':'allocated','native_api':'vkAllocateMemory',
                'size_bytes':2048,'locality':'device_local',
                'native_properties':{'vulkan.memory_type_index':0,'vulkan.heap_index':0,
                                     'vulkan.memory_property_flags':3,'vulkan.heap_flags':1}}],
            'bindings':[] if profile == 'cpu' else [{'buffer_id':'weights','allocation_id':'a0',
                'buffer_offset_bytes':0,'allocation_offset_bytes':None,'size_bytes':1000}],
        }
        for group in payload['placement']['layer_groups']:
            assigned = ['cpu'] if group['location'] == 'cpu' else ['gpu0']
            group['assignment'] = {'status':'known','device_ids':assigned,
                                   'sources':['loader_selection'],'reasons':[]}
        if profile == 'no-mmap':
            payload['buffers'][2].update(memory_kind='host_pinned',device_ids=['gpu0'])
            payload['placement']['layer_groups'][0]['device_ids'] = ['gpu0']
        if locality == 'malformed':
            payload['allocation_locality']['allocations'] = [{'id':'a0','device_ids':None}]
    (root / "memory-report").write_text(json.dumps(payload))
    (root / "memory-status").write_text(str(status))
    output = tmp_path / "results"
    command = [sys.executable, str(SCRIPT), "--profile", profile, "--model", str(model),
               "--executable", str(executable), "--output", str(output)]
    if locality:
        command += ['--locality', '--expected-fingerprint',fingerprint]
    if profile == "vision":
        projector = tmp_path / "projector.gguf"
        projector.touch()
        command += ["--mmproj", str(projector)]
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1] / "src"),
                   "STUB_ROOT": str(root), "STUB_BUILD": fingerprint,
                   "TLLAMA_MEMORY_SOURCE": "log", "TLLAMA_MEMORY_COMPARE": "false"}
    run = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=20)
    valid = status == 200 and locality != 'malformed'
    assert run.returncode == (0 if valid else 1), run.stdout + run.stderr
    raw = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
    concise = [json.loads(line) for line in (output / "summary.jsonl").read_text().splitlines()]
    assert run.stdout == (output / "summary.jsonl").read_text()
    measurements = [record for record in raw if "telemetry" in record]
    assert len(measurements) == (4 if profile == "vision" else 2)
    assert all(record["telemetry"]["mode"] == "auto" for record in measurements)
    assert all("comparison" in record["telemetry"] for record in measurements)
    assert all(record['raw_native']['http_status'] == status for record in measurements)
    assert all(record['raw_native']['body'] == payload for record in measurements)
    assert not any("telemetry" in record for record in concise)
    assert not any('raw_native' in record for record in concise)
    assert next(record for record in raw if record["stage"] == "chat")["response"]["choices"]
    assert raw[-2]["stage"] == "closed" and raw[-2]["directory_removed"] is True
    assert raw[-2]["returncode"] in (0, -signal.SIGTERM)
    assert raw[-1]["ok"] is valid
    if valid:
        summary = measurements[0]["summary"]
        assert all(summary["checks"].values())
        # The shared compute leaf has two owners but contributes its capacity once.
        assert sum(summary["backing_bytes"].values()) == summary["totals"]["allocated_bytes"] == 1350
        assert summary["source"] == "native_memory_api"
        if locality:
            evidence = summary['locality']
            assert evidence['status'] == ('unavailable' if profile == 'cpu' else 'partial')
            assert evidence['allocation_count'] == (0 if profile == 'cpu' else 1)
            assert measurements[0]['telemetry']['measurement']['report']['allocation_locality'] == payload['allocation_locality']
            if profile == 'no-mmap':
                assignment = evidence['assignments'][0]
                assert assignment['device_ids'] == ['gpu0']
                assert assignment['assignment']['device_ids'] == ['cpu']
    elif status != 200:
        assert measurements[0]["summary"]["checks"]["native_inventory_known"] is False
        assert measurements[0]["summary"]["fallback_reason"] == {
            404:'endpoint_not_supported',401:'authentication_failed',503:'not_ready'}[status]
    else:
        # Optional extension failure leaves the independently valid core intact.
        summary = measurements[0]['summary']
        assert summary['checks']['native_inventory_known'] is True
        assert summary['checks']['locality_shape'] is False
        assert summary['totals']['allocated_bytes'] == 1350
    assert (output / "worker.log").is_file()
    # A repeat must preserve the existing results without launching another child.
    before = (root / "launch").read_text()
    repeat = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=10)
    assert repeat.returncode == 1 and (root / "launch").read_text() == before


def test_explicit_diagnostic_fingerprint_mismatch_stops_before_inference(tmp_path, monkeypatch, capsys):
    root = tmp_path/'stub'
    root.mkdir()
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    model = tmp_path/'model.gguf'
    model.touch()
    output = tmp_path/'output'
    monkeypatch.setenv('STUB_ROOT', str(root))
    monkeypatch.setenv('STUB_BUILD', 'different-build')
    monkeypatch.setenv('TLLAMA_RUNTIME_VERSION_POLICY', 'warn')
    worker_class = smoke.NativeWorker
    monkeypatch.setattr(smoke, 'NativeWorker', lambda spec: worker_class(replace(spec, transport='tcp')))
    code = smoke.main(['--profile','cpu','--model',str(model),
        '--executable',str(executable),'--output',str(output),
        '--expected-fingerprint','b11552-3144c167ef', '--load-timeout','5','--request-timeout','5'])
    assert code == 1, capsys.readouterr()
    raw = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
    assert raw[-1]['stage'] == 'error' and 'Unsupported runtime fingerprint' in raw[-1]['error']
    assert any(r['stage'] == 'closed' and r['directory_removed'] for r in raw)
    requests = [json.loads(line) for line in (root/'requests').read_text().splitlines()]
    assert not any(r['path'] in {'/completion','/v1/chat/completions'} for r in requests)


@pytest.mark.parametrize('expected', [None, 'b11552-ec2797836f', 'candidate'])
def test_capture_pin_uses_shared_startup_policy(tmp_path, monkeypatch, expected):
    from argparse import Namespace
    monkeypatch.setenv('TLLAMA_RUNTIME_VERSION_POLICY', 'warn')
    args = Namespace(profile='cpu', model=tmp_path/'model.gguf', mmproj=None, ctx=2048,
        gpu_layers=1, expected_fingerprint=expected, executable=None, device='none',
        type_k=None, type_v=None, flash_attn='on', memory_tensor_diagnostics=0,
        load_timeout=5, request_timeout=5)
    launch = smoke.launch_spec(args)
    assert launch.version_policy == ('strict' if expected is not None else 'warn')
    assert launch.expected_runtime_fingerprint == expected
    if expected is not None:
        assert launch.accepted_runtime_fingerprints == {expected}
