"""DFlash evidence capture against synthetic native TCP processes, no GPU claim."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from tllama.native_memory_api import NativeMemoryReport
from test_model_create import gguf_file
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights, interval, relationship


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'dflash-qualification-smoke.py'
smoke = runpy.run_path(str(SCRIPT))
BUILD = 'b11850-0e9433bfd3'


def arguments(tmp_path, *extra):
    return ['--model', str(tmp_path / 'model.gguf'), '--draft-model', str(tmp_path / 'draft.gguf'),
            '--device', 'none', '--expected-fingerprint', BUILD, '--output', str(tmp_path / 'capture'), *extra]


def test_dry_run_does_not_read_gguf_create_output_or_inherit_native_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('TLLAMA_RUNTIME_SPEC_TYPE', 'draft-mtp')
    monkeypatch.setenv('LLAMA_ARG_SPEC_DRAFT_MODEL', '/wrong/model.gguf')
    assert smoke['main'](arguments(tmp_path, '--dry-run', '--no-mmap', '--draft-n-max', '7',
        '--draft-device', 'none', '--draft-gpu-layers', 'auto', '--draft-type-k', 'q8_0', '--draft-ubatch', '128')) == 0
    rows = json.loads(capsys.readouterr().out)
    assert set(rows) == {'off', 'dflash'}
    for profile, command in rows.items():
        assert command.count('--spec-type') == 1
        assert command[command.index('--spec-type') + 1] == ('none' if profile == 'off' else 'draft-dflash')
        assert command[command.index('--load-mode') + 1] == 'none'
        assert '--offline' in command and '--draft' not in command
    assert '--spec-draft-model' not in rows['off'] and '--spec-draft-n-max' not in rows['off']
    assert '--spec-draft-type-k' not in rows['off'] and '--spec-draft-device' not in rows['off']
    assert rows['dflash'][rows['dflash'].index('--spec-draft-n-max') + 1] == '7'
    assert rows['dflash'][rows['dflash'].index('--spec-draft-ngl') + 1] == 'auto'
    assert rows['dflash'][rows['dflash'].index('--spec-draft-n-ubatch') + 1] == '128'
    assert not (tmp_path/'capture').exists()
    specs = smoke['launch_specs'](smoke['parser']().parse_args(arguments(tmp_path)))
    assert all(spec.version_policy == 'strict' and spec.expected_runtime_fingerprint == BUILD for spec in specs.values())


@pytest.mark.parametrize('extra', [['--draft-n-max','0'], ['--draft-n-max','2147483648'],
    ['--predict','15'], ['--predict','4096'], ['--ctx','0'], ['--expected-fingerprint',' '],
    ['--request-timeout','nan'], ['--load-timeout','0'], ['--draft-ubatch','-1'],
    ['--draft-type-v','q4_k'], ['--draft-gpu-layers','-2'],
    ['--draft-device','none','--draft-gpu-layers','all'], ['--memory-tensor-diagnostics','4097']])
def test_bad_settings_fail_without_output_or_worker(tmp_path, extra):
    assert smoke['main'](arguments(tmp_path, *extra)) != 0
    assert not (tmp_path/'capture').exists()


@pytest.fixture
def draft_inventory(weights):
    weights['runtime_fingerprint'] = BUILD
    weights['owners'][0]['roles'] = ['main']
    weights['owners'][2].update(id='draft-context', roles=['draft'])
    for buffer in weights['buffers']:
        for rel in buffer['owners']:
            if rel['owner_id'] == 'mtp':
                rel['owner_id'] = 'draft-context'
    weights['owners'].append({'id':'draft-model', 'kind':'model', 'roles':['draft']})
    # All CPU for this synthetic subprocess; sharing is a relationship, not
    # another allocation. Distinct draft storage is counted independently.
    for part in weights['weight_storage']['intervals']:
        part['assignment']['device_ids'] = ['cpu']
    weights['buffers'][0]['owners'].append({'owner_id':'draft-model', 'component':'model_weights'})
    weights['weight_storage']['intervals'][0]['relationships'].append(relationship('output_weights', owner='draft-model'))
    draft = deepcopy(weights['buffers'][0])
    draft.update(id='draft-weights', owners=[{'owner_id':'draft-model','component':'model_weights'}],
                 memory_kind='host', device_ids=['cpu'])
    weights['buffers'].append(draft)
    weights['weight_storage']['intervals'].append(interval('draft-payload','draft-weights',64,34,
        relationship('repeating_layers',0,'draft-model'), devices=('cpu',)))
    return weights


def test_shared_role_payload_is_a_union_and_external_draft_needs_independent_owner(tmp_path, draft_inventory):
    specs = smoke['launch_specs'](smoke['parser']().parse_args(arguments(tmp_path)))
    result = smoke['weight_evidence'](NativeMemoryReport(draft_inventory), True, specs['dflash'])
    assert all(result['checks'].values())
    assert result['unique_interval_bytes'] == result['totals']['all']['payload_bytes'] == 224
    assert result['totals']['main']['payload_bytes'] == 190 and result['totals']['draft']['payload_bytes'] == 68
    assert not all(smoke['weight_evidence'](NativeMemoryReport(draft_inventory), False, specs['off'])['checks'].values())
    draft_inventory['owners'][0]['roles'].append('draft')
    result = smoke['weight_evidence'](NativeMemoryReport(draft_inventory), True, specs['dflash'])
    assert not result['checks']['independent_draft_owner']


def test_cpu_draft_can_borrow_shared_target_gpu_weights_without_failing_placement(tmp_path, draft_inventory):
    for part in draft_inventory['weight_storage']['intervals'][:-1]:
        part['assignment']['device_ids'] = ['gpu0']
    specs = smoke['launch_specs'](smoke['parser']().parse_args(arguments(tmp_path,
        '--device','Vulkan0','--draft-device','none')))
    result = smoke['weight_evidence'](NativeMemoryReport(draft_inventory), True, specs['dflash'])
    assert all(result['checks'].values())
    assert result['totals']['draft']['accelerator_bytes'] == 34
    assert result['totals']['draft']['cpu_bytes'] == 34


@pytest.mark.parametrize('mode,expected', [('normal',None), ('different-text','review'), ('changed-control','review'),
    ('no-drafting','dflash:draft-counters'), ('no-stream-drafting','dflash:draft-counters'),
    ('no-full-drafting','dflash:draft-counters'), ('no-acceptance','dflash:draft-counters'),
    ('bad-counts','Inconsistent draft counters'), ('changed-weights','dflash:after-cancel:raw'),
    ('no-draft-owner','dflash:loaded:raw'), ('off-leaks-draft','off:loaded:raw'),
    ('partial-memory','dflash:loaded:raw'), ('no-memory','off:loaded:http'),
    ('auth-memory','off:loaded:http'), ('no-finish','Missing or invalid native finish'),
    ('no-done','complete terminal event'), ('http500','500'), ('wrong-build','Unsupported runtime fingerprint'),
    ('reused-cache','measured cache_n=0'), ('unknown-cache','measured cache_n=0'),
    ('changed-shard','dflash:files-unchanged'), ('fatal','device kernel image is invalid')])
def test_capture_preserves_failures_closes_children_and_checks_complete_source_groups(tmp_path, draft_inventory, mode, expected):
    root = tmp_path/'stub'
    root.mkdir()
    (root/'memory-report').write_text(json.dumps(draft_inventory))
    if mode == 'no-memory':
        (root/'memory-report').unlink()
    if mode == 'auth-memory':
        (root/'memory-status').write_text('401')
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + SCRIPT.parents[1].joinpath('tests/native_worker_stub.py').read_text())
    executable.chmod(0o700)
    gguf_file(tmp_path/'model.gguf')
    draft = gguf_file(tmp_path/'draft-00001-of-00002.gguf', arch='dflash', shard=0, count=2)
    continuation = gguf_file(tmp_path/'draft-00002-of-00002.gguf', arch='dflash', shard=1, count=2)
    definition = tmp_path/'original.toml'
    definition.write_text('# untouched sentinel\n[llm]\nmodel="model.gguf"\n')
    original = definition.read_bytes()
    env = {key:value for key,value in os.environ.items() if not key.startswith(('TLLAMA_','LLAMA_ARG_','STUB_'))}
    env.update(STUB_ROOT=str(root), STUB_BUILD='wrong' if mode == 'wrong-build' else BUILD,
               STUB_DFLASH_QUAL=mode, STUB_DFLASH_MUTATE_SHARD=str(continuation),
               PYTHONPATH=str(SCRIPT.parents[1]/'src'))
    if mode == 'fatal':
        env['STUB_MODE'] = 'fatal'
    command = [sys.executable, str(SCRIPT), *arguments(tmp_path, '--draft-model',str(draft),
        '--executable',str(executable), '--transport','tcp', '--load-timeout','15', '--request-timeout','5')]
    reply = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    output = tmp_path/'capture'
    result = json.loads((output/'result.json').read_text())
    passed = expected in (None,'review')
    assert result['ok'] is passed and result['protocol_ok'] is passed, (result, reply.stderr)
    assert reply.returncode == (0 if passed else 1)
    assert result['review_required'] and result['public_api'] == 'not_exercised'
    assert definition.read_bytes() == original
    summaries = [json.loads(line) for line in (output/'summary.jsonl').read_text().splitlines()]
    launches = [json.loads(line) for line in (root/'launch').read_text().splitlines()]
    assert launches and all(not Path(row['cwd']).exists() for row in launches)
    assert any(row['stage'].endswith(':cleanup') for row in summaries)
    if expected == 'review':
        assert result['observations'] and result['error'] is None
    elif expected:
        assert expected in result['error'], (result, reply.stderr)
    else:
        assert len(launches) == 2 and len((root/'cancelled').read_text().splitlines()) == 2
        assert not result['observations']
        assert all(row['ok'] for row in summaries if row['stage'].endswith(':cleanup'))
        assembled = [row for row in summaries if row['stage'] == 'assembled']
        assert len(assembled) == 12
        assert all(row['metrics']['cache_n'] == 0 and 'mtp_observed' not in row['metrics'] for row in assembled)
        identities = next(row['identity'] for row in summaries if row['stage'] == 'input-files')
        assert str(continuation) in {row[0] for row in identities}


def test_existing_capture_is_never_overwritten_or_relaunched(tmp_path, monkeypatch, capsys):
    gguf_file(tmp_path/'model.gguf')
    gguf_file(tmp_path/'draft.gguf', arch='dflash')
    binary = tmp_path/'binary'
    binary.write_bytes(b'not executed')
    capture = tmp_path/'capture'
    capture.mkdir()
    sentinel = capture/'result.json'
    sentinel.write_text('original evidence')
    def forbidden(*args, **kwargs):
        pytest.fail('Existing capture must be rejected before any worker construction')
    monkeypatch.setitem(smoke['main'].__globals__, 'NativeWorker', forbidden)
    assert smoke['main'](arguments(tmp_path, '--executable',str(binary))) != 0
    assert 'File exists' in capsys.readouterr().out
    assert sentinel.read_text() == 'original evidence'


def test_missing_continuation_fails_before_output_or_child(tmp_path, capsys):
    gguf_file(tmp_path/'model.gguf')
    draft = gguf_file(tmp_path/'draft-00001-of-00002.gguf', arch='dflash', shard=0, count=2)
    assert smoke['main'](arguments(tmp_path, '--draft-model',str(draft))) != 0
    assert 'continuation shard' in capsys.readouterr().out
    assert not (tmp_path/'capture').exists()
