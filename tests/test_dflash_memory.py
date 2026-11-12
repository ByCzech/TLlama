"""DFlash memory ownership and honest fallback; all sizes here are synthetic."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_sizes
from tllama.worker import NativeWorker

from test_helper_definitions import publish, reference
from test_native_dflash import definition
from test_native_memory import VULKAN, END, observe
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights, interval, relationship
from test_native_manager import native_manager, records
from test_native_public_api import public, request_body
from test_native_public_memory import native_public, loaded
from test_native_reporting import ps


@pytest.mark.parametrize('extra', ['', '\nsched_reserve: Vulkan0 compute buffer size = 12 MiB',
    '\nload_tensors: offloaded 41/41 layers to GPU',
    '\nload_tensors: CPU model buffer size = 100 MiB\nllama_kv_cache: CPU KV buffer size = 3 MiB'])
def test_unscoped_target_and_draft_log_never_claims_complete_totals(extra):
    parser = observe(VULKAN + extra, external_draft=True)
    report = parser.snapshot()
    assert report['status'] == 'partial' and report['reason'] == 'draft_allocation_unverified'
    assert report['allocated_bytes'] is report['device_bytes'] is report['host_bytes'] is None
    assert report['buffers'] and report['layer_offload'] is None
    assert ollama_memory_sizes(report) == (None, None)
    before = deepcopy(report)
    parser.observe('load_tensors: CPU model buffer size = 999 MiB')
    assert parser.snapshot() == before


@pytest.mark.parametrize('log,reason,end', [(VULKAN, 'load_log_pending', False),
    ('load_tensors: CPU model buffer size = 1 MiB', 'missing_buffer_reports', True),
    (VULKAN + '\nllama_context: CPU output buffer size = broken MiB', 'invalid_buffer_size', True)])
def test_draft_uncertainty_does_not_mask_pending_or_broken_logs(log, reason, end):
    assert observe(log, end=end, external_draft=True).snapshot()['reason'] == reason


@pytest.mark.parametrize('strategy', ['none', 'draft-mtp', 'draft-dflash'])
def test_worker_enables_log_guard_only_when_external_weights_are_actually_loaded(strategy):
    from pathlib import Path
    from tllama.config import BackendConfig
    from tllama.helpers.model_toml import parse_model_toml
    from tllama.native_config import resolve_worker_config
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}),
        Path('/models/m.gguf'), parse_model_toml(definition('Local/m.gguf', 'Local/d.gguf', strategy)),
        draft=Path('/models/d.gguf'))
    worker = NativeWorker(resolved.launch)
    worker.memory.identify_runtime('b11850-0e9433bfd3')
    for line in VULKAN.splitlines() + [END]:
        worker.memory.observe(line)
    report = worker.memory.snapshot()
    assert report['status'] == ('partial' if strategy == 'draft-dflash' else 'known')
    if strategy != 'draft-dflash':
        assert report == observe().snapshot()


@pytest.fixture
async def draft_public(native_public, weights, tmp_path):
    manager, client, name, toml, raw, options = native_public
    assert raw is weights
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    toml.write_text(toml.read_text() + '\n' + reference('draft', draft)
                    + '[runtime]\nspec_type="draft-dflash"\n')
    # Main weights are accelerated except four input-only bytes. The second
    # context shares a compute buffer; a separate draft owns its own storage.
    raw['owners'][0]['roles'] = ['main']
    raw['owners'][2].update(id='draft-context', roles=['draft'])
    for buffer in raw['buffers']:
        for rel in buffer['owners']:
            if rel['owner_id'] == 'mtp':
                rel['owner_id'] = 'draft-context'
    for part in raw['weight_storage']['intervals'][3:]:
        part['assignment']['device_ids'] = ['gpu0']
    raw['owners'].append({'id':'draft-model', 'kind':'model', 'roles':['draft']})
    # A shared weight interval belongs to both roles but is still counted once.
    raw['buffers'][0]['owners'].append({'owner_id':'draft-model', 'component':'model_weights'})
    raw['weight_storage']['intervals'][0]['relationships'].append(
        relationship('output_weights', owner='draft-model'))
    yield manager, client, name, raw, options


def add_draft_storage(raw, device):
    host = device == 'cpu'
    raw['buffers'].append({
        'id':'draft-weights', 'owners':[{'owner_id':'draft-model', 'component':'model_weights'}],
        'device_ids':[device], 'buffer_type':'CPU' if host else 'Vulkan0',
        'memory_kind':'host' if host else 'device', 'backing':'allocated', 'size_bytes':1000})
    raw['weight_storage']['intervals'].append(interval('draft-payload', 'draft-weights', 64, 34,
        relationship('repeating_layers', 0, 'draft-model'), devices=(device,)))
    raw['placement']['layer_groups'].append({'owner_id':'draft-model', 'component':'repeating_layers',
        'layer_indices':[0], 'location':'cpu' if host else 'accelerator', 'device_ids':[device]})
    for component, size in [('kv_cache',17), ('compute',19)]:
        buffer = deepcopy(raw['buffers'][-1])
        buffer.update(id='draft-' + component, size_bytes=size,
                      owners=[{'owner_id':'draft-context', 'component':component}])
        raw['buffers'].append(buffer)


async def freeze_sampler(worker):
    worker._memory_task.cancel()
    await asyncio.gather(worker._memory_task, return_exceptions=True)


@pytest.mark.parametrize('device,expected', [('cpu',(580,486)), ('gpu0',(556,556))])
async def test_active_dflash_keeps_native_weights_runtime_and_shared_bytes_exact(draft_public, device, expected):
    manager, client, name, raw, _ = draft_public
    add_draft_storage(raw, device)
    report = NativeMemoryReport(raw)
    assert report.totals().allocated_bytes == 2386
    assert report.weight_totals().payload_bytes == 224
    # Role subtotals overlap. They must never be added to compute global SIZE.
    assert report.weight_totals(roles={'main'}).payload_bytes == 190
    assert report.weight_totals(roles={'draft'}).payload_bytes == 68
    _, worker = await loaded(manager, name)
    await freeze_sampler(worker)
    before = records(manager.test.roots[0], 'requests')
    for detailed in (False, True):
        response = await client.get('/api/ps', params={'detailed':str(detailed).lower()})
        [model] = response.json()['models']
        assert (model['size'], model['size_vram']) == expected
        if detailed:
            memory = model['memory']
            assert memory['source'] == 'native_memory_api' and memory['allocated_bytes'] == 2386
            assert memory['weight_storage_totals']['payload_bytes'] == 224
            assert memory['comparison']['log']['report']['reason'] == 'draft_allocation_unverified'
    assert records(manager.test.roots[0], 'requests') == before
    assert len(manager.test.ledger) == 1


@pytest.mark.parametrize('status', [404, 503])
async def test_failed_native_read_cannot_fall_back_to_main_only_log_size(draft_public, status):
    manager, client, name, raw, options = draft_public
    add_draft_storage(raw, 'cpu')
    options['status'] = status
    _, worker = await loaded(manager, name)
    await freeze_sampler(worker)
    [model] = await ps(client)
    assert model['size'] is model['size_vram'] is None
    memory = model['memory']
    assert memory['source'] == 'native_load_log' and memory['buffers']
    assert memory['status'] == 'partial' and memory['reason'] == 'draft_allocation_unverified'
    assert memory['fallback_reason'] == ('endpoint_not_supported' if status == 404 else 'not_ready')
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    assert len(manager.test.ledger) == 1


async def test_native_failure_and_recovery_never_reuse_old_totals_or_mix_sources(draft_public):
    manager, client, name, raw, _ = draft_public
    add_draft_storage(raw, 'cpu')
    _, worker = await loaded(manager, name)
    await freeze_sampler(worker)
    [initial] = await ps(client)
    assert initial['size'] == 580
    root = manager.test.roots[0]
    (root/'memory-status').write_text('503')
    await worker.refresh_memory()
    [failed] = await ps(client)
    assert failed['size'] is failed['size_vram'] is None
    assert failed['memory']['reason'] == 'draft_allocation_unverified'
    # A valid partial native report wins over log fallback, retaining subtotals.
    raw['snapshot_sequence'] += 1
    raw['status'] = 'partial'
    raw['missing_components'] = [{'owner_id':'draft-context', 'component':'compute', 'reason':'not_observed'}]
    (root/'memory-status').write_text('200')
    (root/'memory-report').write_text(json.dumps(raw))
    await worker.refresh_memory()
    [partial] = await ps(client)
    assert partial['size'] is None and partial['memory']['source'] == 'native_memory_api'
    assert partial['memory']['known_bytes'] == 2386 and partial['memory']['fallback_reason'] is None
    raw['snapshot_sequence'] += 1
    raw.update(status='known', missing_components=[])
    (root/'memory-report').write_text(json.dumps(raw))
    await worker.refresh_memory()
    [recovered] = await ps(client)
    assert (recovered['size'], recovered['size_vram']) == (580,486)
    assert initial['memory']['allocated_bytes'] == 2386 and len(manager.test.ledger) == 1


@pytest.mark.parametrize('mode', ['log', 'native'])
async def test_explicit_source_policy_remains_honest_and_inference_usable(draft_public, monkeypatch, mode):
    manager, client, name, raw, options = draft_public
    add_draft_storage(raw, 'cpu')
    options['status'] = 503
    factory = manager._native._factory
    monkeypatch.setattr(manager._native, '_factory', lambda spec: factory(
        replace(spec, memory_source=mode, memory_compare=False)))
    _, worker = await loaded(manager, name)
    if worker._memory_task is not None:
        await freeze_sampler(worker)
    [model] = await ps(client)
    assert model['size'] is model['size_vram'] is None
    assert model['memory']['source'] == ('native_load_log' if mode == 'log' else None)
    assert (await worker.chat({}))['choices']
