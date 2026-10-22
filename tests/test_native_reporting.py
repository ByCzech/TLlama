"""Public /api/ps observes ownership without definition I/O or leases."""

import asyncio
from datetime import datetime, timezone
import hashlib
import signal

import pytest

from tllama.helpers.metadata_cache import save_digest_cache
from tllama.native_reporting import ollama_memory_sizes
from test_native_memory import VULKAN, observe
from test_native_public_api import public, request_body, stream_mode
from test_native_manager import native_manager, records, wait_until


MIB = 1024**2


async def ps(client):
    response = await client.get('/api/ps')
    assert response.status_code == 200
    return response.json()['models']


async def test_ps_reports_native_allocations_and_effective_context_without_extra_work(public):
    manager, client, name, _ = public
    generation = await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    await wait_until(lambda: worker.memory.snapshot()['status'] == 'known')
    before = records(manager.test.roots[0], 'requests')
    for _ in range(3):
        [model] = await ps(client)
        assert model['name'] == name and model['generation'] == generation
        assert model['pid'] == worker.pid and model['state'] == 'ready' and model['leases'] == 0
        assert model['size'] == int(16.5 * MIB) and model['size_vram'] == 0
        assert model['context_length'] == 2048 and model['runner'] == 'llamacpp'
        assert model['memory']['physical_residency_bytes'] is None
        assert datetime.fromisoformat(model['expires_at']).year > datetime.now(timezone.utc).year + 20
    assert records(manager.test.roots[0], 'requests') == before
    assert len(manager.test.ledger) == 1


async def test_missing_memory_is_null_while_inference_remains_usable(public):
    manager, client, name, _ = public
    manager.test.options['mode'] = 'memory-missing'
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    [model] = await ps(client)
    assert model['size'] is None and model['size_vram'] is None
    assert model['memory']['reason'] == 'missing_buffer_reports'
    assert model['memory']['buffers']['CPU']['model'] == 10 * MIB


async def test_loading_generation_has_unknown_context_and_ps_does_not_wait_for_load(public):
    manager, client, name, _ = public
    manager.test.options['mode'] = 'loading'
    task = asyncio.create_task(manager.preload_native_model(name))
    try:
        await wait_until(lambda: manager.test.ledger and manager.test.ledger[0].process is not None)
        [model] = await ps(client)
        assert model['state'] == 'loading' and model['context_length'] is None
        await task
        assert (await ps(client))[0]['state'] == 'ready'
    finally:
        await asyncio.gather(task, return_exceptions=True)


async def test_toml_and_weight_changes_cannot_rewrite_live_reporting_snapshot(public, monkeypatch):
    manager, client, name, toml = public
    monkeypatch.setattr(manager, '_get_model_metadata_sync', lambda _: {'arch':'qwen3', 'params':35_000_000_000, 'bits':'IQ3_S'})
    weights = manager.local_models_dir / 'a.gguf'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    save_digest_cache(manager.metadata_cache_dir, name, weights, {'content_sha256':digest, 'source':'local'})
    await manager.preload_native_model(name, keep_alive=-1)
    [before] = await ps(client)
    assert before['digest'] == digest and before['details']['parameter_size'] == '35b'
    toml.write_text('invalid [toml')
    weights.write_bytes(b'new different weights')
    def forbidden(*args, **kwargs):
        pytest.fail('Reporting reread mutable model definitions or hashed changed weights')
    for method in ('get_model_metadata', 'get_model_digest',
                   '_build_model_file_info', '_resolve_virtual_model_spec', 'list_local_models'):
        monkeypatch.setattr(manager, method, forbidden)
    [after] = await ps(client)
    assert after['details'] == before['details'] and after['digest'] == digest
    assert after['size'] == before['size'] and after['generation'] == before['generation']
    weights.unlink()
    toml.unlink()
    assert (await ps(client))[0]['digest'] == digest


async def test_ps_reads_cannot_extend_idle_ttl(public):
    manager, client, name, _ = public
    now = [1000.0]
    manager._native._clock = lambda: now[0]
    generation = await manager.preload_native_model(name, keep_alive=5)
    for _ in range(3):
        now[0] += 1
        [model] = await ps(client)
        assert model['leases'] == 0
        remaining = (datetime.fromisoformat(model['expires_at']) - datetime.now(timezone.utc)).total_seconds()
        assert 5 - (now[0] - 1000) - 0.2 < remaining <= 5 - (now[0] - 1000) + 0.2
    now[0] = 1006
    await manager._native.expire()
    await manager.wait_native_closed(generation)
    assert await ps(client) == []


async def test_draining_stream_stays_listed_until_its_lease_is_released(public):
    manager, client, name, _ = public
    stream_mode(manager, 'controlled')
    task = asyncio.create_task(client.post('/api/chat', json=request_body(name, '/api/chat', True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / 'started').exists())
        [active] = await ps(client)
        assert active['leases'] == 1 and active['state'] == 'ready'
        await manager.unload_native_model(name)
        [draining] = await ps(client)
        assert draining['generation'] == active['generation'] and draining['size'] == active['size']
        assert draining['state'] == 'draining' and draining['leases'] == 1
        assert draining['retirement_reason'] == 'explicit unload'
        assert datetime.fromisoformat(draining['expires_at']) <= datetime.now(timezone.utc)
    finally:
        (manager.test.roots[0] / 'release').touch()
        await task
    await manager.wait_native_closed(active['generation'])
    assert await ps(client) == []


async def test_two_generations_of_one_name_are_reported_separately(public):
    manager, client, name, toml = public
    manager._native.max_loaded_models = 2
    async with manager.acquire_native_model(name, keep_alive=-1) as old:
        toml.write_text(toml.read_text() + '\n[runtime]\nn_ctx=8192\n')
        async with manager.acquire_native_model(name, keep_alive=-1) as new:
            models = await ps(client)
            assert len(models) == 2 and {m['generation'] for m in models} == {old.generation, new.generation}
            assert {m['state'] for m in models} == {'ready', 'draining'}
            assert len({m['pid'] for m in models}) == 2
    await manager.wait_native_closed(old.generation)
    assert len(await ps(client)) == 1


async def test_dead_child_is_not_reported_as_still_allocating_memory(public):
    manager, client, name, _ = public
    await manager.preload_native_model(name, keep_alive=-1)
    worker = manager.test.ledger[0]
    worker.process.send_signal(signal.SIGKILL)
    await worker.wait_for_exit()
    assert await ps(client) == []


async def test_returned_reporting_data_cannot_mutate_a_generation(public):
    manager, _, name, _ = public
    await manager.preload_native_model(name)
    report = manager.native_reporting_snapshot()[0]
    report['metadata']['arch'] = 'wrong'
    report['memory']['measurement']['report']['buffers']['CPU']['model'] = 0
    again = manager.native_reporting_snapshot()[0]
    assert again['metadata'].get('arch') != 'wrong'
    assert again['memory']['measurement']['report']['buffers']['CPU']['model'] == 10 * MIB


async def test_large_accepted_keep_alive_cannot_overflow_reporting_timestamp(public):
    manager, client, name, _ = public
    await manager.preload_native_model(name, keep_alive=10**20)
    [model] = await ps(client)
    assert datetime.fromisoformat(model['expires_at']).year > datetime.now(timezone.utc).year + 20


async def test_memory_parser_failure_cannot_break_startup_or_inference(public, monkeypatch):
    from tllama.native_memory import NativeMemory
    manager, client, name, _ = public
    def broken(*args):
        raise ValueError('diagnostic parser failed')
    monkeypatch.setattr(NativeMemory, 'observe', broken)
    assert (await client.post('/api/chat', json=request_body(name, '/api/chat'))).status_code == 200
    [model] = await ps(client)
    assert model['size'] is None and model['memory']['reason'] == 'parser_error'


@pytest.mark.parametrize('device', ['Vulkan0', 'CUDA0', 'ROCm0', 'SYCL0'])
@pytest.mark.parametrize('input_buffer', ['CPU_Mapped', 'CPU', 'device_Host'])
def test_full_offload_ps_excludes_input_and_helpers_without_rewriting_allocations(device, input_buffer):
    log = VULKAN.replace('Vulkan0', device).replace('Vulkan_Host', device.rstrip('0') + '_Host')
    log = log.replace('CPU_Mapped', input_buffer.replace('device', device.rstrip('0')))
    parser = observe(log + '\nload_tensors: offloaded 41/41 layers to GPU')
    memory = parser.snapshot()
    assert ollama_memory_sizes(memory) == (memory['device_bytes'], memory['device_bytes'])
    assert memory['host_bytes'] == sum(round(size * MIB) for size in (397.85, 0.95, 20.30))
    assert memory == parser.snapshot()


@pytest.mark.parametrize('summary', ['', '20/41', '0/41', '42/41', '0/0'])
def test_ps_keeps_host_weights_without_proven_full_offload(summary):
    log = VULKAN + (f'\nload_tensors: offloaded {summary} layers to GPU' if summary else '')
    memory = observe(log).snapshot()
    assert ollama_memory_sizes(memory) == (memory['allocated_bytes'], memory['device_bytes'])
    assert memory['allocated_bytes'] > memory['device_bytes']


@pytest.mark.parametrize('origin,backend,kind', [('llama_kv_cache', 'CPU', 'KV'),
    ('llama_memory_recurrent', 'CPU', 'RS'), ('sched_reserve', 'CPU', 'compute'),
    ('llama_kv_cache', 'Vulkan_Host', 'KV'), ('llama_memory_recurrent', 'Vulkan_Host', 'RS')])
def test_ollama_full_layer_offload_summary_keeps_cpu_state_in_detailed_inventory(origin, backend, kind):
    log = VULKAN + f'\n{origin}: {backend} {kind} buffer size = 50 MiB\nload_tensors: offloaded 41/41 layers to GPU'
    memory = observe(log).snapshot()
    assert ollama_memory_sizes(memory) == (memory['device_bytes'], memory['device_bytes'])
    assert memory['host_bytes'] >= 50 * MIB


def test_full_offload_cpu_output_is_a_result_buffer_not_cpu_layer_execution():
    memory = observe(VULKAN.replace('Vulkan_Host output', 'CPU output')
                     + '\nload_tensors: offloaded 41/41 layers to GPU').snapshot()
    assert ollama_memory_sizes(memory) == (memory['device_bytes'], memory['device_bytes'])


@pytest.mark.parametrize('options', [{'end':False}, {'projector':True}])
def test_full_offload_cannot_turn_incomplete_allocations_into_a_known_ps_total(options):
    memory = observe(VULKAN + '\nload_tensors: offloaded 41/41 layers to GPU', **options).snapshot()
    assert ollama_memory_sizes(memory) == (None, None)


def test_unverified_build_does_not_apply_the_pinned_host_buffer_exclusion():
    memory = observe(VULKAN + '\nload_tensors: offloaded 41/41 layers to GPU',
                     runtime_fingerprint='new-runtime').snapshot()
    assert memory['runtime_verified'] is False
    assert ollama_memory_sizes(memory) == (memory['allocated_bytes'], memory['device_bytes'])


def test_verified_v1_log_retains_full_offload_processor_reporting():
    log = VULKAN + "\nload_tensors: offloaded 41/41 layers to GPU\n" + (
        "srv  load_model_i: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'true'")
    memory = observe(log, end=False, runtime_fingerprint='b11552-d1570abd3d').snapshot()
    assert memory['status'] == 'known' and memory['runtime_verified'] is True
    assert ollama_memory_sizes(memory) == (memory['device_bytes'], memory['device_bytes'])


def test_full_offload_summary_without_device_weights_cannot_hide_cpu_allocations():
    memory = observe(VULKAN.replace('Vulkan0 model', 'CPU model')
                     + '\nload_tensors: offloaded 41/41 layers to GPU').snapshot()
    assert ollama_memory_sizes(memory) == (memory['allocated_bytes'], memory['device_bytes'])


async def test_public_ps_restores_full_gpu_placement_from_actual_worker_log(public):
    manager, client, name, _ = public
    # The fixture requests device=none; placement must come from the actual
    # load report, not from requested device/layer settings.
    manager.test.options['mode'] = 'memory-full-offload'
    await manager.preload_native_model(name, keep_alive=-1)
    [model] = await ps(client)
    assert model['size'] == model['size_vram'] == 16 * MIB
    memory = model['memory']
    assert memory['allocated_bytes'] == int(19.5 * MIB)
    assert memory['host_bytes'] == int(3.5 * MIB)
    assert memory['layer_offload'] == {'offloaded_layers':41, 'total_layers':41}
