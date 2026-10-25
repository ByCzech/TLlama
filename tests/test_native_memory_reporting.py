"""Public memory inventory and Ollama processor policy stay independently useful."""

from copy import deepcopy

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_sizes, ollama_processes, public_memory
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload


@pytest.fixture
def accelerator(payload):
    payload['placement']['layer_groups'][-1].update(location='accelerator', device_ids=['gpu0'])
    return payload


def selected(payload, **kwargs):
    telemetry = MemoryTelemetry(observe(), **kwargs)
    telemetry.accept(NativeMemoryReport(payload))
    return telemetry.snapshot()


def memory(payload):
    return public_memory(selected(payload))


def host_buffer(payload, component, *, owner='context', kind='host', size=7):
    buffer = deepcopy(payload['buffers'][2])
    buffer.update(id='extra', owners=[{'owner_id':owner, 'component':component}],
                  memory_kind=kind, backing='allocated', size_bytes=size)
    payload['buffers'].append(buffer)


@pytest.mark.parametrize('backend,device_type', [('Vulkan', 'igpu'), ('CUDA', 'gpu'), ('ROCm', 'gpu')])
def test_full_accelerator_placement_matches_ollama_but_preserves_complete_inventory(accelerator, backend, device_type):
    accelerator['devices'][1].update(backend=backend, type=device_type, name='arbitrary name')
    report = memory(accelerator)
    before = deepcopy(report)
    assert ollama_memory_sizes(report) == (1300, 1300)
    assert report == before
    assert report['allocated_bytes'] == 1350 and report['host_bytes'] == 50
    assert report['buffers'][2]['size_bytes'] == 30
    assert report['precision'] == 'exact_bytes' and report['scope'] == 'backend_buffers'
    assert report['source'] == 'native_memory_api' and report['physical_residency_bytes'] is None


@pytest.mark.parametrize('component,owner,kind', [
    ('kv_cache', 'context', 'host'), ('recurrent_state', 'context', 'host'),
    ('state_other', 'context', 'host'), ('compute', 'context', 'host'),
    ('kv_cache', 'context', 'host_pinned'), ('recurrent_state', 'context', 'host_pinned'),
    ('projector_weights', 'projector', 'host'), ('projector_compute', 'projector', 'host'),
    ('projector_compute', 'projector', 'host_pinned'),
])
@pytest.mark.parametrize('full', [True, False])
def test_ollama_full_offload_summary_keeps_host_allocations_only_in_detail(accelerator, component, owner, kind, full):
    if not full:
        accelerator['placement']['layer_groups'][-1].update(location='cpu', device_ids=['cpu'])
    host_buffer(accelerator, component, owner=owner, kind=kind)
    report = memory(accelerator)
    assert ollama_memory_sizes(report) == ((1300 if full else 1357), 1300)
    assert report['allocated_bytes'] == 1357 and report['host_bytes'] == 57


def test_positively_identified_pinned_context_compute_is_staging(accelerator):
    host_buffer(accelerator, 'compute', kind='host_pinned')
    assert ollama_memory_sizes(memory(accelerator)) == (1300, 1300)


@pytest.mark.parametrize('component', ['kv_cache', 'compute', 'unknown_component'])
def test_ollama_full_offload_summary_does_not_reclassify_shared_host_leaves(accelerator, component):
    accelerator['buffers'][2]['owners'].append({'owner_id':'context', 'component':component})
    assert ollama_memory_sizes(memory(accelerator)) == (1300, 1300)


def test_shared_staging_with_two_removable_owners_is_subtracted_only_once(accelerator):
    accelerator['buffers'][3]['owners'].append({'owner_id':'mtp', 'component':'compute'})
    assert ollama_memory_sizes(memory(accelerator)) == (1300, 1300)


def test_full_offload_host_adapter_remains_in_detailed_inventory(accelerator):
    accelerator['owners'].append({'id':'adapter', 'kind':'adapter', 'roles':['lora']})
    host_buffer(accelerator, 'adapter_weights', owner='adapter')
    report = memory(accelerator)
    assert ollama_memory_sizes(report) == (1300, 1300)
    assert report['host_bytes'] == 57
    assert report['buffers'][-1]['owners'] == [{'owner_id':'adapter', 'component':'adapter_weights'}]


@pytest.mark.parametrize('component', ['repeating_layers', 'output_weights', 'mtp_layers', 'other_weights'])
@pytest.mark.parametrize('location', ['cpu', 'mixed'])
def test_actual_cpu_or_mixed_noninput_weights_prevent_full_gpu_summary(accelerator, component, location):
    accelerator['placement']['layer_groups'].append({
        'owner_id':'model', 'component':component, 'layer_indices':[9] if component.endswith('layers') else [],
        'location':location, 'device_ids':['cpu', 'gpu0']})
    assert ollama_memory_sizes(memory(accelerator)) == (1350, 1300)


def test_separate_cpu_draft_model_prevents_full_gpu_summary(accelerator):
    accelerator['owners'].append({'id':'draft', 'kind':'model', 'roles':['draft']})
    accelerator['placement']['layer_groups'].append({'owner_id':'draft', 'component':'repeating_layers',
        'layer_indices':[0], 'location':'cpu', 'device_ids':['cpu']})
    host_buffer(accelerator, 'model_weights', owner='draft')
    assert ollama_memory_sizes(memory(accelerator)) == (1357, 1300)


def test_cpu_inventory_remains_all_host_even_with_gpu_like_buffer_names(payload):
    for buffer in payload['buffers']:
        buffer.update(memory_kind='host', device_ids=['cpu'])
    for group in payload['placement']['layer_groups']:
        group.update(location='cpu', device_ids=['cpu'])
    assert ollama_memory_sizes(memory(payload)) == (1350, 0)


@pytest.mark.parametrize('status', ['partial', 'unavailable'])
def test_incomplete_inventory_keeps_known_subtotals_but_not_processor_claims(accelerator, status):
    accelerator['status'] = status
    accelerator['buffers'][0]['size_bytes'] = None
    accelerator['missing_components'] = [{'owner_id':'model', 'component':'model_weights', 'reason':'opaque_composite'}]
    report = memory(accelerator)
    assert ollama_memory_sizes(report) == (None, None)
    assert report['known_bytes'] == 350 and report['unsized_buffers'] == 1
    assert report['missing_components'] == accelerator['missing_components']


def test_known_allocations_with_partial_placement_do_not_guess_a_processor_split(accelerator):
    accelerator['placement']['status'] = 'partial'
    accelerator['placement']['layer_groups'][-1]['location'] = 'unknown'
    report = memory(accelerator)
    assert report['status'] == 'known' and report['allocated_bytes'] == 1350
    assert ollama_memory_sizes(report) == (None, None)


def test_unknown_memory_kind_does_not_invent_cpu_or_gpu_totals(accelerator):
    accelerator['buffers'][0]['memory_kind'] = 'unknown'
    report = memory(accelerator)
    assert report['allocated_bytes'] == 1350 and report['unclassified_bytes'] == 1000
    assert ollama_memory_sizes(report) == (None, None)


def test_new_placement_component_is_exposed_without_assuming_its_execution_role(accelerator):
    accelerator['placement']['layer_groups'][0]['component'] = 'future_weights'
    assert ollama_memory_sizes(memory(accelerator)) == (None, None)


def test_missing_model_placement_is_not_inferred_from_allocations(accelerator):
    accelerator['placement']['layer_groups'] = []
    assert ollama_memory_sizes(memory(accelerator)) == (None, None)


def test_full_noninput_offload_does_not_require_an_input_group_for_the_compatibility_summary(accelerator):
    accelerator['placement']['layer_groups'].pop(0)
    assert ollama_memory_sizes(memory(accelerator)) == (1300, 1300)


def test_empty_sleeping_inventory_is_zero_instead_of_unavailable(accelerator):
    accelerator.update(buffers=[], owners=[], devices=[], model_state='sleeping', snapshot_stage='after_unload')
    accelerator['placement']['layer_groups'] = []
    assert ollama_memory_sizes(memory(accelerator)) == (0, 0)


def test_native_only_failure_is_unavailable_even_when_log_has_complete_metrics():
    telemetry = MemoryTelemetry(observe(), mode='native')
    telemetry.fail('authentication_failed', http_status=401)
    report = public_memory(telemetry.snapshot())
    assert report['source'] is None and report['status'] == 'unavailable'
    assert report['reason'] == 'authentication_failed' and report['native']['http_status'] == 401
    assert report['buffers'] is None
    assert ollama_memory_sizes(report) == (None, None)


def test_log_fallback_retains_scope_reason_and_own_allocation_totals():
    telemetry = MemoryTelemetry(observe())
    telemetry.fail('endpoint_not_supported', http_status=404)
    report = public_memory(telemetry.snapshot())
    assert report['source'] == 'native_load_log' and report['fallback_reason'] == 'endpoint_not_supported'
    assert report['scope'] == 'initial_load_allocations' and report['precision'] == 'rounded_mib'
    assert report['allocated_bytes'] == observe().snapshot()['allocated_bytes']
    assert report['observation_age_ms'] is report['estimated_snapshot_age_ms'] is None


def test_snapshot_age_includes_local_cache_age_without_rewriting_native_comparison(accelerator):
    now = [100.0]
    telemetry = MemoryTelemetry(observe(), compare=True, clock=lambda: now[0])
    telemetry.accept(NativeMemoryReport(accelerator))
    now[0] += 2
    report = public_memory(telemetry.snapshot())
    assert report['snapshot_age_ms'] == 23 and report['observation_age_ms'] == 2000
    assert report['estimated_snapshot_age_ms'] == 2023
    assert report['comparison']['native']['report']['snapshot_age_ms'] == 23
    assert report['comparison']['log']['report']['scope'] == 'initial_load_allocations'


def test_public_processor_sizes_explain_the_difference_from_complete_totals(accelerator):
    entry = {'metadata':{}, 'file_info':{}, 'memory':selected(accelerator), 'model':'m', 'generation':1,
             'context_length':4096, 'expires_at':'2099-01-01T00:00:00+00:00', 'state':'ready',
             'pid':123, 'leases':0, 'retirement_reason':None}
    [model] = ollama_processes([entry])['models']
    assert model['size'] == model['size_vram'] == 1300
    assert model['memory']['allocated_bytes'] == 1350
    assert model['memory']['ollama_compatibility'] == {
        'size':1300, 'size_vram':1300, 'basis':'native_weight_placement',
        'reason':'full_accelerator_weights', 'metric':'backend_buffers', 'precision':'exact_bytes',
        'weight_capacity_reduction_bytes':0, 'policy_excluded_bytes':50}


def test_captured_qwen_vulkan_apu_capacities_and_placement_retain_full_gpu_display(payload):
    # Capacities and placement from the user's October 16, 2026 smoke run,
    # b11552-d1570abd3d, Qwen3.6-35B-A3B-IQ3_S, ctx 4096, K/V q4_0.
    # IDs are normalized to the fixture; each captured leaf stays separate.
    sizes = [1071351296, 971280640, 1070108160, 1040748032, 1073631488,
             960579840, 1070108160, 1040748032, 1073631488, 1070108160,
             987842816, 1040748032, 777669376]
    template = deepcopy(payload['buffers'][0])
    input_buffer = deepcopy(payload['buffers'][2])
    input_buffer.update(size_bytes=417177600)
    payload['buffers'] = [input_buffer]
    for index, size in enumerate(sizes):
        buffer = deepcopy(template)
        buffer.update(id=f'weight-{index}', size_bytes=size)
        payload['buffers'].append(buffer)
    for component, kind, size in [('kv_cache', 'device', 23592960),
            ('recurrent_state', 'device', 65863680), ('compute', 'device', 84185096),
            ('compute', 'host_pinned', 21286944), ('output', 'host_pinned', 993312)]:
        buffer = deepcopy(template)
        buffer.update(id=f'context-{component}-{kind}', size_bytes=size, memory_kind=kind,
                      owners=[{'owner_id':'context', 'component':component}])
        payload['buffers'].append(buffer)
    payload['owners'] = payload['owners'][:2]
    payload['owners'][0]['roles'] = ['main']
    payload['placement']['layer_groups'][-1].update(component='output_weights', layer_indices=[],
                                                  location='accelerator', device_ids=['gpu0'])
    payload['placement']['layer_groups'][1]['layer_indices'] = list(range(40))
    report = memory(payload)
    assert report['allocated_bytes'] == 13861655112
    assert report['host_bytes'] == 439457856 and report['device_bytes'] == 13422197256
    assert ollama_memory_sizes(report) == (13422197256, 13422197256)


def test_vision_cpu_compute_regression_reports_full_gpu_without_rewriting_inventory(accelerator):
    from tllama.cli_output import processes_text

    host_buffer(accelerator, 'projector_compute', owner='projector', size=26136832)
    before = deepcopy(accelerator)
    entry = {'metadata':{}, 'file_info':{}, 'memory':selected(accelerator), 'model':'vision',
             'generation':1, 'context_length':262144, 'expires_at':None, 'state':'ready',
             'pid':123, 'leases':0, 'retirement_reason':None}
    result = ollama_processes([entry])
    [model] = result['models']
    assert model['size'] == model['size_vram'] == 1300
    assert model['memory']['allocated_bytes'] == 26138182
    assert model['memory']['host_bytes'] == 26136882
    assert model['memory']['buffers'] == before['buffers']
    assert model['memory']['ollama_compatibility']['policy_excluded_bytes'] == 26136882
    assert accelerator == before
    assert '100% GPU' in processes_text(result)
