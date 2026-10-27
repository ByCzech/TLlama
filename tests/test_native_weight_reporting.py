"""Display accounting replaces weight capacity without losing runtime costs."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tllama.cli_output import processes_text
from tllama.native_reporting import ollama_processes
from tllama.worker import WorkerError
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_memory_reporting import selected
from test_native_weight_storage import weights, interval, relationship
from test_memory_smoke import smoke as memory_smoke


script = Path(__file__).parents[1] / 'scripts' / 'api-smoke.py'
spec = importlib.util.spec_from_file_location('weight_api_smoke', script)
api_smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_smoke)


def reported(raw, **selection):
    entry = {'metadata': {}, 'file_info': {}, 'memory': selected(raw, **selection),
             'model': 'renamed', 'generation': 1, 'context_length': 4096,
             'expires_at': None, 'state': 'ready', 'pid': 123, 'leases': 0,
             'retirement_reason': None}
    return ollama_processes([entry])


def model(raw):
    return reported(raw)['models'][0]


def reconcile(result):
    memory = result['memory']
    summary = memory['ollama_compatibility']
    assert memory['allocated_bytes'] - result['size'] == (
        summary['weight_capacity_reduction_bytes'] + summary['policy_excluded_bytes'])
    assert summary['policy_excluded_bytes'] == (
        summary['excluded_input_weight_bytes'] + summary['excluded_nonweight_host_bytes'])
    assert 'excluded_host_bytes' not in summary
    api_smoke.validate_processor_summary(result)


@pytest.fixture
def full(weights):
    # Keep only the input-only [0,4) on CPU. Input/output [4,8) is GPU;
    # it must survive the display exclusion along with output [8,12).
    for item in weights['weight_storage']['intervals'][3:]:
        item['assignment']['device_ids'] = ['gpu0']
    return weights


def add_buffer(raw, component, size, kind, owner='context'):
    buffer = deepcopy(raw['buffers'][1])
    buffer.update(id='extra-' + str(len(raw['buffers'])), size_bytes=size, memory_kind=kind,
                  owners=[{'owner_id': owner, 'component': component}])
    raw['buffers'].append(buffer)


def test_mixed_ps_uses_payload_plus_nonweight_costs_and_real_assignment(weights):
    before = deepcopy(weights)
    result = model(weights)
    assert (result['size'], result['size_vram']) == (510, 478)
    memory = result['memory']
    assert (memory['allocated_bytes'], memory['host_bytes'], memory['device_bytes']) == (1350, 50, 1300)
    assert memory['weight_storage_totals']['payload_bytes'] == 190
    assert memory['ollama_compatibility'] == {
        'size': 510, 'size_vram': 478, 'basis': 'native_weight_storage',
        'metric': 'weight_payload_and_nonweight_buffers', 'precision': 'exact_bytes',
        'reason': 'weight_payload_and_buffer_totals', 'weight_capacity_reduction_bytes': 840,
        'policy_excluded_bytes': 0, 'excluded_input_weight_bytes': 0,
        'excluded_nonweight_host_bytes': 0,
    }
    assert memory['buffers'] == before['buffers'] and weights == before
    assert '6.3% CPU / 93.7% GPU' in processes_text(reported(weights))
    reconcile(result)


@pytest.mark.parametrize('backing', ['mmap', 'allocated'])
def test_sparse_span_does_not_become_a_model_size(weights, backing):
    mib = 1024**2
    weights['buffers'][0].update(size_bytes=1024*mib, backing=backing)
    weights['weight_storage']['intervals'][:2] = [
        interval('first', 'weights', 0, mib, relationship('repeating_layers', 39)),
        interval('last', 'weights', 1023*mib, mib, relationship('mtp_layers', 40)),
    ]
    result = model(weights)
    assert (result['size'], result['size_vram']) == (2*mib+332, 2*mib+300)
    assert result['memory']['allocated_bytes'] == 1024*mib+350
    assert result['memory']['ollama_compatibility']['weight_capacity_reduction_bytes'] == 1022*mib+18
    reconcile(result)


@pytest.mark.parametrize('backend,kind', [('Vulkan', 'igpu'), ('CUDA', 'gpu'), ('ROCm', 'gpu'), ('SYCL', 'accel')])
def test_full_noninput_offload_uses_gpu_payload_and_keeps_tied_output(full, backend, kind):
    full['devices'][1].update(backend=backend, type=kind)
    result = model(full)
    assert result['size'] == result['size_vram'] == 486  # 186 weight + 300 nonweight device
    assert result['memory']['ollama_compatibility']['excluded_input_weight_bytes'] == 4
    assert result['memory']['ollama_compatibility']['excluded_nonweight_host_bytes'] == 20
    assert result['memory']['ollama_compatibility']['reason'] == 'full_accelerator_weights'
    assert '100% GPU' in processes_text(reported(full))
    reconcile(result)


@pytest.mark.parametrize('component,kind,owner', [
    ('kv_cache', 'device', 'context'), ('recurrent_state', 'device', 'context'),
    ('state_other', 'host', 'context'), ('compute', 'host_pinned', 'context'),
    ('output', 'host_pinned', 'context'), ('moe_cache', 'device', 'context'),
    ('control_vector', 'host', 'context'), ('projector_compute', 'host', 'projector'),
    ('projector_weights', 'device', 'projector'), ('adapter_weights', 'device', 'adapter'),
    ('adapter_weights', 'host', 'adapter'),
])
@pytest.mark.parametrize('all_gpu', [False, True])
def test_runtime_costs_remain_in_their_existing_inventory_and_display_policy(full, component, kind, owner, all_gpu):
    if not all_gpu:
        full['weight_storage']['intervals'][3]['assignment']['device_ids'] = ['cpu']
    if owner == 'adapter':
        full['owners'].append({'id': 'adapter', 'kind': 'adapter', 'roles': ['lora']})
    add_buffer(full, component, 17, kind, owner)
    result = model(full)
    device = 17 if kind == 'device' else 0
    assert result['size'] == (486+device if all_gpu else 527)
    assert result['size_vram'] == (486 if all_gpu else 482) + device
    assert result['memory']['allocated_bytes'] == 1367
    assert result['memory']['buffers'][-1]['size_bytes'] == 17
    reconcile(result)


def test_cpu_projector_compute_does_not_create_cpu_offload(full):
    add_buffer(full, 'projector_compute', 26136832, 'host', 'projector')
    result = model(full)
    assert result['size'] == result['size_vram'] == 486
    assert result['memory']['host_bytes'] == 26136882
    assert result['memory']['ollama_compatibility']['excluded_nonweight_host_bytes'] == 26136852
    reconcile(result)


def test_cpu_assignment_is_not_disguised_by_gpu_provider_or_device_buffer(full):
    item = full['weight_storage']['intervals'][0]
    item['assignment']['device_ids'] = ['cpu']
    result = model(full)
    assert (result['size'], result['size_vram']) == (510, 452)
    assert '100% GPU' not in processes_text(reported(full))
    reconcile(result)


def test_cpu_tied_input_output_interval_prevents_full_gpu_and_is_not_filtered(full):
    full['weight_storage']['intervals'][3]['assignment']['device_ids'] = ['cpu']
    result = model(full)
    assert (result['size'], result['size_vram']) == (510, 482)
    assert result['memory']['ollama_compatibility']['policy_excluded_bytes'] == 0
    reconcile(result)


def test_gpu_assigned_host_weights_can_exceed_device_capacity_without_measuring_vram(full):
    full['buffers'][0].update(memory_kind='host_pinned', device_ids=['gpu1'])
    result = model(full)
    assert result['size'] == result['size_vram'] == 486
    assert result['memory']['device_bytes'] == 300
    assert result['memory']['host_bytes'] == 1050
    assert result['memory']['physical_residency_bytes'] is None
    reconcile(result)  # The smoke must no longer require size_vram == device_bytes.


def test_unknown_weight_buffer_kind_does_not_override_proved_assignment(weights):
    weights['buffers'][0]['memory_kind'] = 'unknown'
    result = model(weights)
    assert (result['size'], result['size_vram']) == (510, 478)
    assert result['memory']['device_bytes'] is None
    assert result['memory']['unclassified_bytes'] == 1000
    reconcile(result)


def test_cpu_only_preserves_payload_and_host_runtime_costs(weights):
    for item in weights['weight_storage']['intervals']:
        item['assignment']['device_ids'] = ['cpu']
    for buffer in weights['buffers']:
        buffer['memory_kind'] = 'host'
    result = model(weights)
    assert (result['size'], result['size_vram']) == (510, 0)
    assert '100% CPU' in processes_text(reported(weights))
    reconcile(result)


def test_separate_cpu_draft_is_included_while_main_mtp_is_counted_once(full):
    full['owners'].append({'id': 'draft', 'kind': 'model', 'roles': ['draft']})
    draft = deepcopy(full['buffers'][0])
    draft.update(id='draft', owners=[{'owner_id': 'draft', 'component': 'model_weights'}])
    full['buffers'].append(draft)
    full['weight_storage']['intervals'].append(
        interval('draft', 'draft', 64, 34, relationship('repeating_layers', 39, 'draft'), devices=('cpu',)))
    result = model(full)
    assert (result['size'], result['size_vram']) == (544, 486)
    assert result['memory']['weight_storage_totals']['payload_bytes'] == 224
    assert result['memory']['ollama_compatibility']['policy_excluded_bytes'] == 0
    reconcile(result)


def test_shared_model_owners_roles_and_nonweight_owners_do_not_duplicate_bytes(full):
    full['owners'].append({'id': 'draft', 'kind': 'model', 'roles': ['draft']})
    full['owners'][0]['roles'].append('future_role')
    full['buffers'][0]['owners'].append({'owner_id': 'draft', 'component': 'model_weights'})
    full['weight_storage']['intervals'][0]['relationships'].append(relationship('repeating_layers', 39, 'draft'))
    full['buffers'][1]['owners'].append({'owner_id': 'projector', 'component': 'projector_compute'})
    result = model(full)
    assert result['size'] == result['size_vram'] == 486
    assert result['memory']['allocated_bytes'] == 1350
    reconcile(result)


@pytest.mark.parametrize('component,owner', [('compute', 'context'), ('projector_weights', 'projector'),
                                           ('future_component', 'model')])
def test_weight_and_nonweight_in_one_buffer_cannot_be_decomposed_as_capacity_minus_payload(weights, component, owner):
    weights['buffers'][0]['owners'].append({'owner_id': owner, 'component': component})
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    memory = result['memory']
    assert memory['weight_storage_totals']['payload_bytes'] == 190
    assert memory['allocated_bytes'] == 1350
    assert memory['ollama_compatibility']['reason'] == 'shared_weight_nonweight_buffer'
    assert memory['ollama_compatibility']['weight_capacity_reduction_bytes'] is None


@pytest.mark.parametrize('status', ['partial', 'unavailable', 'invalid'])
def test_incomplete_present_weight_section_never_falls_back_to_buffer_spans(weights, status):
    section = weights['weight_storage']
    if status == 'invalid':
        section['intervals'][0]['buffer_id'] = 'missing'
    else:
        section.update(status=status, missing_components=[{
            'owner_id': 'model', 'component': 'repeating_layers', 'layer_index': 39, 'reason': 'opaque_tensor_storage'}])
        if status == 'unavailable':
            section['intervals'] = []
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    memory = result['memory']
    assert memory['ollama_compatibility']['reason'] == 'incomplete_weight_storage'
    assert memory['allocated_bytes'] == 1350 and memory['source'] == 'native_memory_api'
    assert memory['fallback_reason'] is None
    assert memory['weight_storage_totals']['payload_bytes'] is None
    assert memory['weight_storage_totals']['known_bytes'] == (190 if status == 'partial' else 0)
    assert 'unknown' in processes_text(reported(weights))


@pytest.mark.parametrize('types', [['cpu', 'gpu0'], ['gpu1']])
def test_mixed_or_unknown_interval_assignment_cannot_be_divided_proportionally(weights, types):
    weights['devices'][2]['type'] = 'unknown'
    weights['weight_storage']['intervals'][0]['assignment']['device_ids'] = types
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    assert result['memory']['ollama_compatibility']['reason'] == 'unclassified_weight_assignment'
    assert result['memory']['weight_storage_totals']['payload_bytes'] == 190


def test_partial_weight_assignment_cannot_use_available_device_ids_as_complete(weights):
    section = weights['weight_storage']
    section['status'] = 'partial'
    section['intervals'][0]['assignment'].update(status='partial', reasons=['assignment_unknown'])
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    assert result['memory']['weight_storage_totals']['known_bytes'] == 190


def test_future_weight_component_keeps_geometry_but_does_not_guess_display_semantics(weights):
    weights['weight_storage']['intervals'][0]['relationships'][0]['component'] = 'future_weights'
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    assert result['memory']['ollama_compatibility']['reason'] == 'unsupported_weight_component'
    assert result['memory']['weight_storage_totals']['payload_bytes'] == 190


def test_unknown_nonweight_kind_prevents_a_display_split(weights):
    weights['buffers'][1]['memory_kind'] = 'unknown'
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    assert result['memory']['ollama_compatibility']['reason'] == 'unknown_nonweight_memory_kind'


def test_incomplete_buffer_inventory_cannot_hide_missing_runtime_costs(weights):
    weights['status'] = 'partial'
    result = model(weights)
    assert result['size'] is result['size_vram'] is None
    assert result['memory']['ollama_compatibility']['reason'] == 'incomplete_inventory'
    assert result['memory']['weight_storage_totals']['payload_bytes'] == 190


def test_weight_assignment_coverage_does_not_depend_on_placement_or_locality_exporters(full):
    full['placement'].update(status='partial', missing_components=[{
        'owner_id': 'model', 'component': 'model_weights', 'reason': 'collector_not_implemented'}])
    full['placement']['layer_groups'][1]['assignment']['device_ids'] = ['missing']
    full['allocation_locality']['bindings'][0]['allocation_id'] = 'missing'
    result = model(full)
    assert result['size'] == result['size_vram'] == 486
    assert result['memory']['extension_errors']
    assert result['memory']['allocation_locality']['status'] == 'unavailable'
    reconcile(result)


def test_older_v1_uses_explicit_capacity_metric_instead_of_claiming_payload(weights):
    del weights['weight_storage']
    result = model(weights)
    assert (result['size'], result['size_vram']) == (1350, 1300)
    summary = result['memory']['ollama_compatibility']
    assert summary['basis'] == 'native_weight_placement'
    assert summary['metric'] == 'backend_buffers' and summary['precision'] == 'exact_bytes'
    assert summary['weight_capacity_reduction_bytes'] == summary['policy_excluded_bytes'] == 0
    assert 'excluded_host_bytes' not in summary
    api_smoke.validate_processor_summary(result)


def test_log_measurement_keeps_own_metric_and_does_not_join_native_payload(weights):
    result = reported(weights, mode='log', compare=True)['models'][0]
    memory = result['memory']
    assert memory['source'] == 'native_load_log' and 'weight_storage' not in memory
    assert memory['ollama_compatibility']['metric'] == 'initial_load_allocations'
    assert memory['ollama_compatibility']['precision'] == 'rounded_mib'
    assert memory['comparison']['native']['weight_storage_totals']['payload_bytes'] == 190
    api_smoke.validate_processor_summary(result)


def test_known_empty_sleeping_worker_is_zero_in_both_metrics(full):
    full.update(model_state='sleeping', snapshot_stage='after_unload', owners=[], devices=[], buffers=[])
    full['placement'].update(status='known', layer_groups=[], missing_components=[])
    full['allocation_locality'].update(status='known', allocations=[], bindings=[], missing_components=[])
    full['weight_storage']['intervals'] = []
    result = model(full)
    assert result['size'] == result['size_vram'] == 0
    assert result['memory']['ollama_compatibility']['reason'] == 'empty_inventory'
    reconcile(result)


def test_memory_smoke_uses_same_metric_and_exclusions_as_ps(full):
    native = selected(full, compare=True)
    summary = memory_smoke.summarize(native, 'no-mmap')
    public = model(full)['memory']['ollama_compatibility']
    assert summary['display'] == public
    assert summary['totals']['allocated_bytes'] == 1350
    assert 'excluded_host_bytes' not in summary['display']


@pytest.mark.parametrize('bad', ['reduction', 'excluded', 'device', 'partial_pair'])
def test_api_smoke_detects_broken_reconciliation_and_sizes(full, bad):
    result = model(full)
    summary = result['memory']['ollama_compatibility']
    if bad == 'reduction':
        summary['weight_capacity_reduction_bytes'] = 0
    elif bad == 'excluded':
        summary['excluded_input_weight_bytes'] += 1
    elif bad == 'device':
        result['size_vram'] += 1
    else:
        result['size'] = summary['size'] = None
    with pytest.raises(WorkerError):
        api_smoke.validate_processor_summary(result)
