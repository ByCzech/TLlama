"""Synthetic CUDA/HIP/SYCL evidence from the pinned 0e9433bfd3 contract.

These examples exercise the Python reader/projection, not native allocation
control flow or hardware qualification. Native attempt/fallback tests live in
llama.cpp. Allocation spans, assignment and physical residency stay distinct.
"""

from copy import deepcopy

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import public_memory, ollama_memory_sizes
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_weight_reporting import full, model


INTEGRATED = ('cuda.device_integrated', 'hip.device_integrated',
              'level_zero.device_integrated')


@pytest.mark.parametrize('property_name', INTEGRATED)
@pytest.mark.parametrize('value', [0, 1])
def test_integrated_zero_and_one_survive_public_projection(full, property_name, value):
    properties = full['allocation_locality']['allocations'][0]['native_properties']
    properties[property_name] = value
    properties['level_zero.memory_ordinal'] = 0
    properties['private.handle'] = object()
    expected = {key: item for key, item in properties.items() if key != 'private.handle'}
    report = model(full)['memory']
    assert report['allocation_locality']['allocations'][0]['native_properties'] == expected
    assert 'extension_errors' not in report
    assert report['ollama_compatibility']['size'] == report['ollama_compatibility']['size_vram'] == 486


@pytest.mark.parametrize('property_name', INTEGRATED)
@pytest.mark.parametrize('value', [True, False, -1, 2, 2**64, 0.0, '1', None, [], {}])
def test_bad_integrated_property_withdraws_only_locality(full, property_name, value):
    full['allocation_locality']['allocations'][0]['native_properties'][property_name] = value
    report = model(full)['memory']
    assert report['allocation_locality']['status'] == 'unavailable'
    assert report['allocation_locality']['allocations'] == report['allocation_locality']['bindings'] == []
    assert report['extension_errors'][0]['path'] == 'report.allocation_locality'
    assert property_name in report['extension_errors'][0]['reason']
    assert report['allocated_bytes'] == 1350
    assert report['placement'] == full['placement']
    assert report['weight_storage'] == full['weight_storage']
    assert report['weight_storage_totals']['payload_bytes'] == 190
    assert ollama_memory_sizes(report) == (486, 486)


def test_absent_integrated_properties_are_not_reported_as_discrete(extended):
    properties = extended['allocation_locality']['allocations'][0]['native_properties']
    properties['level_zero.memory_ordinal'] = 7
    # Unlike the integrated attributes, ordinal and raw Vulkan flags are uint64.
    properties['vulkan.memory_property_flags'] = 2**64 - 1
    report = NativeMemoryReport(extended).snapshot()
    actual = report['allocation_locality']['allocations'][0]['native_properties']
    assert actual == properties
    assert not set(INTEGRATED).intersection(actual)


# Actual successful API names and conservative categories in the native source.
# The hipMalloc row is also the result of unsupported hipMallocManaged fallback.
# A failed attempt has no separate allocation record in the resulting snapshot.
PROFILES = [
    ('CUDA', 'gpu', 'cudaMalloc', 'device', 'device_local', {'cuda.device_integrated': 0}),
    ('CUDA', 'igpu', 'cudaMalloc', 'device', 'unknown', {'cuda.device_integrated': 1}),
    ('CUDA', 'gpu', 'cudaMalloc', 'device', 'unknown', {}),
    ('CUDA', 'gpu', 'cudaMallocManaged', 'managed', 'unknown', {'cuda.device_integrated': 0}),
    ('CUDA', 'gpu', 'cudaMallocHost', 'host', 'unknown', {}),
    ('CUDA', 'gpu', 'cudaMallocHost', 'host_pinned', 'unknown', {}),
    ('HIP', 'gpu', 'hipMalloc', 'device', 'device_local', {'hip.device_integrated': 0}),
    ('HIP', 'igpu', 'hipMalloc', 'device', 'unknown', {'hip.device_integrated': 1}),
    ('HIP', 'gpu', 'hipMallocManaged', 'managed', 'unknown', {'hip.device_integrated': 0}),
    ('HIP', 'gpu', 'hipHostMalloc', 'host_pinned', 'unknown', {}),
    ('SYCL', 'gpu', 'sycl::malloc_device', 'device', 'device_local', {'level_zero.device_integrated': 0}),
    ('SYCL', 'igpu', 'sycl::malloc_device', 'device', 'unknown', {'level_zero.device_integrated': 1}),
    ('SYCL', 'gpu', 'sycl::malloc_device', 'device', 'unknown', {}),
    ('SYCL', 'gpu', 'zeMemAllocDevice', 'device', 'device_local',
     {'level_zero.device_integrated': 0, 'level_zero.memory_ordinal': 0}),
    ('SYCL', 'igpu', 'zeMemAllocDevice', 'device', 'unknown',
     {'level_zero.device_integrated': 1, 'level_zero.memory_ordinal': 0}),
    ('SYCL', 'gpu', 'sycl::malloc_host', 'host', 'unknown', {}),
    ('SYCL', 'gpu', 'aligned_alloc', 'system', 'unknown', {}),
]


@pytest.fixture(params=PROFILES, ids=[f'{p[0]}-{p[2]}-{p[1]}-{p[3]}-{p[4]}' for p in PROFILES])
def backend_report(full, request):
    backend, device_type, api, kind, locality, properties = request.param
    full['devices'][1].update(backend=backend, type=device_type)
    allocation = {
        'id': 'actual-success', 'device_ids': ['gpu0'] if kind in {'device', 'managed'} else [],
        'kind': kind, 'origin': 'allocated', 'native_api': api, 'size_bytes': 2048,
        'locality': locality, 'native_properties': properties.copy(),
    }
    full['allocation_locality'].update(
        allocations=[allocation],
        bindings=[{'buffer_id': 'weights', 'allocation_id': 'actual-success',
                   'buffer_offset_bytes': 0, 'allocation_offset_bytes': 64, 'size_bytes': 1000}],
        missing_components=[{'buffer_id': b['id'], 'reason': 'unsupported_backend'}
                            for b in full['buffers'][1:]],
    )
    return full


def test_successful_backend_paths_preserve_evidence_without_recounting(backend_report):
    expected = deepcopy(backend_report)
    parsed = NativeMemoryReport(backend_report)
    assert parsed.snapshot() == expected
    result = model(backend_report)
    memory = result['memory']
    assert memory['allocation_locality'] == expected['allocation_locality']
    assert memory['weight_storage'] == expected['weight_storage']
    assert memory['allocated_bytes'] == 1350  # Not the padded native span of 2048.
    assert memory['weight_storage_totals']['payload_bytes'] == 190
    assert result['size'] == result['size_vram'] == 486
    assert memory['physical_residency_bytes'] is None
    assert backend_report == expected
    if expected['allocation_locality']['allocations'][0]['native_api'] == 'sycl::malloc_device':
        assert 'level_zero.memory_ordinal' not in memory['allocation_locality']['allocations'][0]['native_properties']


def test_locality_does_not_hide_actual_cpu_weight_assignment(backend_report):
    backend_report['weight_storage']['intervals'][0]['assignment']['device_ids'] = ['cpu']
    result = model(backend_report)
    assert (result['size'], result['size_vram']) == (510, 452)
    assert result['memory']['weight_storage_totals']['cpu_bytes'] == 38
    assert result['memory']['allocation_locality'] == backend_report['allocation_locality']
    assert result['memory']['physical_residency_bytes'] is None


@pytest.mark.parametrize('backend', ['CUDA', 'HIP', 'SYCL'])
def test_failed_host_attempt_cpu_fallback_has_no_stale_accelerator_metadata(full, backend):
    full['devices'][1]['backend'] = backend
    # The failed pinned/USM attempt returned an actual CPU buffer. CPU has no
    # exporter; it must not inherit metadata of the unsuccessful native attempt.
    full['buffers'][0].update(device_ids=['cpu'], buffer_type='CPU', memory_kind='host')
    full['allocation_locality'].update(
        status='unavailable', allocations=[], bindings=[],
        missing_components=[{'buffer_id': b['id'], 'reason': 'unsupported_backend'}
                            for b in full['buffers']],
    )
    result = model(full)
    memory = result['memory']
    assert memory['allocation_locality'] == full['allocation_locality']
    assert memory['source'] == 'native_memory_api' and memory['fallback_reason'] is None
    assert memory['host_bytes'] == 1050 and memory['device_bytes'] == 300
    assert memory['weight_storage_totals']['accelerator_bytes'] == 186
    assert result['size'] == result['size_vram'] == 486


def test_complete_relationships_with_unknown_allocation_attributes_remain_known(full):
    full['allocation_locality'].update(
        status='known', missing_components=[],
        allocations=[{'id': 'a', 'device_ids': [], 'kind': 'unknown', 'origin': 'unknown',
                      'native_api': None, 'size_bytes': None, 'locality': 'unknown',
                      'native_properties': {}}],
        bindings=[{'buffer_id': b['id'], 'allocation_id': 'a',
                   'buffer_offset_bytes': None, 'allocation_offset_bytes': None, 'size_bytes': None}
                  for b in full['buffers']],
    )
    report = model(full)['memory']
    assert report['allocation_locality'] == full['allocation_locality']
    assert report['allocation_locality']['status'] == 'known'
    assert report['weight_storage_totals']['payload_bytes'] == 190
    assert ollama_memory_sizes(report) == (486, 486)


def test_integrated_properties_are_replaced_with_each_snapshot_and_recover_after_error(full):
    properties = full['allocation_locality']['allocations'][0]['native_properties']
    properties['hip.device_integrated'] = 1
    telemetry = MemoryTelemetry(observe(), compare=True)
    telemetry.accept(NativeMemoryReport(full))
    saved = public_memory(telemetry.snapshot())

    full['snapshot_sequence'] += 1
    properties['hip.device_integrated'] = False
    telemetry.accept(NativeMemoryReport(full))
    invalid = public_memory(telemetry.snapshot())
    assert invalid['allocation_locality']['status'] == 'unavailable'
    assert invalid['weight_storage_totals']['payload_bytes'] == 190
    assert ollama_memory_sizes(invalid) == (486, 486)

    full['snapshot_sequence'] += 1
    del properties['hip.device_integrated']
    telemetry.accept(NativeMemoryReport(full))
    current = public_memory(telemetry.snapshot())
    assert 'extension_errors' not in current
    assert 'hip.device_integrated' not in current['allocation_locality']['allocations'][0]['native_properties']
    assert current['comparison']['native']['report']['allocation_locality'] == current['allocation_locality']
    assert 'allocation_locality' not in current['comparison']['log']['report']
    assert saved['allocation_locality']['allocations'][0]['native_properties']['hip.device_integrated'] == 1
    assert ollama_memory_sizes(current) == (486, 486)
