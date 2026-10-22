"""Optional V1 evidence stays separate from capacity and processor summaries."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_sizes, public_memory
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload, assign


@pytest.fixture
def extended(payload):
    # Synthetic many-to-many bindings, padded native span and imported alias.
    for group in payload['placement']['layer_groups']:
        group['assignment'] = {'status':'known', 'device_ids':group['device_ids'][:],
                               'sources':['loader_selection'], 'reasons':[]}
    payload['allocation_locality'] = {
        'status':'partial', 'scope':'backend_buffer_allocations', 'physical_aliasing':'unknown',
        'missing_components':[{'buffer_id':'input','reason':'unsupported_backend'}],
        'allocations':[
            {'id':'a0','device_ids':['gpu0'],'kind':'device','origin':'allocated',
             'native_api':'vkAllocateMemory','size_bytes':2048,'locality':'device_local',
             'native_properties':{'vulkan.memory_type_index':0,'vulkan.heap_index':0,
                                 'vulkan.memory_property_flags':7,'vulkan.heap_flags':1}},
            {'id':'a1','device_ids':[],'kind':'managed','origin':'imported',
             'native_api':None,'size_bytes':None,'locality':'unknown','native_properties':{}},
        ],
        'bindings':[
            {'buffer_id':'weights','allocation_id':'a0','buffer_offset_bytes':0,
             'allocation_offset_bytes':0,'size_bytes':600},
            {'buffer_id':'weights','allocation_id':'a1','buffer_offset_bytes':600,
             'allocation_offset_bytes':None,'size_bytes':400},
            {'buffer_id':'shared','allocation_id':'a0','buffer_offset_bytes':None,
             'allocation_offset_bytes':1024,'size_bytes':None},
        ],
    }
    return payload


def selected(payload):
    telemetry = MemoryTelemetry(observe(), compare=True)
    telemetry.accept(NativeMemoryReport(payload))
    return public_memory(telemetry.snapshot())


def test_optional_evidence_is_copied_without_recounting_padding_or_aliases(extended):
    expected = deepcopy(extended)
    parsed = NativeMemoryReport(extended)
    extended['allocation_locality']['bindings'].clear()
    returned = parsed.snapshot()
    assert returned == expected
    returned['allocation_locality']['allocations'].clear()
    returned['placement']['layer_groups'][0]['assignment']['device_ids'].clear()
    assert parsed.snapshot() == expected
    assert parsed.totals().allocated_bytes == 1350
    assert parsed.totals().device_bytes == 1300


@pytest.mark.parametrize('kind', ['device','host','host_pinned','managed','shared','system','unknown'])
@pytest.mark.parametrize('origin', ['allocated','imported','registered','unknown'])
def test_allocation_kind_and_origin_do_not_reclassify_buffer_capacity(extended, kind, origin):
    allocation = extended['allocation_locality']['allocations'][0]
    allocation.update(kind=kind, origin=origin, locality='non_device_local')
    report = NativeMemoryReport(extended)
    assert report.snapshot()['allocation_locality']['allocations'][0] == allocation
    assert report.totals().allocated_bytes == 1350 and report.totals().host_bytes == 50


def test_unknown_optional_fields_and_native_handles_are_not_exposed(extended):
    expected = deepcopy(extended)
    extended['allocation_locality']['future'] = {'anything':True}
    extended['allocation_locality']['allocations'][0]['native_properties']['private.handle'] = object()
    extended['allocation_locality']['bindings'][0]['future'] = 42
    extended['placement']['layer_groups'][0]['assignment']['future'] = None
    assert NativeMemoryReport(extended).snapshot() == expected


@pytest.mark.parametrize('path,value', [
    ('allocations.0.size_bytes', True), ('allocations.0.size_bytes', -1),
    ('allocations.0.size_bytes', 2**64), ('allocations.0.locality','vram'),
    ('allocations.0.device_ids',['missing']), ('allocations.0.device_ids',['gpu0','gpu0']),
    ('allocations.1.id','a0'), ('allocations.0.native_properties',None),
    ('bindings.0.buffer_id','missing'), ('bindings.0.allocation_id','missing'),
    ('bindings.0.buffer_offset_bytes',True), ('bindings.0.buffer_offset_bytes',1001),
    ('bindings.0.allocation_offset_bytes',2049), ('bindings.0.size_bytes',1001),
    ('bindings.0.allocation_offset_bytes',2**64-1),
    ('missing_components.0.buffer_id','missing'), ('status','known'),
    ('physical_aliasing','disjoint'), ('scope','physical_residency'),
])
def test_bad_locality_is_withdrawn_without_invalidating_capacity_or_assignment(extended, path, value):
    assign(extended['allocation_locality'], path, value)
    report = selected(extended)
    assert report['status'] == 'known' and report['allocated_bytes'] == 1350
    assert report['allocation_locality']['status'] == 'unavailable'
    assert report['allocation_locality']['allocations'] == report['allocation_locality']['bindings'] == []
    assert report['allocation_locality']['missing_components'] == [{'buffer_id':None,'reason':'invalid_metadata'}]
    assert report['extension_errors'][0]['path'] == 'report.allocation_locality'
    assert report['native']['support'] == 'supported' and report['fallback_reason'] is None
    assert report['placement']['layer_groups'][0]['assignment'] == extended['placement']['layer_groups'][0]['assignment']


def test_native_property_values_are_unsigned_and_zero_is_not_missing(extended):
    props = extended['allocation_locality']['allocations'][0]['native_properties']
    props['level_zero.memory_ordinal'] = 0
    assert NativeMemoryReport(extended).snapshot()['allocation_locality']['allocations'][0]['native_properties'] == props
    props['level_zero.memory_ordinal'] = False
    assert NativeMemoryReport(extended).snapshot()['extension_errors']


def test_duplicate_bindings_are_not_published_twice(extended):
    extended['allocation_locality']['bindings'] *= 2
    assert NativeMemoryReport(extended).snapshot()['allocation_locality']['status'] == 'unavailable'


@pytest.mark.parametrize('field', ['buffer_offset_bytes','allocation_offset_bytes','size_bytes'])
def test_each_range_field_can_be_unknown_independently(extended, field):
    extended['allocation_locality']['bindings'][0][field] = None
    assert NativeMemoryReport(extended).snapshot() == extended


def test_known_relationship_coverage_can_have_unknown_attributes_and_buffer_capacity(extended):
    extended['status'] = 'partial'
    extended['buffers'][0]['size_bytes'] = None
    section = extended['allocation_locality']
    section.update(status='known',missing_components=[])
    for allocation in section['allocations']:
        allocation.update(size_bytes=None,device_ids=[],locality='unknown',native_properties={})
    for buffer in extended['buffers'][2:]:
        section['bindings'].append({'buffer_id':buffer['id'],'allocation_id':'a1',
            'buffer_offset_bytes':None,'allocation_offset_bytes':None,'size_bytes':None})
    report = NativeMemoryReport(extended)
    assert report.snapshot()['allocation_locality']['status'] == 'known'
    assert report.totals().allocated_bytes is None


@pytest.mark.parametrize('status', ['partial','unavailable'])
def test_missing_exporters_do_not_trigger_log_fallback_or_erase_known_capacity(extended, status):
    extended['allocation_locality'].update(status=status,allocations=[],bindings=[])
    report = selected(extended)
    assert report['source'] == 'native_memory_api' and report['status'] == 'known'
    assert report['allocated_bytes'] == 1350 and report['fallback_reason'] is None
    assert report['allocation_locality']['status'] == status
    assert report['comparison']['native']['report']['allocation_locality']['status'] == status
    assert 'allocation_locality' not in report['comparison']['log']['report']


@pytest.mark.parametrize('path,value', [
    ('device_ids',['missing']), ('device_ids',['cpu','cpu']), ('status','complete'),
    ('reasons',['conflict']), ('sources',['future_source']), ('sources',['loader_mmap','loader_mmap']),
    ('device_ids',[]),
])
def test_invalid_assignment_cannot_silently_become_a_provider_guess(extended, path, value):
    assign(extended['placement']['layer_groups'][0]['assignment'],path,value)
    report = selected(extended)
    assert report['status'] == 'known' and report['allocated_bytes'] == 1350
    assert report['placement']['status'] == 'known'
    assert report['placement']['layer_groups'][0]['assignment']['status'] == 'unavailable'
    assert report['allocation_locality']['status'] == 'partial'
    assert ollama_memory_sizes(report) == (None,None)


def test_partial_assignment_remains_visible_without_claiming_full_gpu(extended):
    assignment = extended['placement']['layer_groups'][1]['assignment']
    assignment.update(status='partial',reasons=['assignment_unknown'])
    report = selected(extended)
    assert report['placement']['layer_groups'][1]['assignment'] == assignment
    assert ollama_memory_sizes(report) == (None,None)


def test_contradictory_assignment_does_not_invent_a_processor_split(extended):
    extended['placement']['layer_groups'][1]['assignment']['device_ids'] = ['cpu']
    assert ollama_memory_sizes(selected(extended)) == (None,None)


def test_locality_only_snapshot_change_and_sleep_reload_do_not_join_old_ids(extended):
    telemetry = MemoryTelemetry(observe())
    telemetry.accept(NativeMemoryReport(extended))
    before = telemetry.snapshot()
    extended['snapshot_sequence'] += 1
    extended['allocation_locality']['allocations'][0]['locality'] = 'non_device_local'
    telemetry.accept(NativeMemoryReport(extended))
    assert before['measurement']['report']['allocation_locality']['allocations'][0]['locality'] == 'device_local'
    assert telemetry.snapshot()['measurement']['report']['allocation_locality']['allocations'][0]['locality'] == 'non_device_local'
    loaded = deepcopy(extended)
    extended.update(snapshot_sequence=19,model_state='sleeping',snapshot_stage='after_unload',owners=[],devices=[],buffers=[])
    extended['placement']['layer_groups'] = []
    extended['allocation_locality'].update(status='known',missing_components=[],allocations=[],bindings=[])
    telemetry.accept(NativeMemoryReport(extended))
    assert telemetry.snapshot()['measurement']['totals']['allocated_bytes'] == 0
    loaded['snapshot_sequence'] = 20
    loaded['allocation_locality']['bindings'][0]['allocation_id'] = 'old-snapshot-only'
    telemetry.accept(NativeMemoryReport(loaded))
    assert telemetry.snapshot()['measurement']['report']['allocation_locality']['status'] == 'unavailable'
    assert telemetry.snapshot()['measurement']['totals']['allocated_bytes'] == 1350


# Exact endpoint JSON from the maintainer's 2026-10-17 four-profile run,
# tllama-memory-08c-20261017-124635.tar.xz, native b11552-3144c167ef.
# Archive SHA256: 1e7e8e9677b28233d333e67e09a09e08e8639fd12f5e65ac6a10a1001ece291b.
# These are backend spans/allocation evidence, never a physical residency claim.
@pytest.mark.parametrize('profile,total,host,device,display,status,count', [
    ('no-mmap',13861655144,439457888,13422197256,13422197256,'known',19),
    ('cpu',13840372288,13840372288,0,13840372288,'unavailable',0),
    ('partial',14001074752,7230632000,6770442752,14001074752,'partial',12),
    ('vision',14801355656,465594688,14335760968,14335760968,'partial',20),
    ('vision-after-images',14801355656,465594688,14335760968,14335760968,'partial',20),
])
def test_captured_native_reports_survive_adapter_and_public_projection(profile,total,host,device,display,status,count):
    raw = json.loads((Path(__file__).with_name('fixtures')/'native_memory_locality'/f'{profile}.json').read_text())
    parsed = NativeMemoryReport(raw)
    assert parsed.snapshot() == raw
    report = selected(raw)
    assert (report['allocated_bytes'],report['host_bytes'],report['device_bytes']) == (total,host,device)
    assert ollama_memory_sizes(report) == (display,device)
    section = report['allocation_locality']
    assert section['status'] == status and len(section['allocations']) == count
    assert len(section['bindings']) == count
    if profile == 'no-mmap':
        group = report['placement']['layer_groups'][0]
        assert group['location'] == 'cpu' and group['device_ids'] == ['d1']
        assert group['assignment']['device_ids'] == ['d2']
        local = section['allocations'][0]
        assert local['native_properties']['vulkan.memory_property_flags'] == 7
        assert local['locality'] == 'device_local'  # HOST_VISIBLE too.
    if profile.startswith('vision'):
        assert any(b['memory_kind'] == 'host' and b['size_bytes'] == 26136832
                   and any(o['component'] == 'projector_compute' for o in b['owners']) for b in report['buffers'])
