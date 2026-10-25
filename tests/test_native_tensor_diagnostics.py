"""Bounded tensor-role evidence is independent of deduplicated weight totals."""

from copy import deepcopy

import pytest

from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_sizes, public_memory
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights


@pytest.fixture
def diagnosed(weights):
    record = dict(id='t1', tensor_name=r'blk.39.\xff\\weight', owner_id='model',
        component='repeating_layers', layer_index=39, buffer_id='weights',
        storage_status='known', storage_reason=None, size_bytes=34, ranges_complete=True,
        ranges=[dict(offset_bytes=64, size_bytes=34)],
        assignment=deepcopy(weights['weight_storage']['intervals'][0]['assignment']))
    weights['weight_storage']['tensor_diagnostics'] = dict(status='known', record_limit=4096,
        range_limit=16384, truncated=False, name_encoding='byte_escape', missing_components=[],
        records=[record])
    return weights


def display(report):
    telemetry = MemoryTelemetry(observe(''), compare=True)
    telemetry.accept(report)
    return ollama_memory_sizes(public_memory(telemetry.snapshot()))


def diagnostic(payload):
    return payload['weight_storage']['tensor_diagnostics']


def missing(section, reason):
    section.update(status='partial', missing_components=[dict(owner_id=None,
        component='model_weights', layer_index=None, reason=reason)])


def test_names_roles_and_duplicate_tensor_visits_do_not_change_totals(diagnosed):
    original = NativeMemoryReport(diagnosed)
    second = deepcopy(diagnostic(diagnosed)['records'][0])
    second.update(id='t2', component='mtp_layers', layer_index=40)
    diagnostic(diagnosed)['records'].append(second)
    report = NativeMemoryReport(diagnosed)
    assert report.snapshot() == diagnosed
    assert report.weight_totals() == original.weight_totals()
    assert report.totals() == original.totals()
    assert display(report) == display(original) == (510, 478)
    copied = report.snapshot()
    diagnostic(copied)['records'][0]['ranges'].clear()
    assert report.snapshot() == diagnosed


@pytest.mark.parametrize('name', [None, '', 'plain ASCII', r'back\\slash', r'\x00\xff\xC3\xa9'])
def test_byte_escape_names_are_preserved_without_decoding(diagnosed, name):
    diagnostic(diagnosed)['records'][0]['tensor_name'] = name
    assert NativeMemoryReport(diagnosed).snapshot() == diagnosed


@pytest.mark.parametrize('reason', ['diagnostic_record_limit', 'diagnostic_range_limit',
                                  'diagnostic_collection_failed', 'future_reason'])
def test_partial_coverage_never_withdraws_main_storage(diagnosed, reason):
    section = diagnostic(diagnosed)
    missing(section, reason)
    section['truncated'] = reason.endswith('_limit')
    if reason == 'diagnostic_range_limit':
        section['records'][0].update(ranges_complete=False, ranges=[])
    parsed = NativeMemoryReport(diagnosed)
    assert parsed.snapshot() == diagnosed
    assert parsed.weight_totals().payload_bytes == 190
    assert display(parsed) == (510, 478)


def test_unavailable_geometry_and_unknown_role_assignment_are_independent(diagnosed):
    section = diagnostic(diagnosed)
    missing(section, 'opaque_tensor_storage')
    section['records'][0].update(tensor_name=None, component=None, layer_index=None,
        buffer_id=None, storage_status='unavailable', storage_reason='opaque_tensor_storage',
        size_bytes=None, ranges_complete=False, ranges=[],
        assignment=dict(status='unavailable', device_ids=[], sources=[], reasons=['assignment_unknown']))
    assert NativeMemoryReport(diagnosed).snapshot() == diagnosed
    section['records'][0].update(buffer_id='weights', component='future_weights', storage_status='known',
        storage_reason=None, size_bytes=0, ranges_complete=True)
    assert NativeMemoryReport(diagnosed).snapshot() == diagnosed


def test_empty_unloaded_and_failed_diagnostics_replace_old_records(diagnosed):
    telemetry = MemoryTelemetry(observe(''), compare=True)
    telemetry.accept(NativeMemoryReport(diagnosed))
    for status, reason in [('known', None), ('unavailable', 'model_inspection_unavailable')]:
        section = diagnostic(diagnosed)
        section.update(status=status, records=[], missing_components=[])
        if reason:
            section['missing_components'] = [dict(owner_id='model', component='model_weights',
                                                 layer_index=None, reason=reason)]
        diagnosed['snapshot_sequence'] += 1
        telemetry.accept(NativeMemoryReport(diagnosed))
        selected = telemetry.snapshot()['measurement']['report']
        assert diagnostic(selected) == section
        assert diagnostic(selected)['records'] == []
        assert selected['weight_storage']['status'] == 'known'


BAD_SECTION = [('record_limit', 0), ('record_limit', 4097), ('record_limit', True),
    ('range_limit', 0), ('range_limit', 16385), ('range_limit', False),
    ('truncated', 1), ('truncated', True), ('name_encoding', 'utf8'),
    ('records', {}), ('missing_components', None), ('status', 'future')]
BAD_RECORD = [('id', ''), ('owner_id', 'missing'), ('owner_id', 'projector'),
    ('buffer_id', 'missing'), ('buffer_id', 'staging'), ('buffer_id', None),
    ('component', None), ('layer_index', None), ('layer_index', True), ('layer_index', 2**64),
    ('tensor_name', '\xff'), ('tensor_name', '\\bad'), ('tensor_name', '\\xgg'),
    ('size_bytes', None), ('size_bytes', 33), ('size_bytes', 35), ('size_bytes', True),
    ('size_bytes', 2**64), ('storage_reason', 'failure'), ('storage_status', 'partial'),
    ('ranges_complete', 1), ('ranges_complete', False), ('ranges', []),
    ('ranges', [dict(offset_bytes=990, size_bytes=34)]),
    ('ranges', [dict(offset_bytes=2**64-1, size_bytes=34)]),
    ('ranges', [dict(offset_bytes=64, size_bytes=0)]),
    ('ranges', [dict(offset_bytes=True, size_bytes=34)]),
    ('ranges', [dict(offset_bytes=64, size_bytes=17), dict(offset_bytes=65, size_bytes=17)]),
    ('assignment', dict(status='known', device_ids=['absent'], sources=[], reasons=[]))]


@pytest.mark.parametrize('where,key,value', [('section', k, v) for k,v in BAD_SECTION]
                         + [('record', k, v) for k,v in BAD_RECORD])
def test_invalid_diagnostics_withdraw_only_diagnostics(diagnosed, where, key, value):
    before = NativeMemoryReport(diagnosed)
    target = diagnostic(diagnosed) if where == 'section' else diagnostic(diagnosed)['records'][0]
    target[key] = deepcopy(value)
    report = NativeMemoryReport(diagnosed)
    selected = report.snapshot()
    assert diagnostic(selected)['status'] == 'unavailable'
    assert diagnostic(selected)['records'] == []
    assert len(selected['extension_errors']) == 1
    assert selected['extension_errors'][0]['path'] == 'report.weight_storage.tensor_diagnostics'
    assert selected['weight_storage']['intervals'] == before.snapshot()['weight_storage']['intervals']
    assert report.weight_totals() == before.weight_totals()
    assert report.totals() == before.totals()
    assert display(report) == display(before)
    assert selected['allocation_locality'] == before.snapshot()['allocation_locality']


@pytest.mark.parametrize('mode', ['record_count', 'duplicate_id', 'range_count', 'global_range_count'])
def test_limits_and_ids_apply_across_records(diagnosed, mode):
    section = diagnostic(diagnosed)
    record = section['records'][0]
    if mode == 'record_count':
        section['record_limit'] = 1
        section['records'].append(dict(record, id='t2'))
    elif mode == 'duplicate_id':
        section['records'].append(deepcopy(record))
    else:
        diagnosed['buffers'][0]['size_bytes'] = 20000
        record['ranges'] = [dict(offset_bytes=i, size_bytes=1) for i in range(16384)]
        record['size_bytes'] = 16384
        if mode == 'range_count':
            record['ranges'].append(dict(offset_bytes=16384, size_bytes=1))
        else:
            section['records'].append(dict(record, id='t2', ranges=[dict(offset_bytes=0,size_bytes=1)], size_bytes=1))
    snapshot = NativeMemoryReport(diagnosed).snapshot()
    assert diagnostic(snapshot)['status'] == 'unavailable'
    assert snapshot['weight_storage']['status'] == 'known'


def test_full_range_budget_and_unknown_additive_fields(diagnosed):
    section = diagnostic(diagnosed)
    diagnosed['buffers'][0]['size_bytes'] = 16384
    section['records'][0].update(size_bytes=16384,
        ranges=[dict(offset_bytes=i, size_bytes=1, future=42) for i in range(16384)], future='ignored')
    section['future'] = 42
    selected = NativeMemoryReport(diagnosed).snapshot()
    assert 'extension_errors' not in selected
    assert len(diagnostic(selected)['records'][0]['ranges']) == 16384
    assert 'future' not in diagnostic(selected)['records'][0]['ranges'][0]


def test_invalid_diagnostics_can_recover_or_disappear_without_stale_records(diagnosed):
    telemetry = MemoryTelemetry(observe(''), compare=True)
    valid = deepcopy(diagnosed)
    for state in ['valid', 'invalid', 'valid', 'absent']:
        current = deepcopy(valid)
        current['snapshot_sequence'] += {'valid': 1, 'invalid': 1, 'absent': 2}[state]
        if state == 'invalid':
            diagnostic(current)['records'][0]['buffer_id'] = 'missing'
        elif state == 'absent':
            del current['weight_storage']['tensor_diagnostics']
        telemetry.accept(NativeMemoryReport(current))
        selected = telemetry.snapshot()['measurement']['report']
        if state == 'absent':
            assert 'tensor_diagnostics' not in selected['weight_storage']
        else:
            assert diagnostic(selected)['status'] == ('unavailable' if state == 'invalid' else 'known')
        assert selected['weight_storage']['status'] == 'known'
