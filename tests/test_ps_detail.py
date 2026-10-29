from copy import deepcopy

import httpx
import pytest

from tllama.client import TLlamaClient
from tllama.native_memory_api import NativeMemoryReport
from tllama.native_reporting import ollama_memory_sizes, public_memory
from tllama.native_telemetry import MemoryTelemetry
from test_native_memory import observe
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_weight_reporting import full


@pytest.mark.parametrize('mode', ['auto', 'native', 'log'])
@pytest.mark.parametrize('status', ['known', 'partial', 'unavailable'])
def test_cached_sizes_match_detailed_source_policy_and_do_not_copy_inventory(payload, mode, status, monkeypatch):
    payload['status'] = status
    telemetry = MemoryTelemetry(observe(), mode=mode, compare=True)
    telemetry.accept(NativeMemoryReport(payload))
    expected = ollama_memory_sizes(public_memory(telemetry.snapshot()))
    monkeypatch.setattr(telemetry, '_native_snapshot', lambda: pytest.fail('full diagnostics copied'))
    assert telemetry.sizes() == expected
    telemetry.fail('not_ready', http_status=503)
    assert telemetry.sizes() == ((None, None) if mode == 'native' else ollama_memory_sizes(telemetry.log.snapshot()))


@pytest.mark.parametrize('query,detailed', [('', False), ('?detailed=0', False), ('?detailed=false', False),
                                         ('?detailed=1', True), ('?detailed=true', True)])
def test_ps_wire_projection_and_boolean_query(client, payload, monkeypatch, query, detailed):
    from tllama.routers import ollama
    telemetry = MemoryTelemetry(observe())
    telemetry.accept(NativeMemoryReport(payload))
    entry = {'model': 'demo', 'metadata': {'arch': 'qwen3', 'params': 1_000_000_000, 'bits': 'Q4_0'},
             'file_info': {'content_sha256': 'a' * 64}, 'context_length': 4096, 'expires_at': None,
             'generation': 1, 'state': 'ready', 'pid': 123, 'leases': 0, 'retirement_reason': None}
    seen = []
    def snapshot(*, detailed=True):
        seen.append(detailed)
        return [dict(entry, **({'memory': telemetry.snapshot()} if detailed else {'memory_sizes': telemetry.sizes()}))]
    monkeypatch.setattr(ollama.model_manager, 'native_reporting_snapshot', snapshot)
    response = client.get('/api/ps' + query)
    assert response.status_code == 200
    [model] = response.json()['models']
    fields = {'name', 'model', 'digest', 'size', 'size_vram', 'context_length', 'expires_at', 'details', 'runner'}
    assert set(model) == fields | ({'generation', 'state', 'pid', 'leases', 'retirement_reason', 'memory'} if detailed else set())
    assert (model['size'], model['size_vram']) == telemetry.sizes()
    assert seen == [detailed]


def test_invalid_query_fails_before_snapshot(client, monkeypatch):
    from tllama.routers import ollama
    monkeypatch.setattr(ollama.model_manager, 'native_reporting_snapshot', lambda **kw: pytest.fail('snapshot requested'))
    assert client.get('/api/ps?detailed=wrong').status_code == 400


async def test_http_client_requests_explicit_detail():
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={'models': []})
    async with TLlamaClient(transport=httpx.MockTransport(handle)) as client:
        await client.ps()
        await client.ps(detailed=True)
    assert calls[0].endswith('/api/ps') and calls[1].endswith('/api/ps?detailed=true')


@pytest.mark.parametrize('fixture', ['weights', 'full'])
def test_cached_sizes_preserve_weight_payload_and_full_offload_exclusions(request, fixture, monkeypatch):
    telemetry = MemoryTelemetry(observe(), compare=True)
    telemetry.accept(NativeMemoryReport(request.getfixturevalue(fixture)))
    detailed = ollama_memory_sizes(public_memory(telemetry.snapshot()))
    monkeypatch.setattr(telemetry, '_native_snapshot', lambda: pytest.fail('diagnostics copied'))
    assert telemetry.sizes() == detailed
