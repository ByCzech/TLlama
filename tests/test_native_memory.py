"""Pinned load allocations, uncertainty, and bounded native memory observations."""

import pytest

from tllama.native_memory import NativeMemory


MIB = 1024**2
END = 'srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096'
# Allocation lines captured from api-vulkan2.log, native b11552-ec2797836f,
# Qwen3.6-35B-A3B-UD-IQ3_S, ctx=4096, K/V=q4_0/q4_0, October 14, 2026.
VULKAN = '''load_tensors: CPU_Mapped model buffer size = 397.85 MiB
load_tensors: Vulkan0 model buffer size = 12634.81 MiB
llama_context: Vulkan_Host output buffer size = 0.95 MiB
llama_kv_cache: Vulkan0 KV buffer size = 22.50 MiB
llama_memory_recurrent: Vulkan0 RS buffer size = 62.81 MiB
sched_reserve: Vulkan0 compute buffer size = 80.29 MiB
sched_reserve: Vulkan_Host compute buffer size = 20.30 MiB'''


def observe(log=VULKAN, *, end=True, **kwargs):
    parser = NativeMemory(**kwargs)
    for line in log.splitlines() + ([END] if end else []):
        parser.observe(line)
    return parser


def test_captured_vulkan_allocations_include_host_input_tensors_and_recurrent_state():
    report = observe().snapshot()
    assert report['status'] == 'known'
    assert report['buffers']['Vulkan0']['rs'] == round(62.81 * MIB)
    assert report['host_bytes'] == sum(round(value * MIB) for value in (397.85, 0.95, 20.30))
    assert report['device_bytes'] == sum(round(value * MIB) for value in (12634.81, 22.50, 62.81, 80.29))
    assert report['allocated_bytes'] == report['host_bytes'] + report['device_bytes']
    assert report['physical_residency_bytes'] is None
    assert report['scope'] == 'initial_load_allocations' and report['precision'] == 'rounded_mib'


@pytest.mark.parametrize('backend', ['CPU', 'CPU_Mapped', 'CUDA0', 'ROCm0', 'Vulkan1', 'SYCL0'])
def test_backend_allocations_have_explicit_host_or_device_location(backend):
    log = '\n'.join(f'{origin}: {backend} {kind} buffer size = {size} MiB' for origin, kind, size in
                    [('load_tensors','model',10), ('llama_kv_cache','KV',1), ('sched_reserve','compute',2)])
    report = observe(log).snapshot()
    assert report['allocated_bytes'] == 13 * MIB
    assert report['device_bytes'] == (0 if backend.startswith('CPU') else 13 * MIB)


def test_several_devices_are_summed_without_treating_host_staging_as_vram():
    log = VULKAN + '\nload_tensors: CUDA0 model buffer size = 10 MiB\nload_tensors: CUDA_Host model buffer size = 2 MiB'
    first = observe().snapshot()
    report = observe(log).snapshot()
    assert report['allocated_bytes'] == first['allocated_bytes'] + 12 * MIB
    assert report['device_bytes'] == first['device_bytes'] + 10 * MIB


def test_multiple_weight_buffers_are_counted_but_compute_reservations_are_replaced():
    log = VULKAN + '\nload_tensors: Vulkan0 model buffer size = 10 MiB\nsched_reserve: Vulkan0 compute buffer size = 100 MiB'
    report = observe(log).snapshot()
    assert report['buffers']['Vulkan0']['model'] == round(12634.81 * MIB) + 10 * MIB
    assert report['buffers']['Vulkan0']['compute'] == 100 * MIB


def test_reading_or_later_runtime_logs_do_not_recount_initial_allocations():
    parser = observe()
    first = parser.snapshot()
    for line in VULKAN.splitlines():
        parser.observe(line)
    parser.invalidate('late_runtime_line_truncated')
    second = parser.snapshot()
    assert first == second
    second['buffers']['Vulkan0']['model'] = 0
    assert parser.snapshot() == first


@pytest.mark.parametrize('size', ['NaN', 'Infinity', '-1', '1e100', 'broken'])
def test_malformed_allocation_is_unknown_and_never_zero(size):
    parser = observe(VULKAN + f'\nload_tensors: CPU model buffer size = {size} MiB')
    assert parser.snapshot()['allocated_bytes'] is None


@pytest.mark.parametrize('missing', ['load_tensors:', 'sched_reserve:', 'llama_kv_cache:'])
def test_missing_required_categories_do_not_invent_totals(missing):
    log = 'load_tensors: CPU model buffer size = 1 MiB\nllama_kv_cache: CPU KV buffer size = 1 MiB\nsched_reserve: CPU compute buffer size = 1 MiB'
    parser = observe('\n'.join(line for line in log.splitlines() if not line.startswith(missing)))
    assert parser.snapshot()['allocated_bytes'] is None


def test_unknown_device_name_keeps_total_but_not_a_fictitious_vram_split():
    report = observe(VULKAN.replace('Vulkan0', 'NewBackend0')).snapshot()
    assert report['allocated_bytes'] is not None
    assert report['device_bytes'] is None and report['reason'] == 'unknown_buffer_location'


@pytest.mark.parametrize('reason', ['truncated_load_log', 'parser_error'])
def test_truncation_or_parser_error_does_not_publish_a_partial_total(reason):
    parser = observe(end=False)
    parser.invalidate(reason)
    parser.observe(END)
    assert parser.snapshot()['reason'] == reason and parser.snapshot()['allocated_bytes'] is None


def test_pending_log_and_unsupported_runtime_are_explicit():
    assert observe(end=False).snapshot()['reason'] == 'load_log_pending'
    assert observe(runtime_fingerprint='new-runtime').snapshot()['reason'] == 'unsupported_runtime'


def test_projector_weight_summary_is_not_misreported_as_allocated_buffer():
    log = VULKAN + '\nclip_ctx: CLIP using Vulkan0 backend\nload_hparams: model size: 613.14 MiB\nreserve_compute_meta: Vulkan0 compute buffer size = 100 MiB'
    report = observe(log, projector=True).snapshot()
    assert report['projector_weights'] == {'Vulkan0':round(613.14 * MIB)}
    assert report['projector_weights_source'] == 'native_model_size_summary'
    assert report['buffers']['Vulkan0']['compute'] == round(80.29 * MIB)
    assert report['buffers']['Vulkan0']['projector_compute'] == 100 * MIB
    assert report['allocated_bytes'] is None and report['reason'] == 'projector_allocation_unreported'


def test_projector_size_cannot_be_paired_twice_or_across_contexts():
    log = VULKAN + '\nclip_ctx: CLIP using CPU backend\nclip_ctx: CLIP using Vulkan0 backend\nload_hparams: model size: 2 MiB'
    assert observe(log, projector=True).snapshot()['reason'] == 'multiple_projector_contexts'


def test_structured_observations_are_bounded():
    log = '\n'.join(f'load_tensors: Vulkan{i} model buffer size = 1 MiB' for i in range(260))
    report = observe(log).snapshot()
    assert report['reason'] == 'buffer_limit' and len(report['buffers']) == 256


def test_unknown_native_state_buffer_cannot_be_silently_dropped_from_total():
    report = observe(VULKAN + '\nllama_kv_cache_dsv4: Vulkan0 DSV4 sliding state buffer size = 5 MiB').snapshot()
    assert report['allocated_bytes'] is None and report['reason'] == 'unrecognized_buffer_report'
