"""What /api/ps reports about a loaded model's memory.

The numbers come from llama.cpp's own load-time accounting; these tests pin
how they are combined, which is the part that has to stay stable for an
Ollama client to show a sensible size and processor split.
"""

BUCKETS = [
    f"{device}_{bucket}_mib"
    for device in ("gpu", "cpu")
    for bucket in ("model", "kv", "compute", "output", "rs")
] + [
    f"gpu_host_{bucket}_mib"
    for bucket in ("model", "kv", "compute", "output", "rs")
]

MIB = 1024 * 1024


def all_buckets(value=0.0, **overrides):
    stats = {name: value for name in BUCKETS}
    stats.update(overrides)
    return stats


def test_resident_memory_is_model_plus_kv(manager):
    accounting = manager._build_memory_accounting(
        all_buckets(gpu_model_mib=100.0, gpu_kv_mib=20.0, cpu_model_mib=8.0, cpu_kv_mib=2.0)
    )

    assert accounting["gpu_loaded_mib"] == 120.0
    assert accounting["cpu_loaded_mib"] == 10.0
    assert accounting["loaded_total_mib"] == 130.0


def test_host_side_compute_staging_is_kept_out_of_the_reported_size(manager):
    """It is a pinned-memory fallback, not residency, and it distorts ps."""
    without = manager._build_memory_accounting(all_buckets(gpu_model_mib=100.0))
    with_staging = manager._build_memory_accounting(
        all_buckets(gpu_model_mib=100.0, gpu_host_compute_mib=500.0)
    )

    assert with_staging["ps_size_mib"] == without["ps_size_mib"]
    assert with_staging["total_runtime_mib"] > without["total_runtime_mib"]


def test_only_truly_cpu_resident_memory_counts_against_vram(manager):
    """GPU helper buffers stay GPU-associated in the processor split."""
    accounting = manager._build_memory_accounting(
        all_buckets(gpu_model_mib=100.0, cpu_model_mib=25.0, gpu_host_model_mib=7.0)
    )

    assert accounting["ps_size_vram_mib"] == accounting["ps_size_mib"] - 25.0


def test_the_reported_split_never_goes_negative(manager):
    accounting = manager._build_memory_accounting(all_buckets(cpu_model_mib=500.0))

    assert accounting["ps_size_vram_mib"] == 0.0


def test_missing_and_unusable_values_are_treated_as_zero(manager):
    """The stats come from parsed load output, so a gap is normal."""
    accounting = manager._build_memory_accounting({"gpu_model_mib": None, "gpu_kv_mib": "nonsense"})

    assert accounting["loaded_total_mib"] == 0.0
    assert accounting["ps_size_bytes"] == 0


def test_runtime_free_input_produces_a_complete_answer(manager):
    """Nothing may be absent from the reply just because a bucket was."""
    assert manager._build_memory_accounting({}).keys() == manager._build_memory_accounting(
        all_buckets(1.0)
    ).keys()


def test_runtime_totals_preserve_the_entry_they_decorate(manager):
    entry = {"model": "m", "path": "/models/m.gguf", **all_buckets(gpu_model_mib=64.0)}

    decorated = manager._with_runtime_totals(entry)

    assert decorated["model"] == "m"
    assert decorated["path"] == "/models/m.gguf"
    assert entry.keys() <= decorated.keys()


def test_runtime_totals_expose_the_names_an_ollama_client_reads(manager):
    decorated = manager._with_runtime_totals(
        all_buckets(gpu_model_mib=100.0, gpu_kv_mib=20.0, cpu_model_mib=10.0)
    )

    assert decorated["size"] == decorated["ps_size_bytes"]
    assert decorated["size_vram"] == decorated["ps_size_vram_bytes"]
    assert decorated["size_ram"] == decorated["cpu_loaded_bytes"]


def test_byte_fields_follow_their_mib_counterparts(manager):
    decorated = manager._with_runtime_totals(all_buckets(gpu_model_mib=3.0, gpu_kv_mib=1.0))

    assert decorated["gpu_loaded_bytes"] == int(4.0 * MIB)
    assert decorated["size"] == int(decorated["ps_size_mib"] * MIB)
