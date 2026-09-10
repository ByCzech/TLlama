"""Where the accounting behind /api/ps is worked out.

It runs once, when the model is registered, over the whole parsed load
log. What registration then stores is the buckets and that result -- not
the log, and not the layer counts that were in it. /api/ps used to hand
the stored entry back through the same accounting, which silently asked
questions the entry could no longer answer and overwrote the correct
figures with the answers.

Measured on a Ryzen 5700G iGPU: registration put size_vram at 16695112499
bytes, equal to size, and /api/ps returned 16277936538 for the same
model. The client showed 2% CPU for a model with all 41 layers on
Vulkan0. The capacity cap went the same way -- device_residency is not in
the entry either, so on a card the load overflows the cap never applied
to anything a client saw.
"""

import pytest

from tllama.backend import ModelManager


BUCKETS = [
    f"{device}_{bucket}_mib"
    for device in ("gpu", "cpu")
    for bucket in ("model", "kv", "compute", "output", "rs", "projector")
] + [
    f"gpu_host_{bucket}_mib"
    for bucket in ("model", "kv", "compute", "output", "rs")
]


@pytest.fixture
def manager():
    return ModelManager.__new__(ModelManager)


def stats(**overrides):
    values = {name: 0.0 for name in BUCKETS}
    values.update(overrides)
    return values


def registered(manager, load_stats):
    """The entry _register_loaded_model builds: buckets and the result.

    Deliberately not the whole of load_stats. The point of these tests is
    what survives into the entry and what does not.
    """
    kept = {name: load_stats.get(name, 0.0) for name in BUCKETS}
    return {**kept, **manager._build_memory_accounting(load_stats)}


class TestTheAnswerDoesNotChangeOnTheWayOut:
    def test_ps_reports_what_registration_worked_out(self, manager):
        """The layer counts are gone from the entry by then."""
        load_stats = {
            **stats(gpu_model_mib=12634.81, gpu_kv_mib=2080.00, cpu_model_mib=397.85),
            "layer_devices_seen": True,
            "cpu_layers": 0,
            "gpu_layers": 41,
        }
        entry = registered(manager, load_stats)

        item = manager._with_runtime_totals(entry)

        assert item["size_vram"] == entry["ps_size_vram_bytes"]
        assert item["size"] == entry["ps_size_bytes"]

    def test_the_capacity_cap_survives_into_ps(self, manager):
        """device_residency is not in the entry, so it can only be read once."""
        load_stats = {
            **stats(gpu_model_mib=26000.00),
            "device_residency": {"capacity_mib": 16368.00, "overflow_mib": {}},
        }
        entry = registered(manager, load_stats)

        item = manager._with_runtime_totals(entry)

        assert item["size_vram"] == manager._mib_to_bytes(16368.00)
        assert item["size"] > item["size_vram"]

    def test_asking_twice_would_have_given_a_different_answer(self, manager):
        """Why reading the stored result is the fix and not tidying.

        Recomputing from the entry is not a second opinion, it is a worse
        one: the inputs that decided the first answer are not there.
        """
        load_stats = {
            **stats(gpu_model_mib=26000.00),
            "device_residency": {"capacity_mib": 16368.00, "overflow_mib": {}},
        }
        entry = registered(manager, load_stats)

        recomputed = manager._build_memory_accounting(entry)

        assert recomputed["ps_size_vram_mib"] != entry["ps_size_vram_mib"]


class TestWhatCountsAgainstVram:
    def test_a_projector_left_in_host_ram_still_counts(self, manager):
        """The case the layer-count guard got wrong.

        Every layer is on the device, so the guard zeroed the CPU side
        wholesale -- and took a projector computed on the CPU with it.
        That is work off the GPU, which is the whole subject of the
        split.
        """
        accounting = manager._build_memory_accounting({
            **stats(gpu_model_mib=12634.81, gpu_kv_mib=22.50, cpu_projector_mib=613.14),
            "layer_devices_seen": True,
            "cpu_layers": 0,
            "gpu_layers": 41,
        })

        assert accounting["ps_size_vram_mib"] == pytest.approx(
            accounting["ps_size_mib"] - 613.14
        )

    def test_a_kv_cache_on_the_cpu_counts(self, manager):
        accounting = manager._build_memory_accounting(
            stats(gpu_model_mib=6249.97, cpu_kv_mib=455.00)
        )

        assert accounting["ps_size_vram_mib"] == pytest.approx(
            accounting["ps_size_mib"] - 455.00
        )

    def test_host_side_helper_buffers_do_not(self, manager):
        """Pinned staging for the device's own work, not work on the CPU."""
        accounting = manager._build_memory_accounting(
            stats(gpu_model_mib=100.0, gpu_host_output_mib=0.95)
        )

        assert accounting["ps_size_vram_mib"] == accounting["ps_size_mib"]
