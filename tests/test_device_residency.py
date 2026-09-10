"""What a device holds, when the log and the device disagree.

A driver may take an allocation larger than the card and put the rest in
host memory. Measured on an RX 6900 XT of 16368.00 MiB: a load claiming
26334.90 MiB of it succeeded, and every buffer line in the log named that
card. The log cannot tell that apart from a load that fitted, so the
capacity is asked of the device instead.

The capacity is the total, never the free memory. In that same overflowed
state memory_free came back as 2**64 bytes, having been computed as total
minus used and wrapped around a size_t. The total stayed right.
"""

import pytest

from tllama.backend import _device_residency
from tllama.helpers.llama_stats import parse_llama_verbose_load_log


# Shortened from the run that overflowed. Vulkan1 is the RX 6900 XT.
OVERFLOWED = """load_tensors: layer   0 assigned to device Vulkan1, is_swa = 0
load_tensors: offloaded 1/1 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan1 model buffer size = 12634.81 MiB
llama_kv_cache: layer   1: dev = Vulkan1
llama_kv_cache:    Vulkan1 KV buffer size = 11520.00 MiB
llama_memory_recurrent:    Vulkan1 RS buffer size =    62.81 MiB
sched_reserve:    Vulkan1 compute buffer size =  2117.28 MiB
sched_reserve: Vulkan_Host compute buffer size =  2056.29 MiB
"""

FITTED = """load_tensors: layer   0 assigned to device CUDA0, is_swa = 0
load_tensors: offloaded 1/1 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:        CUDA0 model buffer size = 12634.81 MiB
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache:      CUDA0 KV buffer size =    65.00 MiB
sched_reserve:      CUDA0 compute buffer size =   497.00 MiB
"""

RX_6900_XT = 16368.00 * 1024 * 1024
TESLA_P100 = 16269.25 * 1024 * 1024


@pytest.fixture
def devices(monkeypatch):
    """Stand in for the ctypes enumeration, which needs a real library."""
    def factory(totals):
        monkeypatch.setattr("tllama.backend.device_memory_totals", lambda: totals)

    return factory


class TestReadingTheDevices:
    def test_a_claim_larger_than_the_card_is_noticed(self, devices):
        devices({"Vulkan1": RX_6900_XT})

        residency = _device_residency(parse_llama_verbose_load_log(OVERFLOWED))

        assert residency["overflow_mib"] == pytest.approx({"Vulkan1": 9966.90})

    def test_a_claim_that_fits_is_not(self, devices):
        devices({"CUDA0": TESLA_P100})

        residency = _device_residency(parse_llama_verbose_load_log(FITTED))

        assert residency["overflow_mib"] == {}

    def test_host_buffers_are_not_charged_to_the_device(self, devices):
        # Vulkan_Host carries a device's name and is pinned host memory.
        # Counting its 2056.29 MiB against the card would report an
        # overflow on a load that had none.
        devices({"Vulkan1": RX_6900_XT})

        residency = _device_residency(parse_llama_verbose_load_log(OVERFLOWED))

        assert "Vulkan_Host" not in residency["overflow_mib"]

    def test_capacity_is_the_sum_of_the_devices_asked_for(self, devices):
        devices({"Vulkan1": RX_6900_XT, "Vulkan0": TESLA_P100})

        residency = _device_residency(parse_llama_verbose_load_log(OVERFLOWED))

        assert residency["capacity_mib"] == pytest.approx(16368.00)

    def test_unreadable_devices_change_nothing(self, devices):
        # Before any model is loaded there are no devices to read:
        # llama_backend_init() is what fills the registry and until then
        # the count is zero.
        devices(None)

        assert _device_residency(parse_llama_verbose_load_log(OVERFLOWED)) == {}


class TestWhatReachesTheClient:
    def test_vram_is_capped_at_what_the_card_holds(self, manager):
        stats = parse_llama_verbose_load_log(OVERFLOWED)
        stats["device_residency"] = {"capacity_mib": 16368.00, "overflow_mib": {}}

        accounting = manager._build_memory_accounting(stats)

        assert accounting["ps_size_vram_mib"] == pytest.approx(16368.00)

    def test_the_rest_is_reported_as_not_in_vram(self, manager):
        # What the Ollama client divides to show its split.
        stats = parse_llama_verbose_load_log(OVERFLOWED)
        stats["device_residency"] = {"capacity_mib": 16368.00, "overflow_mib": {}}

        accounting = manager._build_memory_accounting(stats)

        assert accounting["ps_size_mib"] > accounting["ps_size_vram_mib"]

    def test_a_load_that_fits_is_left_alone(self, manager):
        stats = parse_llama_verbose_load_log(FITTED)
        stats["device_residency"] = {"capacity_mib": 16269.25, "overflow_mib": {}}

        accounting = manager._build_memory_accounting(stats)

        assert accounting["ps_size_vram_mib"] == accounting["ps_size_mib"]

    def test_without_device_capacity_nothing_is_capped(self, manager):
        accounting = manager._build_memory_accounting(
            parse_llama_verbose_load_log(OVERFLOWED)
        )

        assert accounting["ps_size_vram_mib"] == accounting["ps_size_mib"]
