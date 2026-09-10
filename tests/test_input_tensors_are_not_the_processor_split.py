"""What /api/ps counts as memory the processor split is about.

token_embd.weight is never on the device. Its buffer type list does not
contain one, so llama.cpp puts it in host RAM on every load, at every
offload setting, on a machine with 34 GB free the same as on a full one.
It serves a GGML_OP_GET_ROWS: one row copied per token. Nothing about it
moves work off the GPU.

Counting it made /api/ps report a fully offloaded model as partly on the
CPU -- 2% and 3% in the two measured cases -- which is the reading a
client draws from size_vram falling short of size, and it is wrong. It
also made the answer depend on use_mmap, a setting about how the weights
are read rather than about where anything is computed.

Numbers here are from four captured loads of Qwen3.6-35B-A3B-UD-IQ3_S on
a Ryzen 5700G iGPU, three fully offloaded and one at 20 of 41 layers.
"""

import pytest

from tllama.backend import ModelManager
from tllama.helpers.llama_stats import parse_llama_verbose_load_log


# 41/41 layers on the device. Everything but the model buffer on the CPU
# side is absent because there was nothing there.
FULLY_OFFLOADED = """
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan0 model buffer size = 12634.81 MiB
llama_context: Vulkan_Host  output buffer size =     0.95 MiB
llama_kv_cache:    Vulkan0 KV buffer size =  2080.00 MiB
llama_memory_recurrent:    Vulkan0 RS buffer size =    62.81 MiB
sched_reserve:    Vulkan0 compute buffer size =   745.28 MiB
sched_reserve: Vulkan_Host compute buffer size =   264.29 MiB
"""

# The same load with use_mmap off. llama.cpp stops calling the buffer
# CPU_Mapped and calls it Vulkan_Host; the bytes and the tensor are the
# same, and Vulkan0 does not change.
FULLY_OFFLOADED_WITHOUT_MMAP = FULLY_OFFLOADED.replace(
    "load_tensors:   CPU_Mapped model buffer size =   397.85 MiB",
    "load_tensors: Vulkan_Host model buffer size =   397.85 MiB",
)

# 20 of 41. The 6782.69 MiB covers token_embd.weight and every tensor of
# the 21 layers that stayed behind -- llama.cpp reported the move as
# "tensor 'token_embd.weight' (q6_K) (and 384 others)" and never says how
# the bytes divide.
PARTIALLY_OFFLOADED = """
load_tensors: offloaded 20/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  6782.69 MiB
load_tensors:      Vulkan0 model buffer size =  6249.97 MiB
llama_context: Vulkan_Host  output buffer size =     0.95 MiB
llama_kv_cache:        CPU KV buffer size =   455.00 MiB
llama_kv_cache:    Vulkan0 KV buffer size =   455.00 MiB
llama_memory_recurrent:        CPU RS buffer size =    33.50 MiB
llama_memory_recurrent:    Vulkan0 RS buffer size =    29.31 MiB
sched_reserve:    Vulkan0 compute buffer size =   503.26 MiB
sched_reserve:        CPU compute buffer size =    21.00 MiB
sched_reserve: Vulkan_Host compute buffer size =   160.46 MiB
"""


def ps_fields(log_text):
    """size and size_vram as /api/ps would report them.

    _build_memory_accounting() runs once when the model is registered and
    again out of the stored entry on every /api/ps, and the second result
    is what reaches a client. Going through both is the only way to ask
    the question a client asks.
    """
    manager = ModelManager.__new__(ModelManager)
    stats = parse_llama_verbose_load_log(log_text)
    entry = {**stats, **manager._build_memory_accounting(stats)}
    return manager._with_runtime_totals(entry)


class TestAFullyOffloadedModelIsAllOnTheDevice:
    def test_ps_does_not_report_any_of_it_outside_vram(self):
        item = ps_fields(FULLY_OFFLOADED)

        assert item["size_vram"] == item["size"]

    def test_the_excluded_weights_are_named_rather_than_dropped(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["input_model_mib"] == 397.85
        assert stats["cpu_model_mib"] == 0.0

    def test_the_log_is_still_recorded_as_it_was_written(self):
        """_device_residency() compares this against device capacity."""
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["buffers"]["CPU_Mapped"]["model"] == 397.85

    def test_the_device_side_keeps_every_byte(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["gpu_model_mib"] == 12634.81
        assert stats["gpu_kv_mib"] == 2080.00
        assert stats["gpu_rs_mib"] == 62.81
        assert stats["gpu_compute_mib"] == 745.28


class TestHowTheWeightsAreReadIsNotWhereTheWorkHappens:
    def test_turning_mmap_off_does_not_move_the_numbers(self):
        """The buffer changes name between CPU_Mapped and <device>_Host.

        Those names sit in different buckets, so before this the setting
        moved size_vram by the whole 397.85 MiB while the model, the
        layers and the work were identical.
        """
        with_mmap = ps_fields(FULLY_OFFLOADED)
        without_mmap = ps_fields(FULLY_OFFLOADED_WITHOUT_MMAP)

        assert without_mmap["size"] == with_mmap["size"]
        assert without_mmap["size_vram"] == with_mmap["size_vram"]


class TestAPartlyOffloadedModelStillSaysSo:
    def test_the_cpu_share_survives(self):
        """21 layers and half the KV cache really are computed there."""
        item = ps_fields(PARTIALLY_OFFLOADED)

        assert item["size_vram"] < item["size"]

    def test_the_buffer_is_left_whole_because_the_log_does_not_divide_it(self):
        stats = parse_llama_verbose_load_log(PARTIALLY_OFFLOADED)

        assert stats["cpu_model_mib"] == 6782.69
        assert stats["input_model_mib"] == 0.0

    def test_what_it_costs_to_leave_it_whole(self):
        """The measured size of the part that should not be there.

        Small enough that the answer -- the CPU is computing -- does not
        change, which is why the buffer is left alone rather than having
        a number invented for it.
        """
        item = ps_fields(PARTIALLY_OFFLOADED)

        outside_vram = (item["size"] - item["size_vram"]) / 1024 / 1024
        assert outside_vram == pytest.approx(7237.69, abs=0.01)
