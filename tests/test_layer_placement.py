"""Which processor does the work, taken from where the layers went.

Buffer sizes cannot answer this. llama.cpp puts an input tensor whose
buffer type list holds no device -- the token embedding table, some
hundreds of MiB -- into a CPU buffer even when every layer is on the GPU,
and announces it under the same name it uses for half a model that really
is on the CPU. Measured on the same model and machine: 397.85 MiB fully
offloaded against 6782.69 MiB with n_gpu_layers = 20, both CPU_Mapped.

What separates them is one line per layer, saying where that layer went,
which llama.cpp writes for the weights and again for the KV cache.
"""

import pytest

from tllama.helpers.llama_stats import parse_llama_verbose_load_log


# All 41 layers on the device, and the embedding table left behind.
# Shortened from a real Vulkan load; the layer lines are verbatim.
FULLY_OFFLOADED = """load_tensors: layer   0 assigned to device Vulkan1, is_swa = 0
load_tensors: layer   1 assigned to device Vulkan1, is_swa = 0
done_getting_tensors: tensor 'token_embd.weight' (q6_K) (and 0 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan1 model buffer size = 12634.81 MiB
load_tensors: offloaded 41/41 layers to GPU
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: dev = Vulkan1
llama_kv_cache:    Vulkan1 KV buffer size =    65.00 MiB
"""

# n_gpu_layers = 20 on the same model: half the weights genuinely on the
# CPU, under the same CPU_Mapped name.
PARTLY_OFFLOADED = """load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer  20 assigned to device CPU, is_swa = 0
load_tensors: layer  21 assigned to device CUDA0, is_swa = 0
load_tensors:   CPU_Mapped model buffer size =  6782.69 MiB
load_tensors:        CUDA0 model buffer size =  6249.97 MiB
load_tensors: offloaded 20/41 layers to GPU
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  23: dev = CUDA0
llama_kv_cache:        CPU KV buffer size =    32.50 MiB
llama_kv_cache:      CUDA0 KV buffer size =    32.50 MiB
"""

# n_gpu_layers = 0.
NOT_OFFLOADED = """load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer  40 assigned to device CPU, is_swa = 0
load_tensors:   CPU_Mapped model buffer size = 13032.66 MiB
load_tensors: offloaded 0/41 layers to GPU
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache:        CPU KV buffer size =    65.00 MiB
"""


class TestCountingLayers:
    def test_a_fully_offloaded_load_has_none_on_the_cpu(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["gpu_layers"] == 2
        assert stats["cpu_layers"] == 0

    def test_a_split_load_is_counted_on_both_sides(self):
        stats = parse_llama_verbose_load_log(PARTLY_OFFLOADED)

        assert stats["cpu_layers"] == 2
        assert stats["gpu_layers"] == 1

    def test_a_kv_layer_with_no_cache_is_not_counted_anywhere(self):
        # 'filtered' means the layer has no KV cache at all -- a hybrid
        # model keeps recurrent state for most of its layers. Reading it
        # as a device name would put a phantom layer on the GPU.
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["gpu_kv_layers"] == 1
        assert stats["cpu_kv_layers"] == 0

    def test_kv_layers_are_counted_where_they_landed(self):
        stats = parse_llama_verbose_load_log(PARTLY_OFFLOADED)

        assert stats["cpu_kv_layers"] == 1
        assert stats["gpu_kv_layers"] == 1


class TestTheProcessorSplit:
    def test_weights_left_behind_do_not_make_it_a_cpu_load(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["processor"] == "GPU"

    def test_a_split_load_says_so(self):
        stats = parse_llama_verbose_load_log(PARTLY_OFFLOADED)

        assert stats["processor"] == "GPU+CPU"

    def test_nothing_offloaded_says_cpu(self):
        stats = parse_llama_verbose_load_log(NOT_OFFLOADED)

        assert stats["processor"] == "CPU"

    def test_a_log_without_layer_lines_says_so(self):
        stats = parse_llama_verbose_load_log(
            "load_tensors:      Vulkan1 model buffer size = 100.00 MiB\n"
        )

        assert stats["layer_devices_seen"] is False


class TestWhatReachesTheClient:
    """size_vram is the only thing on the wire that carries this.

    /api/ps sends Ollama's schema and there is no processor field in it;
    the client works the split out from size_vram against size. So the
    subtraction is where the change has to land.
    """

    def test_a_fully_offloaded_model_reports_all_of_it_as_vram(self, manager):
        accounting = manager._build_memory_accounting(
            parse_llama_verbose_load_log(FULLY_OFFLOADED)
        )

        assert accounting["ps_size_vram_mib"] == accounting["ps_size_mib"]

    def test_a_split_model_reports_less_vram_than_size(self, manager):
        accounting = manager._build_memory_accounting(
            parse_llama_verbose_load_log(PARTLY_OFFLOADED)
        )

        assert accounting["ps_size_mib"] - accounting["ps_size_vram_mib"] == pytest.approx(6815.19)

    def test_a_log_without_layer_lines_keeps_subtracting(self, manager):
        # The fallback has to stay conservative: without knowing where the
        # layers went, claiming it is all in VRAM would be a guess in the
        # direction that hides a problem.
        accounting = manager._build_memory_accounting(
            parse_llama_verbose_load_log(
                "load_tensors:   CPU_Mapped model buffer size =   397.85 MiB\n"
                "load_tensors:      Vulkan1 model buffer size = 12634.81 MiB\n"
            )
        )

        assert accounting["ps_size_mib"] - accounting["ps_size_vram_mib"] == pytest.approx(397.85)
