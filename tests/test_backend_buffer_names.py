"""Which buffer types the load-log parser recognises.

llama.cpp names a buffer after whatever the buffer type calls itself, and
that set is open -- ggml/src carries a dozen backends and more arrive.
The parser used to hold a hand-written list of names, so anything absent
from it was read as nothing at all, silently and with no way to tell the
difference from a model that genuinely allocated nothing there.

The case that made this concrete is CPU_Mapped: the mmapped part of the
weights that stays in host RAM. It appears on every load in the captured
logs, on both Vulkan and CUDA, even when every layer is offloaded.
"""

from tllama.helpers.llama_stats import parse_llama_verbose_load_log

# Cut from a real load on an RX 6900 XT: 41 of 41 layers on the GPU, and
# 397.85 MiB of weights still in host RAM.
FULLY_OFFLOADED = """
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan1 model buffer size = 12634.81 MiB
llama_context: Vulkan_Host  output buffer size =     0.95 MiB
llama_kv_cache:    Vulkan1 KV buffer size =    22.50 MiB
sched_reserve:    Vulkan1 compute buffer size =   554.28 MiB
"""


class TestHostResidentWeightsAreSeen:
    def test_mapped_weights_land_in_the_cpu_bucket(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["cpu_model_mib"] == 397.85
        assert stats["buffers"]["CPU_Mapped"]["model"] == 397.85

    def test_they_change_the_reported_processor_split(self):
        """Weights in RAM are weights in RAM, however many layers offloaded."""
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["processor"] == "GPU+CPU"

    def test_the_gpu_side_is_unaffected(self):
        stats = parse_llama_verbose_load_log(FULLY_OFFLOADED)

        assert stats["gpu_model_mib"] == 12634.81
        assert stats["gpu_kv_mib"] == 22.50
        assert stats["gpu_compute_mib"] == 554.28
        assert stats["gpu_host_output_mib"] == 0.95


class TestOtherUnlistedBufferTypes:
    def test_the_cpu_variants_ggml_defines_are_all_cpu(self):
        stats = parse_llama_verbose_load_log(
            "CPU_HBM model buffer size = 10.00 MiB\n"
            "CPU_REPACK model buffer size = 20.00 MiB\n"
        )

        assert stats["cpu_model_mib"] == 30.0

    def test_a_backend_nobody_wrote_down_is_still_counted(self):
        """OpenCL, CANN, WebGPU and the rest were all read as nothing."""
        stats = parse_llama_verbose_load_log(
            "load_tensors:      OpenCL model buffer size =  1000.00 MiB\n"
        )

        assert stats["gpu_model_mib"] == 1000.0
        assert stats["processor"] == "GPU"

    def test_a_host_helper_stays_out_of_the_residency_split(self):
        stats = parse_llama_verbose_load_log(
            "CANN0 model buffer size = 100.00 MiB\n"
            "CANN_Host compute buffer size = 50.00 MiB\n"
        )

        assert stats["gpu_host_compute_mib"] == 50.0
        assert stats["gpu_model_mib"] == 100.0


class TestTheLooserPatternStillDiscriminates:
    def test_a_projector_size_line_is_not_read_as_a_buffer(self):
        """'load_hparams: model size:' has no 'buffer' in it."""
        stats = parse_llama_verbose_load_log(
            "load_hparams: model size:         613.14 MiB\n"
        )

        assert stats["gpu_model_mib"] == 0.0
        assert stats["cpu_model_mib"] == 0.0

    def test_a_failed_allocation_is_not_read_as_an_allocation(self):
        stats = parse_llama_verbose_load_log(
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4657.81 MiB "
            "on device 0: cudaMalloc failed: out of memory\n"
            "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 4884066560\n"
        )

        assert stats["gpu_compute_mib"] == 0.0
        assert stats["gpu_model_mib"] == 0.0

    def test_prose_around_the_numbers_is_not_a_backend_name(self):
        stats = parse_llama_verbose_load_log(
            "sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1\n"
            "sched_reserve: graph nodes  = 3847\n"
            "sched_reserve: reserve took 19.47 ms, sched copies = 1\n"
        )

        assert stats["gpu_mib"] == 0.0
        assert stats["cpu_mib"] == 0.0
        assert stats["buffers"] == {}
