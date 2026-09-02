"""Reading a projector's weights out of a llama.cpp load log.

Every other buffer llama.cpp reports names its backend on the same line.
A projector does not: it announces the backend once, in clip_ctx, and the
size several lines later under a load_hparams prefix shared with a dozen
unrelated lines. Until the two were paired, those weights -- 613 MiB for
the model these excerpts came from -- were parsed by nothing.

The excerpts below are cut from real runs on real hardware: Vulkan on an
RX 6900 XT, CUDA on a Tesla P100, and the same Vulkan load with flash
attention off, which changes the surrounding numbers a great deal and the
two lines that matter not at all.
"""

from tllama.helpers.llama_stats import parse_llama_verbose_load_log

# RX 6900 XT, Vulkan, flash attention enabled.
VULKAN_PROJECTOR = """
clip_ctx: CLIP using Vulkan1 backend
load_hparams: projector:          qwen3vl_merger
load_hparams: n_embd:             1152
load_hparams: image_size:         768
load_hparams: patch_size:         16
load_hparams: image_min_pixels:   8192
load_hparams: image_max_pixels:   4194304

load_hparams: model size:         613.14 MiB
load_hparams: metadata size:      0.12 MiB
load_tensors: loaded 334 tensors from mmproj/qwen3.6-mmproj-Q8_0.gguf
get_dummy_batch: warmup with image size = 1472 x 1472
reserve_compute_meta:    Vulkan1 compute buffer size =   248.10 MiB
reserve_compute_meta:        CPU compute buffer size =    24.93 MiB
reserve_compute_meta: graph splits = 1, nodes = 823
warmup: flash attention is enabled
"""

# Tesla P100, CUDA, flash attention enabled.
CUDA_PROJECTOR = """
clip_ctx: CLIP using CUDA0 backend
load_hparams: model size:         613.14 MiB
load_hparams: metadata size:      0.12 MiB
reserve_compute_meta:      CUDA0 compute buffer size =   248.10 MiB
reserve_compute_meta:        CPU compute buffer size =    24.93 MiB
warmup: flash attention is enabled
"""

# The same Tesla, flash attention off. The vision attention matrix is
# materialised in full, the compute buffer reservation fails outright, and
# no compute buffer line is printed at all -- but the weights are still
# resident and still reported the same way.
CUDA_PROJECTOR_RESERVE_FAILED = """
clip_ctx: CLIP using CUDA0 backend
load_hparams: model size:         613.14 MiB
load_hparams: metadata size:      0.12 MiB
get_dummy_batch: warmup with image size = 1472 x 1472
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4657.81 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 4884066560
reserve_compute_meta: graph splits = 1, nodes = 877
warmup: flash attention is disabled
"""

TEXT_ONLY = """
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan1 model buffer size = 12634.81 MiB
llama_context: n_ctx                 = 4096
llama_kv_cache:    Vulkan1 KV buffer size =    22.50 MiB
sched_reserve:    Vulkan1 compute buffer size =   554.28 MiB
"""


class TestTheWeightsAreFound:
    def test_a_vulkan_projector_lands_on_its_device(self):
        stats = parse_llama_verbose_load_log(VULKAN_PROJECTOR)

        assert stats["gpu_projector_mib"] == 613.14
        assert stats["cpu_projector_mib"] == 0.0
        assert stats["buffers"]["Vulkan1"]["projector"] == 613.14

    def test_a_cuda_projector_lands_on_its_device(self):
        stats = parse_llama_verbose_load_log(CUDA_PROJECTOR)

        assert stats["gpu_projector_mib"] == 613.14
        assert stats["buffers"]["CUDA0"]["projector"] == 613.14

    def test_the_weights_are_read_even_when_the_reservation_failed(self):
        """The compute buffer never existed; the weights did."""
        stats = parse_llama_verbose_load_log(CUDA_PROJECTOR_RESERVE_FAILED)

        assert stats["gpu_projector_mib"] == 613.14
        assert stats["gpu_compute_mib"] == 0.0

    def test_a_projector_on_the_cpu_is_counted_as_cpu(self):
        stats = parse_llama_verbose_load_log(
            "clip_ctx: CLIP using CPU backend\n"
            "load_hparams: model size:         613.14 MiB\n"
        )

        assert stats["cpu_projector_mib"] == 613.14
        assert stats["gpu_projector_mib"] == 0.0


class TestNothingElseIsMistakenForIt:
    def test_a_text_only_load_reports_no_projector(self):
        stats = parse_llama_verbose_load_log(TEXT_ONLY)

        assert stats["gpu_projector_mib"] == 0.0
        assert stats["cpu_projector_mib"] == 0.0

    def test_a_text_only_load_is_otherwise_unchanged(self):
        """The pairing must not disturb what was already parsed."""
        stats = parse_llama_verbose_load_log(TEXT_ONLY)

        assert stats["gpu_model_mib"] == 12634.81
        assert stats["gpu_kv_mib"] == 22.50
        assert stats["gpu_compute_mib"] == 554.28
        assert stats["offloaded_layers"] == 41

    def test_other_load_hparams_lines_are_not_read_as_a_size(self):
        """'load_hparams:' prefixes n_embd, image_size and much else."""
        stats = parse_llama_verbose_load_log(
            "clip_ctx: CLIP using Vulkan1 backend\n"
            "load_hparams: n_embd:             1152\n"
            "load_hparams: image_size:         768\n"
            "load_hparams: metadata size:      0.12 MiB\n"
        )

        assert stats["gpu_projector_mib"] == 0.0

    def test_a_size_line_before_any_clip_ctx_is_ignored(self):
        """Order matters: the backend has to be announced first."""
        stats = parse_llama_verbose_load_log(
            "load_hparams: model size:         613.14 MiB\n"
            "clip_ctx: CLIP using Vulkan1 backend\n"
        )

        assert stats["gpu_projector_mib"] == 0.0

    def test_the_projector_is_not_folded_into_the_model_bucket(self):
        """They are separate allocations and ps has to tell them apart."""
        stats = parse_llama_verbose_load_log(TEXT_ONLY + VULKAN_PROJECTOR)

        assert stats["gpu_model_mib"] == 12634.81
        assert stats["gpu_projector_mib"] == 613.14


def test_two_projectors_in_one_log_are_both_counted():
    """A capture window can span more than one load."""
    stats = parse_llama_verbose_load_log(VULKAN_PROJECTOR + CUDA_PROJECTOR)

    assert stats["buffers"]["Vulkan1"]["projector"] == 613.14
    assert stats["buffers"]["CUDA0"]["projector"] == 613.14
    assert stats["gpu_projector_mib"] == 1226.28
