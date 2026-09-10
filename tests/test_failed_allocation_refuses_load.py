"""A load whose allocations did not all succeed is refused.

ggml writes the failure to the log and then, on the projector path,
returns success anyway: mtmd_init_from_file() does not check whether the
compute graph was reserved. The model arrives looking healthy and the
first request carrying an image kills the process.

The two lines being matched are both GGML_LOG_ERROR in
ggml/src/ggml-alloc.c and carry __func__ as their prefix. The buffer type
name in the middle is filled in by ggml_backend_buft_name(), so one
pattern serves every backend.
"""

import logging

import pytest

from tllama.helpers.llama_stats import parse_llama_verbose_load_log


# Taken from a CUDA load that went on to segfault on its first image.
RESERVE_FAILED = """get_dummy_batch: warmup with image size = 1472 x 1472
ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4657.81 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 4884066560
reserve_compute_meta: graph splits = 1, nodes = 877
warmup: flash attention is disabled
"""

WEIGHTS_FAILED = """load_tensors: loading model tensors, this can take a while...
alloc_tensor_range: failed to allocate Vulkan1 buffer of size 13246894080
"""

HEALTHY = """load_tensors:   CPU_Mapped model buffer size =   397.85 MiB
load_tensors:      Vulkan1 model buffer size = 12634.81 MiB
load_tensors: offloaded 41/41 layers to GPU
llama_kv_cache:    Vulkan1 KV buffer size =    65.00 MiB
sched_reserve:    Vulkan1 compute buffer size =   497.28 MiB
sched_reserve: graph splits = 2
reserve_compute_meta:    Vulkan1 compute buffer size =   248.10 MiB
"""


class TestFindingTheFailure:
    def test_a_failed_graph_reservation_is_found(self):
        stats = parse_llama_verbose_load_log(RESERVE_FAILED)

        assert stats["failed_allocations"] == [
            {
                "func": "ggml_gallocr_reserve_n_impl",
                "backend": "CUDA0",
                "bytes": 4884066560,
            }
        ]

    def test_a_failed_weight_allocation_is_found(self):
        # The other of the two lines, from the path that allocates weights
        # rather than a compute graph.
        stats = parse_llama_verbose_load_log(WEIGHTS_FAILED)

        assert stats["failed_allocations"] == [
            {
                "func": "alloc_tensor_range",
                "backend": "Vulkan1",
                "bytes": 13246894080,
            }
        ]

    def test_a_load_that_went_through_reports_nothing(self):
        # The counterweight, and the one that matters: this log has a
        # cudaMalloc-shaped vocabulary all over it and must not be read as
        # a failure.
        stats = parse_llama_verbose_load_log(HEALTHY)

        assert stats["failed_allocations"] == []

    def test_the_word_failed_elsewhere_is_not_enough(self):
        # ggml prints other failures that are recovered from or are about
        # something else entirely. Only the two allocation lines count,
        # which is why the function name is part of the match.
        stats = parse_llama_verbose_load_log(
            "ggml_vulkan: failed to allocate Vulkan1 buffer of size 1234\n"
            "clip_model_load: failed to load something else\n"
        )

        assert stats["failed_allocations"] == []


class TestRefusingTheLoad:
    def test_the_load_is_refused(self, manager, monkeypatch):
        monkeypatch.setattr(
            "tllama.backend.load_llama_with_captured_stats",
            lambda *args, **kwargs: (
                object(),
                parse_llama_verbose_load_log(RESERVE_FAILED),
                RESERVE_FAILED,
            ),
        )
        monkeypatch.setattr(
            manager, "_build_llama_load_kwargs", lambda *args, **kwargs: {}
        )

        with pytest.raises(RuntimeError) as raised:
            manager._load_model_sync("Local/model.gguf", 2048)

        assert "CUDA0" in str(raised.value)
        assert "4657.81 MiB" in str(raised.value)

    def test_it_names_the_setting_that_always_works(self, manager, monkeypatch):
        """The others may not apply; this one always does.

        flash_attn may already be on, n_ctx may already be where it is
        wanted, and warmup only exists for a projector. Fewer layers on
        the device always leaves it more room, and it is what Ollama
        reaches for on its own -- it estimates the fit and reduces the
        offload, which llama.cpp does not do and cannot be asked to. On
        the measured card Ollama loaded the same model at 25% CPU where
        this refused, and the difference is one setting the message did
        not mention.
        """
        monkeypatch.setattr(
            "tllama.backend.load_llama_with_captured_stats",
            lambda *args, **kwargs: (
                object(),
                parse_llama_verbose_load_log(RESERVE_FAILED),
                RESERVE_FAILED,
            ),
        )
        monkeypatch.setattr(
            manager, "_build_llama_load_kwargs", lambda *args, **kwargs: {}
        )

        with pytest.raises(RuntimeError) as raised:
            manager._load_model_sync("Local/model.gguf", 2048)

        assert "n_gpu_layers" in str(raised.value)

    def test_a_healthy_load_is_not_refused(self, manager, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            "tllama.backend.load_llama_with_captured_stats",
            lambda *args, **kwargs: (
                sentinel,
                parse_llama_verbose_load_log(HEALTHY),
                HEALTHY,
            ),
        )
        monkeypatch.setattr(
            manager, "_build_llama_load_kwargs", lambda *args, **kwargs: {}
        )

        llm, _, _ = manager._load_model_sync("Local/model.gguf", 2048)

        assert llm is sentinel

    def test_the_log_behind_the_refusal_is_shown(self, manager, monkeypatch, caplog):
        """The load worked, so its output went to DEBUG with every other.

        This one is not routine: the model is about to be discarded and
        those lines are why. They should not need a setting turned on
        after the fact, when the load is gone.
        """
        monkeypatch.setattr(
            "tllama.backend.load_llama_with_captured_stats",
            lambda *args, **kwargs: (
                object(),
                parse_llama_verbose_load_log(RESERVE_FAILED),
                RESERVE_FAILED,
            ),
        )
        monkeypatch.setattr(
            manager, "_build_llama_load_kwargs", lambda *args, **kwargs: {}
        )

        with caplog.at_level(logging.ERROR, logger="tllama.backend"):
            with pytest.raises(RuntimeError):
                manager._load_model_sync("Local/model.gguf", 2048)

        assert "failed to allocate" in caplog.text

    def test_the_largest_failure_is_the_one_named(self, manager, monkeypatch):
        # Several can be reported for one load. Naming the biggest gives
        # the number worth acting on rather than whichever came first.
        log = (
            "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 1048576\n"
            "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 4884066560\n"
        )
        monkeypatch.setattr(
            "tllama.backend.load_llama_with_captured_stats",
            lambda *args, **kwargs: (
                object(), parse_llama_verbose_load_log(log), log
            ),
        )
        monkeypatch.setattr(
            manager, "_build_llama_load_kwargs", lambda *args, **kwargs: {}
        )

        with pytest.raises(RuntimeError) as raised:
            manager._load_model_sync("Local/model.gguf", 2048)

        assert "4657.81 MiB" in str(raised.value)
        assert "and 1 more" in str(raised.value)
