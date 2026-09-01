"""The projector is brought up at load time, not at the first image.

MTMDChatHandler initialises its mtmd context lazily. Left alone, that
puts the projector's weights and compute buffers outside the window
load_llama_with_captured_stats reads, so /api/ps reports a size for a
vision model that omits the projector -- which is exactly the number
somebody uses to decide whether the model fits in VRAM.
"""

import io
import sys
import contextlib

import pytest

from tllama.helpers.llama_stats import load_llama_with_captured_stats


class FakeHandler:
    """Records whether, and when, the mtmd context was asked for."""

    def __init__(self, log_line="Vulkan0 compute buffer size = 248.10 MiB"):
        self.initialised_with = None
        self._log_line = log_line

    def _init_mtmd_context(self, llama_model):
        self.initialised_with = llama_model
        print(self._log_line, file=sys.stderr)


class FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat_handler = kwargs.get("chat_handler")
        print("Vulkan0 model buffer size = 1000.00 MiB", file=sys.stderr)


class TestTheCaptureWindow:
    def test_after_load_runs_and_receives_the_model(self):
        seen = []

        llm, _stats, _log = load_llama_with_captured_stats(
            FakeLlama, after_load=seen.append
        )

        assert seen == [llm]

    def test_what_after_load_prints_is_part_of_the_reading(self):
        def emit(llm):
            print("Vulkan0 compute buffer size = 248.10 MiB", file=sys.stderr)

        _llm, stats, log = load_llama_with_captured_stats(FakeLlama, after_load=emit)

        assert "248.10" in log
        assert stats["gpu_compute_mib"] == pytest.approx(248.10)

    def test_no_after_load_still_loads(self):
        llm, stats, log = load_llama_with_captured_stats(FakeLlama)

        assert isinstance(llm, FakeLlama)
        assert stats["gpu_model_mib"] == pytest.approx(1000.00)

    def test_after_load_output_does_not_escape_to_the_real_stderr(self):
        def emit(llm):
            print("must not leak", file=sys.stderr)

        outer = io.StringIO()
        with contextlib.redirect_stderr(outer):
            _llm, _stats, log = load_llama_with_captured_stats(
                FakeLlama, after_load=emit
            )

        assert "must not leak" in log
        assert outer.getvalue() == ""


class TestTheProjectorComesUpAtLoad:
    def test_a_model_with_a_projector_has_it_initialised(self, manager, monkeypatch):
        from tllama import backend

        handler = FakeHandler()
        monkeypatch.setattr(backend, "Llama", FakeLlama)
        monkeypatch.setattr(
            manager,
            "_build_llama_load_kwargs",
            lambda *a, **k: {"chat_handler": handler},
        )

        llm, _stats, _log = manager._load_model_sync("Local/model.gguf", 2048)

        assert handler.initialised_with is llm

    def test_the_projectors_buffers_land_in_the_reading(self, manager, monkeypatch):
        from tllama import backend

        handler = FakeHandler()
        monkeypatch.setattr(backend, "Llama", FakeLlama)
        monkeypatch.setattr(
            manager,
            "_build_llama_load_kwargs",
            lambda *a, **k: {"chat_handler": handler},
        )

        _llm, stats, _log = manager._load_model_sync("Local/model.gguf", 2048)

        # 248.10 from the projector on top of the model's own buffers,
        # which is the whole point: /api/ps has to see both.
        assert stats["gpu_compute_mib"] == pytest.approx(248.10)

    def test_a_model_without_a_projector_loads_unchanged(self, manager, monkeypatch):
        from tllama import backend

        monkeypatch.setattr(backend, "Llama", FakeLlama)
        monkeypatch.setattr(manager, "_build_llama_load_kwargs", lambda *a, **k: {})

        llm, stats, _log = manager._load_model_sync("Local/model.gguf", 2048)

        assert llm.chat_handler is None
        assert stats["gpu_model_mib"] == pytest.approx(1000.00)

    def test_a_plain_text_chat_handler_is_not_mistaken_for_a_projector(
        self, manager, monkeypatch
    ):
        from tllama import backend

        # A [template] override attaches an opaque callable with no mtmd
        # context to bring up; asking it for one would be an AttributeError.
        def text_handler(*args, **kwargs):
            raise AssertionError("must not be called during loading")

        monkeypatch.setattr(backend, "Llama", FakeLlama)
        monkeypatch.setattr(
            manager,
            "_build_llama_load_kwargs",
            lambda *a, **k: {"chat_handler": text_handler},
        )

        llm, _stats, _log = manager._load_model_sync("Local/model.gguf", 2048)

        assert llm.chat_handler is text_handler
