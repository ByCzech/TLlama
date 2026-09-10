"""Getting the projector to report itself where anyone can hear it.

mtmd keeps a logger of its own, separate from llama.cpp's and ggml's, and
clip_log_callback_default writes to the C-level stderr with fputs. That is
below contextlib.redirect_stderr, which swaps the Python object, so the
capture a load runs under never saw a line of it.

Measured on a vision load of Qwen3.6-35B-A3B: 613.14 MiB of projector
weights and a 248.10 MiB compute buffer went to the console while
/api/ps reported neither. The parser could always have read them --
PROJECTOR_BACKEND and PROJECTOR_SIZE match those lines unchanged -- it
was never handed the text.

mtmd_log_set() is the fix and llama-cpp-python already binds it. The
same reading also explains a second thing: clip's default callback starts
with (void) level, so it printed its whole 334-line tensor inventory at
every load with nothing able to filter it.
"""

import logging

import pytest

from tllama.backend import _initialise_projector
from tllama.helpers.llama_stats import parse_llama_verbose_load_log


# Cut from a real Vulkan load. Every line here reached the console and
# none of it reached the capture.
PROJECTOR_OUTPUT = """clip_ctx: CLIP using Vulkan0 backend
load_hparams: projector:          qwen3vl_merger
load_hparams: model size:         613.14 MiB
load_hparams: metadata size:      0.12 MiB
get_dummy_batch: warmup with image size = 1472 x 1472
reserve_compute_meta:    Vulkan0 compute buffer size =   248.10 MiB
reserve_compute_meta:        CPU compute buffer size =    24.93 MiB
reserve_compute_meta: graph splits = 1, nodes = 823
warmup: flash attention is enabled
"""


class Handler:
    def __init__(self):
        self.initialised_with = None

    def _init_mtmd_context(self, llm):
        self.initialised_with = llm


class Model:
    def __init__(self, handler):
        self.chat_handler = handler


class TestTheProjectorIsToldWhereToWrite:
    def test_mtmd_is_redirected_before_it_says_anything(self, monkeypatch):
        """Everything worth reading is printed inside _init_mtmd_context."""
        order = []

        from llama_cpp import mtmd_cpp

        monkeypatch.setattr(
            mtmd_cpp, "mtmd_log_set",
            lambda cb, ud: order.append("redirected"),
        )

        handler = Handler()
        original = handler._init_mtmd_context

        def watched(llm):
            order.append("initialised")
            return original(llm)

        handler._init_mtmd_context = watched

        _initialise_projector(Model(handler))

        assert order == ["redirected", "initialised"]

    def test_it_is_handed_the_callback_the_capture_can_see(self, monkeypatch):
        """llama-cpp-python's own, so the two streams arrive together.

        Its callback prints to sys.stderr, which is what redirect_stderr
        replaces. Any other destination would leave the projector's
        output where it already was.
        """
        given = []

        from llama_cpp import mtmd_cpp
        from llama_cpp._logger import llama_log_callback

        monkeypatch.setattr(
            mtmd_cpp, "mtmd_log_set", lambda cb, ud: given.append(cb)
        )

        _initialise_projector(Model(Handler()))

        assert given == [llama_log_callback]

    def test_a_model_without_a_projector_is_left_alone(self, monkeypatch):
        called = []

        from llama_cpp import mtmd_cpp

        monkeypatch.setattr(
            mtmd_cpp, "mtmd_log_set", lambda cb, ud: called.append(cb)
        )

        class Plain:
            chat_handler = None

        _initialise_projector(Plain())

        assert called == []


class TestWhenTheHookCannotBeReached:
    def test_the_load_still_happens(self, monkeypatch):
        """A gap in a memory report is not a reason to refuse a model."""
        from llama_cpp import mtmd_cpp

        monkeypatch.delattr(mtmd_cpp, "mtmd_log_set")

        handler = Handler()
        model = Model(handler)

        _initialise_projector(model)

        assert handler.initialised_with is model

    def test_it_says_what_will_be_missing(self, monkeypatch, caplog):
        from llama_cpp import mtmd_cpp

        monkeypatch.delattr(mtmd_cpp, "mtmd_log_set")

        with caplog.at_level(logging.ERROR, logger="tllama.backend"):
            _initialise_projector(Model(Handler()))

        assert "/api/ps" in caplog.text

    def test_a_failing_call_is_not_swallowed(self, monkeypatch, caplog):
        from llama_cpp import mtmd_cpp

        def refuse(cb, ud):
            raise OSError("undefined symbol: mtmd_log_set")

        monkeypatch.setattr(mtmd_cpp, "mtmd_log_set", refuse)

        with caplog.at_level(logging.ERROR, logger="tllama.backend"):
            _initialise_projector(Model(Handler()))

        assert "undefined symbol" in caplog.text


class TestWhatTheParserDoesWithIt:
    """The parser was always ready; only the text was missing."""

    def test_the_weights_land_in_the_projector_bucket(self):
        stats = parse_llama_verbose_load_log(PROJECTOR_OUTPUT)

        assert stats["gpu_projector_mib"] == 613.14

    def test_the_compute_buffer_lands_in_the_compute_bucket(self):
        stats = parse_llama_verbose_load_log(PROJECTOR_OUTPUT)

        assert stats["gpu_compute_mib"] == 248.10
        assert stats["cpu_compute_mib"] == 24.93

    def test_the_metadata_line_is_not_mistaken_for_the_weights(self):
        """Two 'size:' lines in a row, only one of them the model."""
        stats = parse_llama_verbose_load_log(PROJECTOR_OUTPUT)

        assert stats["gpu_projector_mib"] != pytest.approx(0.12)
