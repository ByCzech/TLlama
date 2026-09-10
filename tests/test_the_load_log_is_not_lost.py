"""Where llama.cpp's account of a load ends up.

The load runs under redirect_stderr so its numbers can be parsed, which
means none of it reaches the server's own stderr. Nothing read it back
out: the text went into a local, the local went into a caller's variable,
and the variable had no reader anywhere in the tree. Diagnosing anything
about a load meant editing the source and reloading.

On the failing path it was worse than unread. getvalue() was called after
the with block, so a throwing load discarded the text entirely, and what
a client got was ValueError("Failed to create llama_context") -- the same
constant string for a card out of memory and for a quantised V cache
asked for without flash attention. The reason existed and was destroyed.
"""

import logging

import pytest

from tllama.helpers.llama_stats import load_llama_with_captured_stats


CUDA_OOM = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4657.81 MiB on "
    "device 0: cudaMalloc failed: out of memory\n"
)

HEALTHY = "load_tensors: offloaded 41/41 layers to GPU\n"


def talker(text, exception=None):
    """A stand-in for Llama() that writes to stderr the way llama.cpp does."""
    def build(**kwargs):
        import sys

        sys.stderr.write(text)
        if exception is not None:
            raise exception
        return object()

    return build


class TestAFailedLoad:
    def test_what_llama_cpp_said_is_logged(self, caplog):
        with caplog.at_level(logging.ERROR, logger="tllama.helpers.llama_stats"):
            with pytest.raises(ValueError):
                load_llama_with_captured_stats(
                    talker(CUDA_OOM, ValueError("Failed to create llama_context"))
                )

        assert "cudaMalloc failed: out of memory" in caplog.text

    def test_the_exception_is_unchanged(self, caplog):
        """Callers turn this into an HTTP status; the log is beside it."""
        with pytest.raises(ValueError) as raised:
            load_llama_with_captured_stats(
                talker(CUDA_OOM, ValueError("Failed to create llama_context"))
            )

        assert str(raised.value) == "Failed to create llama_context"

    def test_it_is_loud_enough_to_be_seen_without_turning_anything_on(self, caplog):
        with caplog.at_level(logging.ERROR, logger="tllama.helpers.llama_stats"):
            with pytest.raises(ValueError):
                load_llama_with_captured_stats(talker(CUDA_OOM, ValueError("boom")))

        assert [record.levelno for record in caplog.records] == [logging.ERROR]

    def test_an_empty_capture_says_so_rather_than_logging_nothing(self, caplog):
        """Silence and "the load said nothing" are different findings."""
        with caplog.at_level(logging.ERROR, logger="tllama.helpers.llama_stats"):
            with pytest.raises(ValueError):
                load_llama_with_captured_stats(talker("", ValueError("boom")))

        assert caplog.records


class TestALoadThatWorked:
    def test_its_output_is_not_thrown_away(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="tllama.helpers.llama_stats"):
            load_llama_with_captured_stats(talker(HEALTHY))

        assert "offloaded 41/41 layers to GPU" in caplog.text

    def test_it_stays_out_of_a_default_log(self, caplog):
        """Two thousand lines per load, one setting away when wanted."""
        with caplog.at_level(logging.INFO, logger="tllama.helpers.llama_stats"):
            load_llama_with_captured_stats(talker(HEALTHY))

        assert caplog.records == []

    def test_the_numbers_still_come_back(self, caplog):
        _, stats, log_text = load_llama_with_captured_stats(talker(HEALTHY))

        assert stats["offloaded_layers"] == 41
        assert log_text == HEALTHY
