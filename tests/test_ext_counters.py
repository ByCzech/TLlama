"""Reading llama.cpp's own counters through ctypes.

Everything in _ext is best effort by design: a symbol that moved, a binding
that changed, or a call that fails has to degrade to None rather than break
a request. These tests pin that contract, because it is the thing that makes
reaching past llama-cpp-python acceptable at all.
"""

import ctypes
import sys
import types

import pytest

from tllama import _ext


class FakePerfData(ctypes.Structure):
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
        ("n_reused", ctypes.c_int32),
    ]


class FakeLlama:
    """Shaped like llama_cpp.Llama as far as _ext reaches into it."""

    def __init__(self, prompt_tokens=0, generated_tokens=0):
        self._ctx = types.SimpleNamespace(ctx=object())
        self.counts = [prompt_tokens, generated_tokens]

    def advance(self, prompt_tokens, generated_tokens):
        self.counts[0] += prompt_tokens
        self.counts[1] += generated_tokens


@pytest.fixture
def binding(monkeypatch):
    """Stand in for llama_cpp.llama_cpp with a working llama_perf_context."""
    holder = {}

    def llama_perf_context(ctx):
        model = holder["model"]
        return FakePerfData(n_p_eval=model.counts[0], n_eval=model.counts[1])

    module = types.ModuleType("llama_cpp.llama_cpp")
    module.llama_perf_context = llama_perf_context
    package = types.ModuleType("llama_cpp")
    package.llama_cpp = module
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", module)
    return holder


def test_counters_are_read_from_the_context(binding):
    model = FakeLlama(prompt_tokens=26, generated_tokens=110)
    binding["model"] = model

    assert _ext.eval_counters(model) == (26, 110)


def test_a_request_is_the_difference_between_two_readings(binding):
    """The counters accumulate over the life of the context, not per request."""
    model = FakeLlama(prompt_tokens=26, generated_tokens=110)
    binding["model"] = model
    before = _ext.eval_counters(model)

    model.advance(prompt_tokens=12, generated_tokens=48)

    assert _ext.counted_since(before, model) == (12, 48)


def test_a_context_reset_between_readings_yields_nothing(binding):
    """A negative difference is meaningless, not merely imprecise."""
    model = FakeLlama(prompt_tokens=26, generated_tokens=110)
    binding["model"] = model
    before = _ext.eval_counters(model)

    model.counts = [0, 0]

    assert _ext.counted_since(before, model) == (None, None)


def test_a_missing_symbol_degrades_to_nothing(binding, monkeypatch):
    """The binding lagging upstream must not break a request."""
    monkeypatch.delattr(sys.modules["llama_cpp.llama_cpp"], "llama_perf_context")
    binding["model"] = model = FakeLlama()

    assert _ext.eval_counters(model) is None
    assert _ext.counted_since((0, 0), model) == (None, None)


def test_a_moved_internal_degrades_to_nothing(binding):
    """_ctx.ctx is llama-cpp-python's private shape and may not survive."""
    binding["model"] = model = FakeLlama()
    del model._ctx

    assert _ext.eval_counters(model) is None


def test_a_failing_call_degrades_to_nothing(binding):
    def explode(ctx):
        raise OSError("symbol not found")

    sys.modules["llama_cpp.llama_cpp"].llama_perf_context = explode
    binding["model"] = model = FakeLlama()

    assert _ext.eval_counters(model) is None


def test_an_unavailable_reading_leaves_the_caller_with_nothing_to_report(binding):
    binding["model"] = model = FakeLlama()

    assert _ext.counted_since(None, model) == (None, None)
