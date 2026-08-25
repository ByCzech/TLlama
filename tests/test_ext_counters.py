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

    def llama_perf_context_reset(ctx):
        holder["model"].counts = [0, 0]

    module = types.ModuleType("llama_cpp.llama_cpp")
    module.llama_perf_context = llama_perf_context
    module.llama_perf_context_reset = llama_perf_context_reset
    package = types.ModuleType("llama_cpp")
    package.llama_cpp = module
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", module)
    return holder


def test_counters_are_read_from_the_context(binding):
    model = FakeLlama(prompt_tokens=26, generated_tokens=110)
    binding["model"] = model

    assert _ext.eval_counters(model) == (26, 110)


def test_a_reading_describes_the_request_since_the_reset(binding):
    """The counters restart at each reset, so one reading is one request."""
    binding["model"] = model = FakeLlama(prompt_tokens=99, generated_tokens=999)

    was_reset = _ext.reset_eval_counters(model)
    model.advance(prompt_tokens=24, generated_tokens=636)

    assert was_reset is True
    assert _ext.counted_for_request(was_reset, model) == (24, 636)


def test_leftovers_from_an_earlier_request_do_not_leak_in(binding):
    """The defect this replaced: llama-cpp-python resets these itself, but
    only when verbose is on, so a reading taken without one spans requests."""
    binding["model"] = model = FakeLlama(prompt_tokens=24, generated_tokens=636)

    _ext.reset_eval_counters(model)
    model.advance(prompt_tokens=24, generated_tokens=376)

    assert _ext.counted_for_request(True, model) == (24, 376)


def test_a_reading_without_a_reset_reports_nothing(binding):
    """A count covering an unknown span is worse than an absent one."""
    binding["model"] = model = FakeLlama(prompt_tokens=24, generated_tokens=636)

    assert _ext.counted_for_request(False, model) == (None, None)


def test_a_reset_that_cannot_happen_says_so(binding, monkeypatch):
    monkeypatch.delattr(sys.modules["llama_cpp.llama_cpp"], "llama_perf_context_reset")
    binding["model"] = model = FakeLlama()

    assert _ext.reset_eval_counters(model) is False


def test_a_missing_symbol_degrades_to_nothing(binding, monkeypatch):
    """The binding lagging upstream must not break a request."""
    monkeypatch.delattr(sys.modules["llama_cpp.llama_cpp"], "llama_perf_context")
    binding["model"] = model = FakeLlama()

    assert _ext.eval_counters(model) is None
    assert _ext.counted_for_request(True, model) == (None, None)


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
    del model._ctx

    assert _ext.counted_for_request(True, model) == (None, None)
