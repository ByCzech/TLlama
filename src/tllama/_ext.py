"""Direct access to llama.cpp symbols that llama-cpp-python does not expose.

The binding layer lags upstream, and a fair amount of what llama.cpp offers
is reachable with nothing more than ctypes over the shared library that is
already loaded. This module is where those prototypes accumulate, one at a
time, as something turns out to be missing.

Two rules hold for everything here. It is best effort: a symbol that is not
present, a binding that moved, or a call that fails returns None, and the
caller carries on without it. And it never becomes load bearing: anything
that reads from here has to work, if less precisely, when it returns None.
"""

from typing import Any, Optional


def _perf_context_data(llm: Any) -> Optional[Any]:
    """Read llama.cpp's own counters for a model's context.

    llama_perf_context is a plain C entry point returning a struct by value,
    so ctypes reaches it with no build step. Reaching the context itself goes
    through llama-cpp-python's internals, which is the fragile part and the
    reason for the broad except.
    """
    try:
        from llama_cpp import llama_cpp as binding
    except ImportError:
        return None

    context = getattr(getattr(llm, "_ctx", None), "ctx", None)
    if context is None:
        return None

    perf_context = getattr(binding, "llama_perf_context", None)
    if perf_context is None:
        return None

    try:
        return perf_context(context)
    except Exception:
        return None


def reset_eval_counters(llm: Any) -> bool:
    """Zero a context's evaluation counters ahead of a request.

    llama-cpp-python zeroes them itself, but only when the model was built
    with verbose=True, because it does so as a side effect of printing
    timings. Leaving it at that would make the token counts a server reports
    depend on whether logging happens to be on, so this does it explicitly.

    Returns whether the reset took place. A reading only describes one
    request if it did.
    """
    try:
        from llama_cpp import llama_cpp as binding
    except ImportError:
        return False

    context = getattr(getattr(llm, "_ctx", None), "ctx", None)
    reset = getattr(binding, "llama_perf_context_reset", None)

    if context is None or reset is None:
        return False

    try:
        reset(context)
    except Exception:
        return False

    return True


def eval_counters(llm: Any) -> Optional[tuple[int, int]]:
    """Tokens a context has evaluated, as (prompt, generated).

    These are the figures llama.cpp prints as "prompt eval time / N tokens"
    and "eval time / N runs". They count from the last reset, not over the
    life of the context, so one request is a single reading taken after it
    rather than a difference between two.
    """
    data = _perf_context_data(llm)
    if data is None:
        return None

    try:
        return int(data.n_p_eval), int(data.n_eval)
    except (AttributeError, TypeError, ValueError):
        return None


def counted_for_request(was_reset: bool, llm: Any) -> tuple[Optional[int], Optional[int]]:
    """Tokens evaluated for one request, as (prompt_eval_count, eval_count).

    Yields a pair of None unless the counters were zeroed beforehand. Without
    that the reading spans an unknown number of earlier requests, and a wrong
    count is worse than the absent one the field carried before.
    """
    if not was_reset:
        return None, None

    counters = eval_counters(llm)
    if counters is None:
        return None, None

    return counters
