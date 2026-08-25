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


def eval_counters(llm: Any) -> Optional[tuple[int, int]]:
    """Tokens this context has evaluated so far, as (prompt, generated).

    These are the numbers llama.cpp prints as "prompt eval time / N tokens"
    and "eval time / N runs". They accumulate over the life of the context,
    so a single request is the difference between two readings rather than
    either reading on its own.

    Taking that difference is only meaningful because generation on a model
    is serialised: with two requests interleaved on one context the counters
    would mix.
    """
    data = _perf_context_data(llm)
    if data is None:
        return None

    try:
        return int(data.n_p_eval), int(data.n_eval)
    except (AttributeError, TypeError, ValueError):
        return None


def counted_since(before: Optional[tuple[int, int]], llm: Any) -> tuple[Optional[int], Optional[int]]:
    """Tokens evaluated since a reading, as (prompt_eval_count, eval_count).

    Returns a pair of None when either reading is unavailable, which is what
    the field held before these counters were used at all.
    """
    after = eval_counters(llm)
    if before is None or after is None:
        return None, None

    prompt_tokens = after[0] - before[0]
    generated_tokens = after[1] - before[1]

    # A context reset between the two readings makes the difference
    # meaningless rather than merely imprecise.
    if prompt_tokens < 0 or generated_tokens < 0:
        return None, None

    return prompt_tokens, generated_tokens
