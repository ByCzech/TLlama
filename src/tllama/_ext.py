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

import ctypes
from typing import Any, Optional


class _GgmlBackendDevCaps(ctypes.Structure):
    _fields_ = [
        ("async_", ctypes.c_bool),
        ("host_buffer", ctypes.c_bool),
        ("buffer_from_host_ptr", ctypes.c_bool),
        ("events", ctypes.c_bool),
        ("mmap_support", ctypes.c_bool),
    ]


class _GgmlBackendDevProps(ctypes.Structure):
    """ggml/include/ggml-backend.h, struct ggml_backend_dev_props.

    Measured at 56 bytes on all three machines this was checked against,
    which is what the header comes to.
    """

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("memory_free", ctypes.c_size_t),
        ("memory_total", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("device_id", ctypes.c_char_p),
        ("caps", _GgmlBackendDevCaps),
    ]


# ggml_backend_dev_type, in the header's order. TLlama's own development
# notes give a different one and are wrong.
_DEV_TYPE_CPU = 0
_DEV_TYPE_ACCEL = 3


def device_memory_totals() -> Optional[dict]:
    """How much memory each compute device says it has, by device name.

    Only the total. memory_free is deliberately not read: it is computed
    as total minus used, and on the one case this exists to catch --
    a claim larger than the device -- it had wrapped around a size_t and
    came back as 2**64 bytes. The total stayed right.

    Names are the ones ggml prints in the load log, so a caller can match
    a buffer line to a device without going through PCI ids. That matters:
    one card enumerated under two backends reports two different totals,
    16269 MiB against 16384 MiB for the same Tesla, so the pairing has to
    be by the name the buffer was announced under.

    CPU and ACCEL devices are left out. A CPU device reports the size of
    system RAM as both free and total, and BLAS reports zero for both;
    neither is a capacity anything is measured against here.

    Returns None when the devices cannot be read at all. Note that they
    cannot before a model has been loaded: llama_backend_init() is what
    fills the registry, and until then the device count is zero.
    """
    try:
        from llama_cpp import llama_cpp as binding
    except ImportError:
        return None

    lib = getattr(binding, "_lib", None)
    if lib is None:
        return None

    try:
        count_fn = lib.ggml_backend_dev_count
        get_fn = lib.ggml_backend_dev_get
        props_fn = lib.ggml_backend_dev_get_props
    except AttributeError:
        return None

    try:
        count_fn.argtypes = []
        count_fn.restype = ctypes.c_size_t
        get_fn.argtypes = [ctypes.c_size_t]
        get_fn.restype = ctypes.c_void_p
        props_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GgmlBackendDevProps)]
        props_fn.restype = None

        totals = {}
        for index in range(count_fn()):
            props = _GgmlBackendDevProps()
            props_fn(get_fn(index), ctypes.byref(props))

            if props.type in (_DEV_TYPE_CPU, _DEV_TYPE_ACCEL):
                continue
            if not props.name or not props.memory_total:
                continue

            totals[props.name.decode()] = props.memory_total
    except Exception:
        return None

    return totals or None


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


def eval_durations(llm: Any) -> Optional[tuple[float, float]]:
    """Milliseconds a context spent evaluating, as (prompt, generation).

    The same struct eval_counters() reads, two more of its fields. These
    are what llama.cpp prints as "prompt eval time" and "eval time", and
    like the counts they run from the last reset rather than over the
    life of the context.
    """
    data = _perf_context_data(llm)
    if data is None:
        return None

    try:
        return float(data.t_p_eval_ms), float(data.t_eval_ms)
    except (AttributeError, TypeError, ValueError):
        return None


def timed_for_request(was_reset: bool, llm: Any) -> tuple[Optional[int], Optional[int]]:
    """Nanoseconds spent on one request, as (prompt_eval, eval).

    Nanoseconds because that is the unit the Ollama reply uses for every
    other duration in it. A pair of None unless the counters were zeroed
    beforehand, for the same reason counted_for_request() gives: without
    that the reading covers an unknown number of earlier requests, and a
    figure a client divides into a tokens-per-second rate is worse wrong
    than absent.

    t_load_ms is deliberately not read here. llama_context::perf_reset()
    zeroes the two evaluation timers and leaves that one alone, so it
    holds the time the model took to load and would report it again on
    every request made against that context for as long as it lives.
    """
    if not was_reset:
        return None, None

    durations = eval_durations(llm)
    if durations is None:
        return None, None

    prompt_ms, eval_ms = durations
    return int(prompt_ms * 1_000_000), int(eval_ms * 1_000_000)


class _ggml_backend_dev_caps(ctypes.Structure):
    _fields_ = [
        ("async_", ctypes.c_bool),
        ("host_buffer", ctypes.c_bool),
        ("buffer_from_host_ptr", ctypes.c_bool),
        ("events", ctypes.c_bool),
        ("mmap_support", ctypes.c_bool),
    ]


class _ggml_backend_dev_props(ctypes.Structure):
    """Field for field from ggml/include/ggml-backend.h.

    Verified against the vendored header and by size: 56 bytes, with
    device_id at offset 40. Getting this wrong reads whatever happens to
    be at those offsets, so it is checked rather than assumed -- a wrong
    layout would be a plausible-looking number, not a crash.
    """

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("memory_free", ctypes.c_size_t),
        ("memory_total", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("device_id", ctypes.c_char_p),
        ("caps", _ggml_backend_dev_caps),
    ]


# CPU, GPU, IGPU, ACCEL, META, in that order, from the same header. An
# integrated GPU is its own kind and is neither of the two it sits
# between.
_DEVICE_TYPE_GPU = 1
_DEVICE_TYPE_IGPU = 2


def device_memory_totals() -> Optional[dict]:
    """How much memory each compute device says it has, in bytes.

    Only meaningful once a model has been loaded: the backend registry is
    empty until llama_backend_init() runs, and before that
    ggml_backend_dev_count() answers zero. Measured, not assumed.

    memory_free is deliberately not read. On an overcommitted Vulkan
    device it is computed as total minus used and wraps: one measurement
    reported 17592186029327.65 MiB free on a 16 GiB card. memory_total
    stayed correct throughout.

    CPU and accelerator devices are left out. Their totals are either the
    machine's RAM reported twice or zero, and neither is a capacity
    anything can be measured against.
    """
    try:
        from llama_cpp import llama_cpp as binding
    except ImportError:
        return None

    library = getattr(binding, "_lib", None)
    if library is None:
        return None

    try:
        count_fn = library.ggml_backend_dev_count
        get_fn = library.ggml_backend_dev_get
        props_fn = library.ggml_backend_dev_get_props
    except AttributeError:
        return None

    try:
        count_fn.argtypes = []
        count_fn.restype = ctypes.c_size_t
        get_fn.argtypes = [ctypes.c_size_t]
        get_fn.restype = ctypes.c_void_p
        props_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ggml_backend_dev_props),
        ]
        props_fn.restype = None

        totals = {}
        for index in range(count_fn()):
            props = _ggml_backend_dev_props()
            props_fn(get_fn(index), ctypes.byref(props))
            if props.type not in (_DEVICE_TYPE_GPU, _DEVICE_TYPE_IGPU):
                continue
            if not props.name or props.memory_total <= 0:
                continue
            totals[props.name.decode()] = props.memory_total
        return totals
    except Exception:
        return None
