import io
import re
import contextlib

# llama.cpp prints whatever the buffer type calls itself, and that set is
# open: ggml/src has a dozen backends and each names its own buffers.
# A hand-kept list silently drops everything it has not heard of, which is
# how "CPU_Mapped model buffer size" -- the mmapped weights that stay in
# host RAM -- went uncounted. Match the shape of a buffer type name
# instead; the "<kind> buffer size = N MiB" suffix is specific enough to
# carry the match on its own.
BACKEND_PREFIX = r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*"

LOAD_PATTERNS = {
    "offloaded_layers": re.compile(
        r"offloaded\s+(\d+)/(\d+)\s+layers to GPU",
        re.IGNORECASE,
    ),
    "n_ctx": re.compile(
        r"\bn_ctx\s*=\s*(\d+)",
        re.IGNORECASE,
    ),
    "model_buffer": re.compile(
        rf"({BACKEND_PREFIX})\s+model buffer size\s*=\s*([\d.]+)\s*MiB",
        re.IGNORECASE,
    ),
    "kv_buffer": re.compile(
        rf"({BACKEND_PREFIX})\s+KV buffer size\s*=\s*([\d.]+)\s*MiB",
        re.IGNORECASE,
    ),
    "compute_buffer": re.compile(
        rf"({BACKEND_PREFIX})\s+compute buffer size\s*=\s*([\d.]+)\s*MiB",
        re.IGNORECASE,
    ),
    "output_buffer": re.compile(
        rf"({BACKEND_PREFIX})\s+output buffer size\s*=\s*([\d.]+)\s*MiB",
        re.IGNORECASE,
    ),
    "rs_buffer": re.compile(
        rf"({BACKEND_PREFIX})\s+RS buffer size\s*=\s*([\d.]+)\s*MiB",
        re.IGNORECASE,
    ),
}

# A projector announces its weights without naming a backend:
#
#   clip_ctx: CLIP using Vulkan1 backend
#   ...
#   load_hparams: model size:         613.14 MiB
#
# The size line alone cannot be matched safely -- "load_hparams:" prefixes a
# dozen other lines, and neither line carries the backend the way every
# buffer line does. The two have to be paired: the backend comes from
# clip_ctx, the number from the first model size after it.
PROJECTOR_BACKEND = re.compile(
    rf"clip_ctx:\s*CLIP using\s+({BACKEND_PREFIX})\s+backend",
    re.IGNORECASE,
)

PROJECTOR_SIZE = re.compile(
    r"load_hparams:\s*model size:\s*([\d.]+)\s*MiB",
    re.IGNORECASE,
)


# Which device each layer was put on, said twice: once for the weights
# and once for the KV cache, which llama.cpp places per layer according
# to where that layer's weights went (llama-kv-cache.cpp).
#
# This is the only thing in the log that separates "a few hundred MiB of
# weights that were never going to a device" from "half the model is on
# the CPU". Both are announced as a CPU_Mapped model buffer and the
# buffer names cannot tell them apart -- measured, 397.85 MiB against
# 6782.69 MiB under the same name.
#
# A KV line reads either "dev = <name>" or "filtered", the latter for a
# layer that has no KV cache at all; only the first form matches here.
LAYER_DEVICE = re.compile(
    rf"^load_tensors: layer\s+\d+ assigned to device ({BACKEND_PREFIX}),",
    re.MULTILINE,
)

KV_LAYER_DEVICE = re.compile(
    rf"^llama_kv_cache: layer\s+\d+: dev = ({BACKEND_PREFIX})\s*$",
    re.MULTILINE,
)

# ggml reports an allocation it could not make and then, in some paths,
# carries on. Both lines come from ggml/src/ggml-alloc.c and are prefixed
# with __func__, so the function name is what distinguishes them from any
# other line; the buffer type name is filled in by
# ggml_backend_buft_name(), which is why one pattern covers every backend
# rather than only the one this was first seen on.
#
#   ggml_gallocr_reserve_n_impl  reserving a compute graph
#   alloc_tensor_range           allocating weights
FAILED_ALLOCATION = re.compile(
    rf"^(ggml_gallocr_reserve_n_impl|alloc_tensor_range): "
    rf"failed to allocate ({BACKEND_PREFIX}) buffer of size (\d+)",
    re.MULTILINE,
)


def parse_llama_verbose_load_log(log_text: str) -> dict:
    result = {
        "raw_log": log_text,
        "n_ctx": None,
        "offloaded_layers": 0,
        "total_layers": 0,
        "buffers": {},
        "cpu_mib": 0.0,
        "gpu_mib": 0.0,
        "cpu_model_mib": 0.0,
        "gpu_model_mib": 0.0,
        "cpu_kv_mib": 0.0,
        "gpu_kv_mib": 0.0,
        "cpu_compute_mib": 0.0,
        "gpu_compute_mib": 0.0,
        "cpu_output_mib": 0.0,
        "gpu_output_mib": 0.0,
        "cpu_rs_mib": 0.0,
        "gpu_rs_mib": 0.0,
        "cpu_projector_mib": 0.0,
        "gpu_projector_mib": 0.0,

        # Model weights left out of the buckets below because they are
        # input tensors rather than anything a processor computes. Named
        # rather than dropped so the difference between the buckets and
        # the log is on the record instead of being a silent subtraction.
        "input_model_mib": 0.0,
        "gpu_host_model_mib": 0.0,
        "gpu_host_kv_mib": 0.0,
        "gpu_host_compute_mib": 0.0,
        "gpu_host_output_mib": 0.0,
        "gpu_host_rs_mib": 0.0,
        "gpu_host_mib": 0.0,
        "processor": "100% CPU",
        "failed_allocations": [],
        "layer_devices_seen": False,
        "cpu_layers": 0,
        "gpu_layers": 0,
        "cpu_kv_layers": 0,
        "gpu_kv_layers": 0,
    }

    result["failed_allocations"] = [
        {"func": func, "backend": backend, "bytes": int(size)}
        for func, backend, size in FAILED_ALLOCATION.findall(log_text)
    ]

    m = LOAD_PATTERNS["offloaded_layers"].search(log_text)
    if m:
        result["offloaded_layers"] = int(m.group(1))
        result["total_layers"] = int(m.group(2))

    m = LOAD_PATTERNS["n_ctx"].search(log_text)
    if m:
        result["n_ctx"] = int(m.group(1))

    def is_true_cpu_backend(backend: str) -> bool:
        upper = backend.upper()
        return upper == "CPU" or upper.startswith("CPU_")

    def is_host_helper_backend(backend: str) -> bool:
        return backend.upper().endswith("_HOST")

    def is_device_gpu_backend(backend: str) -> bool:
        return not is_true_cpu_backend(backend) and not is_host_helper_backend(backend)

    def add_buffer(kind: str, backend: str, mib: float):
        result["buffers"].setdefault(backend, {})
        result["buffers"][backend][kind] = result["buffers"][backend].get(kind, 0.0) + mib

        if is_true_cpu_backend(backend):
            bucket = "cpu"
        elif is_host_helper_backend(backend):
            bucket = "gpu_host"
        else:
            bucket = "gpu"

        result[f"{bucket}_{kind}_mib"] += mib

    for kind, pattern in (
        ("model", LOAD_PATTERNS["model_buffer"]),
        ("kv", LOAD_PATTERNS["kv_buffer"]),
        ("compute", LOAD_PATTERNS["compute_buffer"]),
        ("output", LOAD_PATTERNS["output_buffer"]),
        ("rs", LOAD_PATTERNS["rs_buffer"]),
    ):
        for backend, mib in pattern.findall(log_text):
            add_buffer(kind, backend, float(mib))

    # A projector's weights are the one allocation llama.cpp reports without
    # a backend prefix, so they used to be dropped on the floor. Pair each
    # clip_ctx line with the next model size line; anything unpaired is left
    # alone rather than guessed at.
    for match in PROJECTOR_BACKEND.finditer(log_text):
        size = PROJECTOR_SIZE.search(log_text, match.end())
        if size:
            add_buffer("projector", match.group(1), float(size.group(1)))

    # What ps answers is which processor does the heavy work, because a
    # layer or a KV cache computed on the CPU is the thing that costs
    # tokens per second. Input tensors are not that. llama.cpp keeps
    # token_embd.weight off the device unconditionally -- its buffer type
    # list does not contain the device at all -- and uses it for a
    # GGML_OP_GET_ROWS, one row copied per token. It is resident whatever
    # happens and it never moves work anywhere.
    #
    # Counting it made ps say a fully offloaded model was 2% on the CPU,
    # and made the figure depend on use_mmap: the same tensor, the same
    # bytes in the same place, lands in a CPU_Mapped buffer with mmap on
    # and a <device>_Host one with it off, which changed size_vram by its
    # whole size for a setting that has nothing to do with where the work
    # happens.
    #
    # With every layer on the device, nothing in a non-device model buffer
    # can be a layer, so all of it is input tensors and all of it goes.
    # That is an exclusion, not an estimate. Short of that the log does
    # not separate them -- one line covers every tensor the same rule
    # moved, 385 of them in the measured 20-of-41 case -- so the buffer is
    # left whole. It is the case where the CPU really is computing, ps
    # says so, and the input tensors are a small part of a number that is
    # already telling the truth.
    #
    # buffers is left alone either way: it is the record of what the log
    # said, and _device_residency() compares it against what the devices
    # can hold.
    fully_offloaded = (
        result["total_layers"] > 0
        and result["offloaded_layers"] == result["total_layers"]
    )
    if fully_offloaded:
        result["input_model_mib"] = (
            result["cpu_model_mib"] + result["gpu_host_model_mib"]
        )
        result["cpu_model_mib"] = 0.0
        result["gpu_host_model_mib"] = 0.0

    result["cpu_mib"] = (
        result["cpu_model_mib"] +
        result["cpu_kv_mib"] +
        result["cpu_compute_mib"] +
        result["cpu_output_mib"] +
        result["cpu_rs_mib"]
    )

    result["gpu_host_mib"] = (
        result["gpu_host_model_mib"] +
        result["gpu_host_kv_mib"] +
        result["gpu_host_compute_mib"] +
        result["gpu_host_output_mib"] +
        result["gpu_host_rs_mib"]
    )

    result["gpu_mib"] = (
        result["gpu_model_mib"] +
        result["gpu_kv_mib"] +
        result["gpu_compute_mib"] +
        result["gpu_output_mib"] +
        result["gpu_rs_mib"]
    )

    for backend in LAYER_DEVICE.findall(log_text):
        key = "cpu_layers" if is_true_cpu_backend(backend) else "gpu_layers"
        result[key] += 1

    for backend in KV_LAYER_DEVICE.findall(log_text):
        key = "cpu_kv_layers" if is_true_cpu_backend(backend) else "gpu_kv_layers"
        result[key] += 1

    result["layer_devices_seen"] = bool(
        result["cpu_layers"] or result["gpu_layers"]
    )

    if result["layer_devices_seen"]:
        # What the split is asking is which processor does the work, and
        # llama.cpp answers that per layer. Buffer sizes cannot: a tensor
        # that no device was ever going to take sits in the same CPU
        # bucket as a layer that will be computed there.
        on_gpu = result["gpu_layers"] + result["gpu_kv_layers"] > 0
        on_cpu = result["cpu_layers"] + result["cpu_kv_layers"] > 0
    else:
        # A log from a build or a version that does not say where its
        # layers went. Falling back to buffer sizes is worse, but it is
        # the only thing left.
        on_gpu = result["gpu_model_mib"] + result["gpu_kv_mib"] > 0
        on_cpu = result["cpu_model_mib"] + result["cpu_kv_mib"] > 0

    if on_gpu and not on_cpu:
        result["processor"] = "GPU"
    elif on_gpu and on_cpu:
        result["processor"] = "GPU+CPU"
    else:
        result["processor"] = "CPU"

    return result


def load_llama_with_captured_stats(llama_cls, after_load=None, **kwargs):
    """Build a model and read back what it said about itself while doing so.

    after_load, when given, is called with the new model while the capture
    is still in place. It exists for work that allocates on top of the
    model and reports it the same way llama.cpp does -- initialising a
    projector, for one -- so that those buffers land in the same reading
    rather than being written to a stderr nobody is watching.
    """
    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        llm = llama_cls(**kwargs)
        if after_load is not None:
            after_load(llm)
    log_text = stderr_capture.getvalue()
    stats = parse_llama_verbose_load_log(log_text)
    return llm, stats, log_text
