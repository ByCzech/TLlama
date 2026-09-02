import io
import re
import contextlib

BACKEND_PREFIX = r"(?:CPU(?:_Host)?|Vulkan\d+|Vulkan_Host|CUDA\d+(?:_Split)?|CUDA_Host|ROCm\d+|ROCm_Host|SYCL\d+|SYCL_Host|Metal)"

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
        "gpu_host_model_mib": 0.0,
        "gpu_host_kv_mib": 0.0,
        "gpu_host_compute_mib": 0.0,
        "gpu_host_output_mib": 0.0,
        "gpu_host_rs_mib": 0.0,
        "gpu_host_mib": 0.0,
        "processor": "100% CPU",
    }

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

    gpu_loaded_mib = result["gpu_model_mib"] + result["gpu_kv_mib"]
    cpu_loaded_mib = result["cpu_model_mib"] + result["cpu_kv_mib"]

    if gpu_loaded_mib > 0 and cpu_loaded_mib == 0:
        result["processor"] = "GPU"
    elif gpu_loaded_mib > 0 and cpu_loaded_mib > 0:
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
