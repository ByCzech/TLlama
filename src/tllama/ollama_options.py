"""Translate Ollama runner options without changing stored model configuration."""

from tllama.native_params import runtime_problem


RUNNER_ALIASES = {
    "num_ctx": "n_ctx", "num_batch": "n_batch", "num_gpu": "n_gpu_layers",
    "main_gpu": "main_gpu", "use_mmap": "use_mmap", "num_thread": "n_threads",
}


def ollama_runtime_overrides(options):
    values = {}
    for public, native in RUNNER_ALIASES.items():
        if public not in options:
            continue
        value = options[public]
        problem = runtime_problem(native, value)
        if problem:
            raise ValueError(f"options.{public}: {problem}")
        values[native] = value
    if "num_batch" in options:
        values["n_ubatch"] = options["num_batch"]
    if "num_thread" in options:
        values["n_threads_batch"] = options["num_thread"]
    if "draft_num_predict" in options:
        value = options["draft_num_predict"]
        problem = runtime_problem("draft_n_max", value)
        if problem:
            raise ValueError(f"options.draft_num_predict: {problem}")
        values["spec_type"] = "draft-mtp" if value else "none"
        if value:
            values["draft_n_max"] = value
    if "num_keep" in options:
        value = options["num_keep"]
        if type(value) is not int or not -1 <= value < 2**31:
            raise ValueError("options.num_keep must be -1 or a nonnegative int32")
        values["context_shift"] = True
    if options.get("num_predict") == -2:
        values["context_shift"] = False
    return values


def validate_ollama_options(options):
    ollama_runtime_overrides(options)
    if "repeat_last_n" in options:
        value = options["repeat_last_n"]
        if type(value) is not int or not -1 <= value < 2**31:
            raise ValueError("options.repeat_last_n must be -1 or a nonnegative int32")
    if "stop" in options:
        value = options["stop"]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(s, str) or "\0" in s for s in value):
            raise ValueError("options.stop must be a string or an array of strings")
