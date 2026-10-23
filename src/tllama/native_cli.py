"""Serialize validated TLlama load settings for the pinned native parser."""

import csv
import io
from pathlib import Path

from tllama.native_params import NUMA_MODES, ROPE_MODES, SPLIT_MODES, MTP_DRAFT_N_MAX_DEFAULT, runtime_problem


VALUE_FLAGS = {
    "n_batch": "--batch-size", "n_ubatch": "--ubatch-size",
    "n_threads": "--threads", "n_threads_batch": "--threads-batch",
    "main_gpu": "--main-gpu", "seed": "--seed",
    "rope_freq_base": "--rope-freq-base", "rope_freq_scale": "--rope-freq-scale",
    "yarn_ext_factor": "--yarn-ext-factor", "yarn_attn_factor": "--yarn-attn-factor",
    "yarn_beta_fast": "--yarn-beta-fast", "yarn_beta_slow": "--yarn-beta-slow",
    "yarn_orig_ctx": "--yarn-orig-ctx",
}
BOOL_FLAGS = {
    "context_shift": ("--context-shift", "--no-context-shift"),
    "offload_kqv": ("--kv-offload", "--no-kv-offload"),
    "op_offload": ("--op-offload", "--no-op-offload"),
    "no_perf": ("--no-perf", "--perf"),
}
ENUM_FLAGS = {"split_mode": ("--split-mode", SPLIT_MODES),
              "rope_scaling_type": ("--rope-scaling", ROPE_MODES),
              "numa": ("--numa", NUMA_MODES)}
RUNTIME_OPTION_NAMES = frozenset(VALUE_FLAGS) | frozenset(BOOL_FLAGS) | frozenset(ENUM_FLAGS) | {
    "use_mmap", "use_mlock", "tensor_split", "swa_full", "kv_overrides",
    "lora_path", "lora_scale", "spm_infill", "verbose",
    "spec_type", "draft_n_max",
}
MMPROJ_FLAGS = {"image_min_tokens": "--image-min-tokens", "image_max_tokens": "--image-max-tokens",
                "batch_max_tokens": "--mtmd-batch-max-tokens"}


def csv_field(value: str) -> str:
    output = io.StringIO()
    csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="").writerow([value])
    return output.getvalue()


def runtime_cli_args(options) -> tuple[str, ...]:
    for name, value in options.items():
        if name not in RUNTIME_OPTION_NAMES:
            raise ValueError(f"{name!r} is not an additional native launch option")
        problem = runtime_problem(name, value)
        if problem:
            raise ValueError(problem)
    args = []
    # Omit speculative flags entirely unless selected. Existing/experimental
    # engines need not understand a new flag for ordinary inference to work.
    if "spec_type" in options:
        args += ["--spec-type", options["spec_type"]]
    if options.get("spec_type") == "draft-mtp":
        args += ["--spec-draft-n-max", str(options.get("draft_n_max", MTP_DRAFT_N_MAX_DEFAULT))]
    for name, flag in VALUE_FLAGS.items():
        if name in options:
            args += [flag, str(options[name])]
    for name, (yes, no) in BOOL_FLAGS.items():
        if name in options:
            args.append(yes if options[name] else no)
    for name, (flag, values) in ENUM_FLAGS.items():
        if name in options and values[options[name]] is not None:
            args += [flag, values[options[name]]]
    if "use_mmap" in options or "use_mlock" in options:
        modes = [name for key, name in (("use_mmap", "mmap"), ("use_mlock", "mlock")) if options.get(key, key == "use_mmap")]
        args += ["--load-mode", "+".join(modes) or "none"]
    if "tensor_split" in options:
        args += ["--tensor-split", ",".join(str(value) for value in options["tensor_split"])]
    if options.get("swa_full"):
        args.append("--swa-full")
    if options.get("spm_infill"):
        args.append("--spm-infill")
    for key, value in options.get("kv_overrides", {}).items():
        if type(value) is bool:
            typed = "bool:" + ("true" if value else "false")
        elif isinstance(value, int):
            typed = "int:" + str(value)
        elif isinstance(value, float):
            typed = "float:" + repr(value)
        else:
            typed = "str:" + value
        args += ["--override-kv", csv_field(key + "=" + typed)]
    if "lora_scale" in options and "lora_path" not in options:
        raise ValueError("lora_scale requires lora_path")
    if "lora_path" in options:
        path = options["lora_path"]
        if not Path(path).is_absolute():
            raise ValueError("resolved lora_path must be absolute")
        scale = options.get("lora_scale", 1.0)
        if scale == 1.0:
            args += ["--lora", csv_field(path)]
        else:
            if ":" in path:
                raise ValueError("the pinned --lora-scaled parser cannot represent ':' in a path")
            args += ["--lora-scaled", csv_field(f"{path}:{scale}")]
    return tuple(args)


def mmproj_cli_args(options) -> tuple[str, ...]:
    args = []
    for name, value in options.items():
        if name == "warmup":
            if type(value) is not bool:
                raise ValueError("mmproj.warmup must be boolean")
            args.append("--warmup" if value else "--no-warmup")
        elif name in MMPROJ_FLAGS:
            if not isinstance(value, int) or isinstance(value, bool) or not -(2**31) <= value < 2**31:
                raise ValueError(f"mmproj.{name} must fit the native int32 parameter")
            args += [MMPROJ_FLAGS[name], str(value)]
        else:
            raise ValueError(f"Unknown projector launch setting {name!r}")
    return tuple(args)
