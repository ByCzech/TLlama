"""Exercise full TOML/global/request resolution and native serialization."""

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import build_native_sampling, resolve_worker_config
from tllama.worker_config import WorkerLaunchSpec


MODEL = Path("/models/m.gguf")
LLM = '[llm]\nmodel="Local/m.gguf"\n'


def config(**kwargs):
    return BackendConfig(models_dir="/models", runtime_overrides={"device": "Vulkan0"}, **kwargs)


def resolve(text=LLM, **kwargs):
    return resolve_worker_config(config(), MODEL, parse_model_toml(text), **kwargs)


def command(resolved):
    return resolved.launch.command(Path("/private/api-key"), Path("/private/template"))


def option(args, name):
    return args[args.index(name) + 1]


def test_public_defaults_are_explicit_and_distinct_from_smoke_defaults():
    resolved = resolve()
    args = command(resolved)
    assert option(args, "--ctx-size") == "0"
    assert option(args, "--n-gpu-layers") == "all"
    assert option(args, "--flash-attn") == "off"
    assert option(args, "--batch-size") == option(args, "--ubatch-size") == "512"
    assert option(args, "--load-mode") == "mmap"
    assert "--threads" not in args and "--threads-batch" not in args
    assert option(args, "--fit") == "off" and option(args, "--parallel") == "1"
    sampling = build_native_sampling({}, resolved=resolved)
    assert sampling["temperature"] == 0.8 and sampling["top_p"] == 0.9
    assert sampling["n_predict"] == -1 and sampling["repeat_last_n"] == 64
    assert "seed" not in sampling and "tfs_z" not in sampling


def test_resolver_and_sampler_do_not_import_binding_or_start_a_process():
    repo = Path(__file__).resolve().parents[1]
    program = '''import sys
from pathlib import Path
from tllama.config import BackendConfig
from tllama.native_config import resolve_worker_config, build_native_sampling
resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}), Path('/missing/model.gguf'))
assert build_native_sampling({}, resolved=resolved)['n_predict'] == -1
assert 'llama_cpp' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", program], env={**os.environ, "PYTHONPATH": str(repo / "src")}, check=True)


def test_metadata_sampling_survives_when_no_toml_is_supplied():
    resolved = resolve_worker_config(config(), MODEL)
    body = build_native_sampling({}, {"sampling_defaults": {"temperature": 0.4},
                                      "stop_defaults": ["END"]}, resolved=resolved)
    assert body["temperature"] == 0.4 and body["stop"] == ["END"]


@pytest.mark.parametrize("device", ["none", "Vulkan0", "CUDA0", "ROCm0", "CUDA0,CUDA1"])
def test_model_selector_beats_global_and_context_keeps_request_priority(device):
    spec = parse_model_toml(LLM + f'[runtime]\ndevice="{device}"\nn_ctx=4096')
    global_config = replace(config(), context_length=8192)
    resolved = resolve_worker_config(global_config, MODEL, spec)
    assert resolved.launch.device == device and resolved.launch.n_ctx == 4096
    assert resolved.sources["device"] == "TOML"
    assert resolve_worker_config(global_config, MODEL, spec, num_ctx=0).launch.n_ctx == 0
    requested = resolve_worker_config(global_config, MODEL, spec, num_ctx=2048)
    assert requested.launch.n_ctx == 2048 and requested.sources["n_ctx"] == "request"


def test_absent_device_defers_automatic_selection_without_discovery():
    resolved = resolve_worker_config(BackendConfig(), MODEL)
    assert resolved.launch.device is None and resolved.launch.n_gpu_layers is None
    assert resolved.sources['device'] == 'backend priority'
    with pytest.raises(ValueError, match='Automatic device selection'):
        command(resolved)


@pytest.mark.parametrize("extra", ['n_gpu_layers=-1', 'n_gpu_layers=1', 'op_offload=true', 'offload_kqv=true'])
def test_cpu_rejects_explicit_conflicting_gpu_placement(extra):
    with pytest.raises(ConfigError, match="conflicts"):
        resolve(LLM + '[runtime]\ndevice="none"\n' + extra)


def test_cpu_defaults_disable_offload_without_mutating_other_workers():
    one = resolve(LLM + '[runtime]\ndevice="none"')
    two = resolve()
    args = command(one)
    assert option(args, "--n-gpu-layers") == "0"
    assert args.count("--no-op-offload") == 1 and "--no-kv-offload" in args
    assert "--no-op-offload" not in command(two)


@pytest.mark.parametrize("toml,expected", [
    ('type_kv="q4_0"', ("q4_0", "q4_0")),
    ('type_k="q5_1"', ("q5_1", "q8_0")),
    ('type_kv="q4_0"\ntype_v="q5_0"', ("q4_0", "q5_0")),
    ('type_k=2\ntype_v=8', ("q4_0", "q8_0")),
])
def test_kv_precedence_is_per_layer_then_per_side(toml, expected):
    global_config = replace(config(), kv_cache_type="f16", k_cache_type="bf16", v_cache_type="q8_0")
    resolved = resolve_worker_config(global_config, MODEL, parse_model_toml(LLM + '[runtime]\n' + toml))
    assert (resolved.launch.type_k, resolved.launch.type_v) == expected


@pytest.mark.parametrize("mmap,mlock,mode", [(False, False, "none"), (True, False, "mmap"),
                                          (False, True, "mlock"), (True, True, "mmap+mlock")])
def test_mmap_mlock_combinations_keep_their_explicit_meaning(mmap, mlock, mode):
    text = LLM + f'[runtime]\nuse_mmap={str(mmap).lower()}\nuse_mlock={str(mlock).lower()}'
    assert option(command(resolve(text)), "--load-mode") == mode


def test_enum_rope_thread_and_split_arguments_reach_native_cli():
    text = LLM + '''[runtime]
n_threads=3
n_threads_batch=7
split_mode=2
main_gpu=0
tensor_split=[0.4, 0.6]
numa=2
rope_scaling_type=2
rope_freq_scale=0.5
yarn_orig_ctx=4096
offload_kqv=false
op_offload=false
swa_full=true
no_perf=true
spm_infill=true
'''
    args = command(resolve(text))
    for flag, value in {"--threads": "3", "--threads-batch": "7", "--split-mode": "row",
                        "--numa": "isolate", "--rope-scaling": "yarn", "--yarn-orig-ctx": "4096",
                        "--rope-freq-scale": "0.5", "--tensor-split": "0.4,0.6", "--main-gpu": "0"}.items():
        assert option(args, flag) == value
    assert {"--no-kv-offload", "--no-op-offload", "--swa-full", "--no-perf", "--spm-infill"} <= set(args)
    unspecified = command(resolve(LLM + '[runtime]\nrope_scaling_type=-1\nnuma=false'))
    assert "--rope-scaling" not in unspecified and "--numa" not in unspecified


def test_typed_kv_overrides_escape_csv_without_shell_interpretation():
    text = LLM + '''[runtime.kv_overrides]
"tokenizer.ggml.add_bos_token"=false
"int.key"=42
"float.key"=1.5
"str.key"='a,b "quoted" $(literal)'
'''
    args = command(resolve(text))
    entries = [next(csv.reader([args[i + 1]]))[0] for i, arg in enumerate(args) if arg == "--override-kv"]
    assert entries == ["tokenizer.ggml.add_bos_token=bool:false", "int.key=int:42",
                       "float.key=float:1.5", 'str.key=str:a,b "quoted" $(literal)']


def test_lora_is_resolved_inside_store_and_scaled_explicitly():
    resolved = resolve(LLM + '[runtime]\nlora_path="Local/a,b.gguf"\nlora_scale=0.5')
    value = next(csv.reader([option(command(resolved), "--lora-scaled")]))[0]
    assert value == "/models/Local/a,b.gguf:0.5"
    with pytest.raises(ConfigError, match="inside the model store"):
        resolve(LLM + '[runtime]\nlora_path="../../outside.gguf"')
    with pytest.raises(ConfigError, match="requires lora_path"):
        resolve(LLM + '[runtime]\nlora_scale=0.5')


def test_sampler_resolution_keeps_every_layer_and_runtime_fallbacks():
    spec = parse_model_toml(LLM + '''[runtime]
seed=123
last_n_tokens_size=17
[sampling]
temperature=0.6
max_tokens=256
stop=["END"]
[sampling.logit_bias]
"42"=0.5
''')
    global_config = replace(config(), sampling_overrides={"temperature": "0.4", "top_k": "30"})
    resolved = resolve_worker_config(global_config, MODEL, spec)
    recommended = {"recommended_sampling": {"temperature": 0.2, "top_k": 20, "top_p": 0.8}}
    body = build_native_sampling({"temperature": 0.7, "num_predict": 8, "mirostat": 2}, recommended, resolved=resolved)
    assert body["temperature"] == 0.7 and body["top_k"] == 30 and body["top_p"] == 0.8
    assert body["n_predict"] == 8 and body["mirostat"] == 2 and body["repeat_last_n"] == 17
    assert body["logit_bias"] == {"42": 0.5} and body["stop"] == ["END"]
    assert "seed" not in body and option(command(resolved), "--seed") == "123"
    assert build_native_sampling({}, recommended, resolved=resolved)["temperature"] == 0.6
    assert build_native_sampling({"seed": 0}, resolved=resolved)["seed"] == 0


@pytest.mark.parametrize("limit", [None, 0, -1, -2, -100])
def test_nonpositive_public_token_limits_are_not_native_zero_prefill(limit):
    assert build_native_sampling({"num_predict": limit})["n_predict"] == -1


def test_disabled_top_k_is_translated_explicitly_and_metadata_is_not_mutated():
    opts = {"top_k": -4}
    metadata = {"recommended_sampling": {"top_p": 0.8}}
    assert build_native_sampling(opts, metadata)["top_k"] == 0
    assert opts == {"top_k": -4} and metadata == {"recommended_sampling": {"top_p": 0.8}}


def test_projector_flags_and_worker_wide_warmup_are_serialized():
    spec = parse_model_toml(LLM + '''[mmproj]
model="Local/projector.gguf"
warmup=false
image_min_tokens=32
image_max_tokens=256
batch_max_tokens=128
''')
    with pytest.raises(ConfigError, match="projector path"):
        resolve_worker_config(config(), MODEL, spec)
    resolved = resolve_worker_config(config(), MODEL, spec, mmproj=Path("/models/Local/projector.gguf"))
    args = command(resolved)
    assert "--no-warmup" in args
    for flag, value in {"--image-min-tokens": "32", "--image-max-tokens": "256",
                        "--mtmd-batch-max-tokens": "128"}.items():
        assert option(args, flag) == value


def test_launch_options_are_copied_and_cannot_override_supervisor_fields():
    options = {"tensor_split": [0.4, 0.6], "kv_overrides": {"k": "v"}}
    launch = WorkerLaunchSpec(model=MODEL, device="Vulkan0", runtime_options=options)
    options["tensor_split"][0] = 0.9
    options["kv_overrides"]["k"] = "changed"
    assert option(launch.command(Path("/private/key")), "--tensor-split") == "0.4,0.6"
    with pytest.raises(TypeError):
        launch.runtime_options["kv_overrides"]["k"] = "changed"
    with pytest.raises(ValueError, match="additional native launch"):
        WorkerLaunchSpec(model=MODEL, device="none", runtime_options={"parallel": 4})


@pytest.mark.parametrize("filename,context,k,v,projector", [
    ("Qwen3.6-35B-A3B-UD-IQ3_S.toml", 4096, "q4_0", "q4_0", False),
    ("Qwen3.6-35B-A3B-UD-mmproj-IQ3_S.toml", 0, "q8_0", "q4_0", True),
])
def test_supplied_profiles_work_through_the_preview_entry_point(tmp_path, filename, context, k, v, projector):
    repo = Path(__file__).resolve().parents[1]
    profile = repo / "tests" / "fixtures" / "native-config" / filename
    env = {name: value for name, value in os.environ.items() if not name.startswith("TLLAMA_")}
    env.update(PYTHONPATH=str(repo / "src"), TLLAMA_RUNTIME_DEVICE="Vulkan0")
    result = subprocess.run([sys.executable, str(repo / "scripts" / "worker-config.py"),
                             "--toml", str(profile), "--models-dir", str(tmp_path)],
                            env=env, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    args = report["argv"]
    assert report["dry_run"] is True and report["gguf_metadata_read"] is False
    assert report["runtime"]["n_ctx"] == context
    assert option(args, "--cache-type-k") == k and option(args, "--cache-type-v") == v
    assert option(args, "--flash-attn") == "on"
    assert ("--mmproj" in args) is projector
