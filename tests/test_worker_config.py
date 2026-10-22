"""Launch contracts isolate native settings and preserve requested placement."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tllama.worker_config import WorkerLaunchSpec


def spec(**kwargs):
    return WorkerLaunchSpec(model=Path("/models/m.gguf"), device="none", **kwargs)


def options(launch):
    return launch.command(Path("/private/key"))


def test_cpu_disables_all_offload_including_the_projector():
    args = options(spec(mmproj=Path("/models/mmproj.gguf")))
    assert args[args.index("--device") + 1] == "none"
    assert args[args.index("--n-gpu-layers") + 1] == "0"
    assert "--no-op-offload" in args
    assert "--no-mmproj-offload" in args


@pytest.mark.parametrize("device", ["CUDA0", "ROCm0", "Vulkan0", "SYCL0", "CUDA0,CUDA1"])
@pytest.mark.parametrize("layers,expected", [(None, "all"), (-1, "all"), (20, "20")])
def test_device_selection_is_configurable_and_minus_one_means_all(device, layers, expected):
    args = options(WorkerLaunchSpec(model=Path("/models/m.gguf"), device=device, n_gpu_layers=layers))
    assert args[args.index("--device") + 1] == device
    assert args[args.index("--n-gpu-layers") + 1] == expected
    assert "--no-op-offload" not in args


def test_separate_kv_and_flash_attention_reach_the_command():
    args = options(spec(type_k="q8_0", type_v="q4_0", flash_attn=True, n_ctx=4096))
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q4_0"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--ctx-size") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--fit") + 1] == "off"


def test_paths_are_passed_as_single_arguments_without_a_shell():
    launch = WorkerLaunchSpec(model=Path("/models/file $(literal).gguf"), device="none")
    args = options(launch)
    assert args[args.index("--model") + 1] == "/models/file $(literal).gguf"


def test_worker_environments_are_independent_and_do_not_mutate_the_parent():
    inherited = {"PATH": "/bin", "LLAMA_ARG_CTX_SIZE": "broken", "LLAMA_ARG_NO_JINJA": "1",
                 "GGML_BACKEND_PATH": "/foreign/plugin", "MTMD_BACKEND_DEVICE": "ROCm0",
                 "AIP_HTTP_PORT": "9999", "CUDA_VISIBLE_DEVICES": "7"}
    original = dict(inherited)
    supplied = {"CUDA_VISIBLE_DEVICES": "2", "GGML_VK_VISIBLE_DEVICES": "0"}
    one = spec(environment=supplied)
    two = spec(environment={"CUDA_VISIBLE_DEVICES": "5"})
    supplied["CUDA_VISIBLE_DEVICES"] = "99"
    assert one.child_environment(inherited) == {
        "PATH": "/bin", "CUDA_VISIBLE_DEVICES": "2", "GGML_VK_VISIBLE_DEVICES": "0",
    }
    assert two.child_environment(inherited)["CUDA_VISIBLE_DEVICES"] == "5"
    assert inherited == original


@pytest.mark.parametrize("kwargs", [
    {"model": "relative.gguf"}, {"executable": "relative"}, {"mmproj": "relative"},
    {"n_ctx": -1}, {"n_ctx": True}, {"port": 65536}, {"port": -1},
    {"n_gpu_layers": 3}, {"n_gpu_layers": -2}, {"flash_attn": "on"},
    {"load_timeout": 0}, {"request_timeout": float("nan")},
    {"terminate_timeout": float("inf")}, {"kill_timeout": True},
    {"environment": {"LLAMA_ARG_HOST": "0.0.0.0"}},
    {"environment": {"GGML_BACKEND_PATH": "/override"}},
    {"environment": {"MTMD_BACKEND_DEVICE": "Vulkan0"}},
    {"environment": {"BAD=NAME": "x"}}, {"environment": {"X": "a\0b"}},
    {"device": "none,Vulkan0"}, {"device": ""}, {"alias": "one,two"},
    {"transport": "remote"}, {"transport": "uds", "port": 1234},
])
def test_invalid_launch_configuration_is_rejected(kwargs):
    values = {"model": Path("/models/m.gguf"), "device": "none", **kwargs}
    with pytest.raises(ValueError):
        WorkerLaunchSpec(**values)


def test_launch_module_does_not_import_inference_bindings():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", "import sys; import tllama.worker_config; "
                    "assert 'llama_cpp' not in sys.modules"], env=env, check=True)


def test_default_command_uses_a_private_socket_and_explicit_tcp_keeps_loopback():
    args = spec().command(Path('/private/key'), socket_file=Path('/private/worker.sock'))
    assert args[args.index('--host') + 1] == '/private/worker.sock'
    assert args[args.index('--api-key-file') + 1] == '/private/key'
    args = options(spec(transport='tcp', port=1234))
    assert args[args.index('--host') + 1] == '127.0.0.1'
    assert args[args.index('--port') + 1] == '1234'


@pytest.mark.parametrize('path', [Path('relative.sock'), Path('/private/no-extension'), Path('/private/a\0.sock')])
def test_socket_paths_must_be_absolute_and_use_the_native_sock_suffix(path):
    with pytest.raises(ValueError, match='socket_file'):
        spec().command(Path('/private/key'), socket_file=path)
