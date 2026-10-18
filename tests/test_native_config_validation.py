"""Native migration rejects misleading settings before expensive model work."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tllama.config import ConfigError
from tllama.helpers.model_toml import TomlModelError, parse_model_toml
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides


LLM = '[llm]\nmodel = "Local/m.gguf"\n'


def test_configuration_import_and_parse_do_not_load_python_binding():
    program = '''import sys
from tllama.helpers.model_toml import parse_model_toml
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides
parse_model_toml('[llm]\\nmodel="Local/m.gguf"\\n[runtime]\\ntype_kv="q4_0"\\n[sampling]\\ntemperature=0.5')
assert coerce_runtime_overrides({'device': 'CUDA0'}) == {'device': 'CUDA0'}
assert coerce_sampling_overrides({'top_k': '20'}) == {'top_k': 20}
assert 'llama_cpp' not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, "-c", program], env=env, check=True)


@pytest.mark.parametrize("name,value", [("chat_format", '"chatml"'), ("vocab_only", "true"),
                                      ("attention_type", "1"), ("logits_all", "false"),
                                      ("lora_base", '"Local/base.gguf"')])
def test_removed_runtime_option_is_reported_in_toml_and_environment(name, value):
    with pytest.raises(TomlModelError, match=name):
        parse_model_toml(LLM + f"[runtime]\n{name}={value}\n", source="Local/m.toml")
    with pytest.raises(ConfigError, match=f"TLLAMA_RUNTIME_{name.upper()}"):
        coerce_runtime_overrides({name: value})


@pytest.mark.parametrize("line", ['n_ctx=true', 'n_batch=0', 'n_threads="eight"',
                                'device="none,Vulkan0"', 'rope_scaling_type=3', 'numa=4',
                                'tensor_split=[0.0, 0.0]', 'type_k="q4_k"',
                                'verbose=false', 'rope_freq_scale=nan'])
def test_runtime_values_cannot_silently_change_meaning(line):
    with pytest.raises(TomlModelError, match="Local/m.toml"):
        parse_model_toml(LLM + "[runtime]\n" + line, source="Local/m.toml")


@pytest.mark.parametrize("name,value", [("temperature", "-1"), ("temperature", "nan"),
                                      ("temperature", "inf"), ("tfs_z", "1.0")])
def test_removed_or_nonfinite_sampler_fails_both_configuration_sources(name, value):
    with pytest.raises(ConfigError, match=name):
        coerce_sampling_overrides({name: value})
    with pytest.raises(TomlModelError, match=name):
        parse_model_toml(LLM + f"[sampling]\n{name}={value}\n")


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/v1/chat/completions"])
@pytest.mark.parametrize("empty", [False, True])
def test_negative_temperature_is_rejected_before_model_actions(client, monkeypatch, path, empty):
    from tllama.routers import ollama, openai

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid sampling reached model loading")

    async def forbidden_unload(*args, **kwargs):
        pytest.fail("Invalid sampling reached model unloading")

    for module in (ollama, openai):
        monkeypatch.setattr(module.model_manager, "acquire_native_model", forbidden)
        monkeypatch.setattr(module.model_manager, "unload_native_model", forbidden_unload)
    body = {"model": "m", "stream": False, "keep_alive": 0}
    if path == "/api/generate":
        body["prompt"] = "" if empty else "hi"
    else:
        body["messages"] = [] if empty else [{"role": "user", "content": "hi"}]
    if path.startswith("/api/"):
        body["options"] = {"temperature": -1}
    else:
        body["temperature"] = -1
    response = client.post(path, json=body)
    assert response.status_code == 400
    assert "temperature" in str(response.json())


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_explicit_tail_free_option_is_rejected_even_when_disabled(client, monkeypatch, path):
    from tllama.routers import ollama

    async def forbidden(*args, **kwargs):
        pytest.fail("Removed sampler reached loading")

    monkeypatch.setattr(ollama.model_manager, "acquire_native_model", forbidden)
    body = {"model": "m", "stream": False, "options": {"tfs_z": 1.0}}
    body.update({"prompt": "hi"} if path.endswith("generate") else {
        "messages": [{"role": "user", "content": "hi"}],
    })
    response = client.post(path, json=body)
    assert response.status_code == 400 and "tfs_z" in str(response.json())
