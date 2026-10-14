"""Public text inputs become native payloads without loading a model."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from tllama.config import BackendConfig
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import resolve_worker_config
from tllama.native_requests import (NativeRequest, UnsupportedNativeInput, prepare_chat,
                                    prepare_completion, prepare_ollama_chat, prepare_openai_chat)
from tllama.schemas.ollama import OllamaChatRequest
from tllama.schemas.openai import ChatCompletionRequest


@pytest.fixture
def resolved():
    spec = parse_model_toml('[llm]\nmodel="Local/m.gguf"\n[system]\nprompt="TOML system"\n'
                            '[sampling]\ntemperature=0.3\ntop_k=12\nstop=["TOML STOP"]')
    config = BackendConfig(runtime_overrides={"device": "none"},
                           sampling_overrides={"top_p": "0.7", "temperature": "0.6"})
    return resolve_worker_config(config, Path('/models/m.gguf'), spec)


def test_same_snapshot_and_precedence_for_both_chat_apis(resolved):
    metadata = {"recommended_sampling": {"temperature": 0.1, "min_p": 0.2},
                "sampling_defaults": {"temperature": 0.99}, "default_system_prompt": "stale TOML"}
    ollama = OllamaChatRequest(model="m", messages=[{"role": "user", "content": "hi"}],
                               options={"temperature": 0, "num_predict": 9})
    openai = ChatCompletionRequest(model="m", messages=[m.model_dump() for m in ollama.messages], temperature=0, max_tokens=9)
    a = prepare_ollama_chat(ollama, resolved=resolved, metadata_info=metadata).body()
    b = prepare_openai_chat(openai, resolved=resolved, metadata_info=metadata).body()
    assert a == b
    assert a["temperature"] == 0 and a["top_p"] == 0.7 and a["top_k"] == 12 and a["min_p"] == 0.2
    assert a["n_predict"] == 9 and a["stop"] == ["TOML STOP"]
    assert a["messages"][0] == {"role": "system", "content": "TOML system"}
    assert metadata["sampling_defaults"]["temperature"] == 0.99


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("content", ["", "explicit"])
def test_explicit_system_content_suppresses_toml_even_when_empty(resolved, role, content):
    messages = [{"role": role, "content": content}, {"role": "user", "content": "hi"}]
    assert prepare_chat(messages, {}, resolved=resolved).body()["messages"] == messages


def test_omitted_openai_sampling_preserves_toml_and_global_defaults(resolved):
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    body = prepare_openai_chat(request, resolved=resolved).body()
    assert body["temperature"] == 0.3 and body["top_p"] == 0.7 and body["n_predict"] == -1
    assert "seed" not in body and "max_tokens" not in body


@pytest.mark.parametrize("limit", [None, 0, -1, -2])
def test_limit_normalization_matches_both_surfaces(resolved, limit):
    oa = ChatCompletionRequest(model="m", messages=[], max_tokens=limit)
    ol = OllamaChatRequest(model="m", messages=[], options={"num_predict": limit})
    assert prepare_openai_chat(oa, resolved=resolved).body()["n_predict"] == -1
    assert prepare_ollama_chat(ol, resolved=resolved).body()["n_predict"] == -1


@pytest.mark.parametrize("value,enabled,effort", [(False, False, "none"), ("FALSE", False, "none"),
    ("none", False, "none"), (True, True, None), ("true", True, None), ("high", True, "high")])
def test_think_aliases_are_typed_and_native_reasoning_is_structured(resolved, value, enabled, effort):
    body = prepare_chat([], {}, resolved=resolved, think=value).body()
    kwargs = body["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is kwargs["Think"] is kwargs["thinking"] is enabled
    assert kwargs["IsThinkSet"] is True
    assert body.get("reasoning_effort") == effort
    assert body["reasoning_format"] == "deepseek"


def test_unspecified_think_does_not_override_template_defaults(resolved):
    body = prepare_chat([], {}, resolved=resolved).body()
    assert "chat_template_kwargs" not in body and "reasoning_effort" not in body


def test_openai_effort_nested_fallback_and_top_level_priority(resolved):
    request = ChatCompletionRequest(model="m", messages=[], reasoning={"effort": "low"})
    assert prepare_openai_chat(request, resolved=resolved).body()["reasoning_effort"] == "low"
    request.reasoning_effort = "none"
    assert prepare_openai_chat(request, resolved=resolved).body()["reasoning_effort"] == "none"


@pytest.mark.parametrize("value", [0, 1, [], {}, 0.5])
def test_think_is_not_coerced_from_non_boolean_non_string(resolved, value):
    with pytest.raises(ValueError, match="think must"):
        prepare_chat([], {}, resolved=resolved, think=value)


@pytest.mark.parametrize("format_value,native", [(None, None), ("", None),
    ("json", {"type": "json_object"}), ({"type": "array", "items": {"type": "integer"}},
     {"type": "json_object", "schema": {"type": "array", "items": {"type": "integer"}}})])
def test_ollama_output_format_has_no_binding_grammar_object(resolved, format_value, native):
    request = OllamaChatRequest(model="m", messages=[], format=format_value)
    assert prepare_ollama_chat(request, resolved=resolved).body().get("response_format") == native


def test_completion_receives_native_json_schema_and_exact_prompt(resolved):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    body = prepare_completion("<bos> rendered exact\n", {"stop": "REQ"}, resolved=resolved, format_value=schema).body()
    assert body["prompt"] == "<bos> rendered exact\n" and body["json_schema"] == schema
    assert body["stop"] == ["REQ"] and "response_format" not in body and "grammar" not in body
    assert prepare_completion("raw", {}, resolved=resolved, format_value="json").body()["json_schema"] == {"type": "object"}


@pytest.mark.parametrize("value", ["yaml", [], 42])
def test_invalid_completion_format_is_refused_before_transport(resolved, value):
    with pytest.raises(ValueError):
        prepare_completion("hi", {}, resolved=resolved, format_value=value)


@pytest.mark.parametrize("value", [{"type": "text"}, {"type": "json_object"},
    {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {"type": "array"}}}])
def test_openai_response_format_native_shapes_are_preserved(resolved, value):
    request = ChatCompletionRequest(model="m", messages=[], response_format=value)
    assert prepare_openai_chat(request, resolved=resolved).body()["response_format"] == value


@pytest.mark.parametrize("value", [{}, {"type": "yaml"}, {"type": "json_object", "schema": "bad"},
    {"type": "json_schema"}, {"type": "json_schema", "json_schema": {"schema": []}}])
def test_unknown_or_incomplete_openai_format_is_not_silently_ignored(resolved, value):
    request = ChatCompletionRequest(model="m", messages=[], response_format=value)
    with pytest.raises(ValueError, match="response_format"):
        prepare_openai_chat(request, resolved=resolved)


def test_text_parts_and_tool_history_are_preserved_without_orchestration(resolved):
    calls = [{"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]
    messages = [{"role": "assistant", "content": None, "tool_calls": calls, "thinking": "earlier thought"},
                {"role": "tool", "tool_call_id": "call-1", "tool_name": "weather", "content": "sunny"},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    body = prepare_chat(messages, {}, resolved=resolved, tools=[{"type": "function", "function": {"name": "weather"}}],
                        tool_choice="none", parallel_tool_calls=False).body()
    assert body["messages"][1]["tool_calls"] == calls
    assert body["messages"][1]["reasoning_content"] == "earlier thought"
    assert body["messages"][2]["name"] == "weather" and body["messages"][2]["tool_call_id"] == "call-1"
    assert body["messages"][3]["content"] == "ab"
    assert body["parallel_tool_calls"] is False and body["tool_choice"] == "none"


def test_forced_named_tool_is_required_with_only_its_original_definition(resolved):
    tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in ["a", "b"]]
    body = prepare_chat([], {}, resolved=resolved, tools=tools,
                        tool_choice={"type": "function", "function": {"name": "b"}}).body()
    assert body["tools"] == [tools[1]] and body["tool_choice"] == "required"
    assert len(tools) == 2


@pytest.mark.parametrize("choice", [{}, {"type": "function", "function": {"name": "missing"}}, "unknown"])
def test_invalid_forced_tool_choice_cannot_fall_back_to_auto(resolved, choice):
    with pytest.raises(ValueError, match="tool_choice"):
        prepare_chat([], {}, resolved=resolved, tools=[], tool_choice=choice)


def test_ollama_tool_history_gains_required_native_type_without_losing_arguments(resolved):
    arguments = {"city": "Praha"}
    body = prepare_chat([{"role": "assistant", "tool_calls": [{"function": {"name": "weather", "arguments": arguments}}]}],
                        {}, resolved=resolved).body()
    assert body["messages"][-1]["tool_calls"] == [{"type": "function", "function": {"name": "weather", "arguments": arguments}}]


@pytest.mark.parametrize("message", [{"role": "user", "images": ["base64"], "content": "hi"},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]},
    {"role": "assistant", "function_call": {"name": "f"}}, {"role": "function", "content": "x"}])
def test_unmapped_inputs_are_refused_instead_of_lost(resolved, message):
    with pytest.raises(ValueError):
        prepare_chat([message], {}, resolved=resolved)


@pytest.mark.parametrize("field,value", [("functions", []), ("function_call", "auto")])
def test_deprecated_function_request_waits_for_explicit_05b_mapping(resolved, field, value):
    request = ChatCompletionRequest(model="m", messages=[], **{field: value})
    with pytest.raises(UnsupportedNativeInput, match="Deprecated"):
        prepare_openai_chat(request, resolved=resolved)


def test_prepared_payload_is_a_deep_snapshot_and_each_body_is_fresh(resolved):
    messages = [{"role": "user", "content": "original"}]
    prepared = prepare_chat(messages, {}, resolved=resolved, tools=[{"function": {"name": "f"}}])
    messages[0]["content"] = "changed"
    body = prepared.body()
    body["tools"][0]["function"]["name"] = "changed"
    assert prepared.body()["tools"][0]["function"]["name"] == "f"
    assert prepared.body()["messages"][-1]["content"] == "original"
    with pytest.raises(TypeError):
        prepared.payload["temperature"] = 9


@pytest.mark.parametrize("payload", [{"x": float("nan")}, {"x": object()}])
def test_non_json_or_nonfinite_request_data_is_refused(payload):
    with pytest.raises(ValueError, match="finite JSON"):
        NativeRequest("chat", payload)


def test_request_adapter_does_not_import_binding_or_start_application():
    code = 'import sys; import tllama.native_requests; assert "llama_cpp" not in sys.modules; assert "tllama.main" not in sys.modules'
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
