"""Prepare text/image chat payloads from a lease's resolved configuration.

No model lookup, transport, prompt rendering or public route selection occurs
here. Generate rendering and live vision capability checks live in native_routes;
generate images and FIM remain separate checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Literal

from tllama.helpers.chat import build_think_kwargs_ex, apply_default_system_prompt
from tllama.helpers.common import validate_format
from tllama.native_config import ResolvedWorkerConfig, build_native_sampling, _plain
from tllama.native_params import freeze_values
from tllama.helpers.vision import normalize_image_content_parts, to_image_data_url


class UnsupportedNativeInput(ValueError):
    """An input whose native public mapping has not been implemented yet."""


def _json_copy(value):
    try:
        return json.loads(json.dumps(_plain(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Native request data must contain finite JSON values") from exc


@dataclass(frozen=True)
class NativeRequest:
    kind: Literal["chat", "completion"]
    payload: Mapping
    reasoning_format: str = "none"
    think: bool | str | None = None

    def __post_init__(self):
        if self.kind not in {"chat", "completion"}:
            raise ValueError("Native request kind must be chat or completion")
        if self.reasoning_format not in {"none", "granite", "qwen_like", "gemma_channel", "gpt_oss_channel", "deepseek"}:
            raise ValueError("Unknown completion reasoning format")
        if self.think is not None and type(self.think) not in (bool, str):
            raise ValueError("think must be a boolean, string or None")
        if not isinstance(self.payload, Mapping):
            raise ValueError("Native request payload must be an object")
        object.__setattr__(self, "payload", freeze_values(_json_copy(self.payload)))

    def body(self):
        return _plain(self.payload)


def normalize_chat_input(messages):
    """Validate text/image inputs before admission, preserving all media order."""
    result = []
    for message in messages:
        value = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
        if value.get("function_call") is not None or value.get("role") == "function":
            raise UnsupportedNativeInput("Deprecated function-call history needs an explicit native mapping")
        content = value.get("content")
        if isinstance(content, list):
            if any(not isinstance(part, dict) or part.get("type") not in {'text', 'image_url'} for part in content):
                raise UnsupportedNativeInput("Native chat adapter accepts only text and image_url content parts")
            if any(part['type'] == 'text' and not isinstance(part.get("text"), str) for part in content):
                raise ValueError("Text content parts need string text")
            content = normalize_image_content_parts(content)
        elif content is None:
            content = ""
        if not isinstance(content, (str, list)) or not isinstance(value.get("role"), str):
            raise ValueError("Chat messages need a string role and text/image content")
        images = value.get('images') or []
        if not isinstance(images, list) or any(not isinstance(image, str) for image in images):
            raise ValueError('Message images must be an array of base64 strings')
        if images:
            content = [{'type':'text', 'text':content}] if isinstance(content, str) else content
            content = content + [{'type':'image_url', 'image_url':{'url':to_image_data_url(image)}} for image in images]
        elif isinstance(content, list) and all(part['type'] == 'text' for part in content):
            content = ''.join(part['text'] for part in content)
        normalized = {"role": value["role"], "content": content}
        for key in ("name", "tool_call_id", "tool_calls", "reasoning_content"):
            if key in value:
                normalized[key] = value[key]
        if "tool_calls" in normalized:
            calls = normalized["tool_calls"]
            if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                raise ValueError("tool_calls must be an array of objects")
            # Ollama's call objects need not carry OpenAI's type discriminator.
            normalized["tool_calls"] = [{"type": "function", **call} for call in calls]
        if value.get("tool_name") is not None:
            normalized["name"] = value["tool_name"]
        if value.get("thinking") is not None:
            normalized["reasoning_content"] = value["thinking"]
        result.append(normalized)
    return result


def _chat_messages(messages, resolved, normalized_messages=None):
    result = normalize_chat_input(messages) if normalized_messages is None else _json_copy(normalized_messages)
    # Presence, including an explicit empty system message, wins over TOML.
    return apply_default_system_prompt(result, {"default_system_prompt": resolved.system_prompt})


def _think_payload(think):
    if think is not None and type(think) not in (bool, str):
        raise ValueError("think must be a boolean, string or None")
    aliases = build_think_kwargs_ex(think)
    effort = aliases.pop("reasoning_effort", None)
    body = {"chat_template_kwargs": aliases} if aliases else {}
    if effort is not None:
        body["reasoning_effort"] = effort
    return body


def _response_format(value):
    if not isinstance(value, dict):
        raise ValueError("response_format must be an object")
    kind = value.get("type")
    if kind not in {"text", "json_object", "json_schema"}:
        raise ValueError("response_format type must be text, json_object or json_schema")
    if kind == "json_object" and "schema" in value and not isinstance(value["schema"], dict):
        raise ValueError("response_format.schema must be an object")
    if kind == "json_schema":
        wrapper = value.get("json_schema")
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
            raise ValueError("response_format.json_schema.schema must be an object")
    return _json_copy(value)


def prepare_chat(messages, opts, *, resolved: ResolvedWorkerConfig, metadata_info=None,
                 think=None, response_format=None, tools=None, tool_choice=None,
                 parallel_tool_calls=None, normalized_messages=None) -> NativeRequest:
    body = build_native_sampling(opts or {}, metadata_info, resolved=resolved)
    body.update({"messages": _chat_messages(messages, resolved, normalized_messages),
                 "reasoning_format": "deepseek"})  # Structured fields, never legacy reasoning-in-content.
    body.update(_think_payload(think))
    if response_format is not None:
        body["response_format"] = _response_format(response_format)
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if tool_choice.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Forced tool_choice needs type=function and function.name")
        selected = [tool for tool in (tools or []) if isinstance(tool, dict)
                    and isinstance(tool.get("function"), dict) and tool["function"].get("name") == function["name"]]
        if len(selected) != 1:
            raise ValueError("Forced tool_choice must identify exactly one declared tool")
        # The pinned parser accepts only auto/none/required. Restricting the
        # declarations to the chosen function preserves forced-call meaning.
        tools, tool_choice = selected, "required"
    elif tool_choice is not None and tool_choice not in {"auto", "none", "required"}:
        raise ValueError("tool_choice must be auto, none, required or a named function")
    for name, value in {"tools": tools, "tool_choice": tool_choice,
                        "parallel_tool_calls": parallel_tool_calls}.items():
        if value is not None:
            body[name] = value
    return NativeRequest("chat", body)


def prepare_ollama_chat(request, *, resolved: ResolvedWorkerConfig, metadata_info=None, normalized_messages=None):
    validate_format(request.format)
    response_format = None
    if request.format == "json":
        response_format = {"type": "json_object"}
    elif isinstance(request.format, dict):
        response_format = {"type": "json_object", "schema": request.format}
    return prepare_chat(request.messages, request.options, resolved=resolved, metadata_info=metadata_info,
                        think=request.think, response_format=response_format, tools=request.tools,
                        normalized_messages=normalized_messages)


def prepare_openai_chat(request, *, resolved: ResolvedWorkerConfig, metadata_info=None, normalized_messages=None):
    # Do not silently send names which the pinned native parser ignores.
    if request.functions is not None or request.function_call is not None:
        raise UnsupportedNativeInput("Deprecated functions/function_call needs an explicit native mapping; use tools/tool_choice")
    opts = {name: value for name, value in {
        "temperature": request.temperature, "top_p": request.top_p,
        "num_predict": request.max_tokens, "stop": request.stop,
    }.items() if value is not None}
    effort = request.reasoning_effort
    if effort is None and isinstance(request.reasoning, dict):
        effort = request.reasoning.get("effort")
    if effort is not None and not isinstance(effort, str):
        raise ValueError("reasoning effort must be a string")
    return prepare_chat(request.messages, opts, resolved=resolved, metadata_info=metadata_info,
                        think=effort, response_format=request.response_format, tools=request.tools,
                        tool_choice=request.tool_choice, parallel_tool_calls=request.parallel_tool_calls,
                        normalized_messages=normalized_messages)


def prepare_completion(prompt: str, opts, *, resolved: ResolvedWorkerConfig, metadata_info=None,
                       format_value=None, reasoning_format="none", think=None):
    """Accept already-rendered or raw text; do not invent template/FIM policy."""
    if not isinstance(prompt, str):
        raise ValueError("Completion prompt must be a string")
    validate_format(format_value)
    body = build_native_sampling(opts or {}, metadata_info, resolved=resolved)
    body["prompt"] = prompt
    if format_value == "json":
        body["json_schema"] = {"type": "object"}
    elif isinstance(format_value, dict):
        body["json_schema"] = format_value
    return NativeRequest("completion", body, reasoning_format, think)
