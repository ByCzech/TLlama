import math

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator
from typing import List, Optional, Any, Dict, Union

from tllama.helpers.common import normalize_keep_alive
from tllama.schemas.generation import GenerationRequest


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    include_usage: StrictBool = False


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None

    # OpenAI tool-call conversation support.
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Preserve the native reasoning field when callers replay an assistant turn.
    reasoning_content: Optional[str] = None

    # Deprecated OpenAI compatibility path.
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(GenerationRequest):
    model: str
    messages: List[ChatMessage]

    stream: bool = False
    # No default value on purpose. chat_completions() forwards this into
    # build_sampling_kwargs() only when it is not None, so any non-None
    # default here is indistinguishable from a client that asked for that
    # value explicitly -- it would win over a virtual model's [sampling]
    # section and over TLlama's own baseline alike, on every request, with
    # nothing a client could do about it. Every other sampling field in
    # this schema already defaults to None for the same reason.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: StrictInt | None = None
    frequency_penalty: StrictFloat | None = None
    presence_penalty: StrictFloat | None = None
    stream_options: StreamOptions | None = None
    # Ollama-compatible extension, applied to residency rather than sampling.
    keep_alive: StrictStr | StrictInt | float | None = None

    @field_validator('keep_alive', mode='before')
    @classmethod
    def valid_keep_alive(cls, value):
        if value is not None:
            if type(value) not in (str, int, float) or type(value) is float and not math.isfinite(value):
                raise ValueError('keep_alive must be a duration or finite seconds')
            normalize_keep_alive(value)
        return value

    stop: Optional[Union[str, List[str]]] = None

    # OpenAI Chat Completions supports JSON mode / structured output shapes
    response_format: Optional[Dict[str, Any]] = None

    # OpenAI tool/function calling.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[StrictBool] = None

    # Deprecated OpenAI compatibility path.
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    # Simple compatibility path for reasoning control
    reasoning_effort: Optional[str] = None

    # Some clients may send nested reasoning config instead
    reasoning: Optional[Dict[str, Any]] = None
