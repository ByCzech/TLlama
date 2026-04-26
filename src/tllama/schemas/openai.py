from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Literal, Union


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None

    # OpenAI tool-call conversation support.
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Deprecated OpenAI compatibility path.
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    stop: Optional[Union[str, List[str]]] = None

    # OpenAI Chat Completions supports JSON mode / structured output shapes
    response_format: Optional[Dict[str, Any]] = None

    # OpenAI tool/function calling.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # Deprecated OpenAI compatibility path.
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    # Simple compatibility path for reasoning control
    reasoning_effort: Optional[str] = None

    # Some clients may send nested reasoning config instead
    reasoning: Optional[Dict[str, Any]] = None
