from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Literal, Union


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None


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

    # Simple compatibility path for reasoning control
    reasoning_effort: Optional[str] = None

    # Some clients may send nested reasoning config instead
    reasoning: Optional[Dict[str, Any]] = None
