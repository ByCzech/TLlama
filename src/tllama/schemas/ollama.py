from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict, Union, Literal

ThinkValue = Optional[Union[bool, Literal["low", "medium", "high"]]]
FormatValue = Optional[Union[str, Dict[str, Any]]]


class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    thinking: Optional[str] = None
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    think: ThinkValue = None
    format: FormatValue = None
    options: dict = Field(default_factory=dict)
    keep_alive: Optional[Union[str, int, float]] = "5m"
    tools: Optional[List[Dict[str, Any]]] = None

    @field_validator("options", mode="before")
    @classmethod
    def prevent_none_options(cls, v):
        return {} if v is None else v


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    think: ThinkValue = None
    format: Optional[object] = None   # "json" nebo JSON schema objekt
    raw: Optional[bool] = False
    keep_alive: Optional[Union[str, int]] = "5m"
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def prevent_none_options(cls, v):
        return {} if v is None else v
