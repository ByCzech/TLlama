from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict, Union, Literal

ThinkValue = Optional[Union[bool, Literal["low", "medium", "high"]]]


class Message(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    think: Optional[bool] = None
    format: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    keep_alive: Optional[str] = "5m"


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
