from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict


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
    think: Optional[bool] = None
    # Default_factory will ensure empty dict, if key is missing
    options: dict = Field(default_factory=dict)

    # This validator will ensure, empty dict even if client set "options": null
    @field_validator('options', mode='before')
    @classmethod
    def prevent_none_options(cls, v):
        if v is None:
            return {}
        return v
