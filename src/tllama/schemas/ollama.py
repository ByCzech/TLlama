from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any


class Message(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    options: Optional[dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    # Default_factory will ensure empty dict, if key is missing
    options: dict = Field(default_factory=dict)

    # This validator will ensure, empty dict even if client set "options": null
    @field_validator('options', mode='before')
    @classmethod
    def prevent_none_options(cls, v):
        if v is None:
            return {}
        return v
