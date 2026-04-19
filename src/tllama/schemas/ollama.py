from pydantic import BaseModel
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
    stream: bool = True
    options: Optional[dict[str, Any]] = None
