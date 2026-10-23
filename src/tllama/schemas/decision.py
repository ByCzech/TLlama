"""Public native-compatible text Decision request, without generation options."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from tllama.decision_contract import prepare_decision_request


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: StrictStr = Field(min_length=1, description='Registered TLlama model reference')
    state: Any = Field(description='Non-null finite JSON state, passed literally to the native template')
    questions: dict[str, Any] = Field(description='1..32 ordered choice, score or noul questions; see the Decision contract')

    @model_validator(mode='before')
    @classmethod
    def literal_contract(cls, value):
        return prepare_decision_request(value)

    @field_validator('model')
    @classmethod
    def nonblank_model(cls, value):
        if not value.strip():
            raise ValueError('model must not be blank')
        return value
