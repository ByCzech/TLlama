"""Strict public embedding requests; unsupported fields must not disappear."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator
from tllama.helpers.common import normalize_keep_alive


class EmbeddingRequestBase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: StrictStr = Field(min_length=1)

    @field_validator('model')
    @classmethod
    def nonblank_model(cls, value):
        if not value.strip():
            raise ValueError('model must not be blank')
        return value


class OllamaEmbeddingBase(EmbeddingRequestBase):
    runner: StrictStr | None = None
    options: dict = Field(default_factory=dict)
    keep_alive: StrictStr | StrictInt | float | None = None

    @field_validator('options', mode='before')
    @classmethod
    def options_or_empty(cls, value):
        return {} if value is None else value

    @field_validator('keep_alive', mode='before')
    @classmethod
    def valid_keep_alive(cls, value):
        if value is not None:
            if type(value) not in (str, int, float) or type(value) is float and not math.isfinite(value):
                raise ValueError('keep_alive must be a duration or finite seconds')
            normalize_keep_alive(value)
        return value

    @field_validator('options')
    @classmethod
    def context_only(cls, value):
        if set(value) - {'num_ctx'}:
            raise ValueError('Embedding options support only num_ctx')
        if 'num_ctx' in value and (type(value['num_ctx']) is not int or not 0 <= value['num_ctx'] < 2**31):
            raise ValueError('num_ctx must be a nonnegative int32')
        return value


class OllamaEmbedRequest(OllamaEmbeddingBase):
    input: StrictStr | list[StrictStr] | None = None
    truncate: StrictBool = True
    dimensions: StrictInt | None = Field(default=None, ge=1)

    @field_validator('truncate', mode='before')
    @classmethod
    def default_truncate(cls, value):
        return True if value is None else value

    @field_validator('input')
    @classmethod
    def bounded_batch(cls, value):
        if isinstance(value, list) and len(value) > 2048:
            raise ValueError('input supports at most 2048 items')
        return value


class OllamaLegacyEmbeddingRequest(OllamaEmbeddingBase):
    prompt: StrictStr = ''

    @field_validator('prompt', mode='before')
    @classmethod
    def empty_prompt(cls, value):
        return '' if value is None else value


class OpenAIEmbeddingRequest(EmbeddingRequestBase):
    input: Any
    dimensions: StrictInt | None = Field(default=None, ge=1)
    encoding_format: Literal['float', 'base64'] = 'float'
    user: StrictStr | None = None

    @field_validator('input')
    @classmethod
    def valid_input(cls, value):
        def tokens(row):
            return isinstance(row, list) and bool(row) and all(type(t) is int and 0 <= t < 2**31 for t in row)

        if isinstance(value, str):
            valid = bool(value)
        elif isinstance(value, list) and value:
            if all(type(t) is int for t in value):
                valid = tokens(value)
            else:
                valid = len(value) <= 2048 and (all(isinstance(t, str) and t for t in value)
                                              or all(tokens(t) for t in value))
        else:
            valid = False
        if not valid:
            raise ValueError('input must be nonempty text, texts, token IDs or token-ID arrays (at most 2048 inputs)')
        return value
