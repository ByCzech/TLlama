"""Recognize unintegrated generation controls instead of silently losing them."""

from pydantic import BaseModel, Field, StrictBool, StrictInt, model_validator


class GenerationRequest(BaseModel):
    logprobs: StrictBool | None = None
    top_logprobs: StrictInt | None = Field(default=None, ge=0, le=20)
    debug_render_only: StrictBool | None = Field(default=None, alias='_debug_render_only')

    @model_validator(mode='after')
    def supported_generation_controls(self):
        if self.logprobs or self.top_logprobs:
            raise ValueError('logprobs/top_logprobs output is not integrated')
        if self.debug_render_only:
            raise ValueError('_debug_render_only is not integrated; no inference was started')
        return self


class OllamaGenerationRequest(GenerationRequest):
    truncate: StrictBool | None = None
    shift: StrictBool | None = None

    @model_validator(mode='after')
    def supported_context_controls(self):
        for name in ('truncate', 'shift'):
            if getattr(self, name) is not None:
                raise ValueError(f'Top-level {name} is not integrated; use the documented '
                                 'num_keep/runtime context policy')
        return self
