"""Explicit local model-definition operations, independent of inference options."""

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, Field


class CopyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: StrictStr = Field(min_length=1)
    destination: StrictStr = Field(min_length=1)
    force: StrictBool = False
