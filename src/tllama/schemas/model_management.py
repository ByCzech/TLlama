"""Explicit local model-definition operations, independent of inference options."""

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, Field


class DeleteRequest(BaseModel):
    # Keep ignored extension fields and the historical name-first alias order.
    name: StrictStr | None = None
    model: StrictStr | None = None


class CopyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: StrictStr = Field(min_length=1)
    destination: StrictStr = Field(min_length=1)
    force: StrictBool = False


class UploadedFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    digest: StrictStr = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    filename: StrictStr = Field(min_length=1, max_length=255)


class CreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: StrictStr = Field(min_length=1)
    definition: StrictStr = Field(min_length=1, max_length=2 * 1024 * 1024)
    uploads: dict[str, list[UploadedFile]] = Field(default_factory=dict)
    force: StrictBool = False
    dry_run: StrictBool = False
