"""Validate Ollama runner selection without changing the compute backend."""

from fastapi import HTTPException


RUNNER = "llamacpp"


def validate_runner(value, *, ignore_unsupported: bool = False) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="runner must be a string or null")
    if value in ("", RUNNER, "ggml") or ignore_unsupported:
        return
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported runner {value!r}; TLlama supports 'llamacpp' (alias: 'ggml').",
    )
