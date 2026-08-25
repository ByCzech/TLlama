from typing import Mapping

import json

from fastapi import Request
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
    request_validation_exception_handler as default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

# Routing failures such as an unknown path raise Starlette's HTTPException
# directly, not FastAPI's subclass, so a handler registered on the subclass
# would not see them. This is the import FastAPI's own documentation uses for
# overriding the default handlers.
from starlette.exceptions import HTTPException as StarletteHTTPException


# The Ollama-compatible surface. Everything else keeps FastAPI's own error
# rendering; /v1 has its own shape and is handled separately.
OLLAMA_API_PREFIX = "/api/"

# Ollama answers a malformed request with 400. FastAPI's default for a schema
# violation is 422, which no Ollama client expects.
OLLAMA_VALIDATION_STATUS = 400


def is_ollama_api_path(path: str) -> bool:
    """Whether a request belongs to the Ollama-compatible surface."""
    return path.startswith(OLLAMA_API_PREFIX)


def ollama_error_response(
    status_code: int,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build an error body in the shape Ollama clients parse.

    Ollama reports errors on its own API as a flat string under an "error"
    key, not as a structured object. The official Python client relies on
    exactly that: ResponseError parses the response body as JSON and takes
    .get('error'), falling back to the raw body. A FastAPI {"detail": ...}
    body therefore surfaces to the user as the entire JSON document, braces
    and all, instead of a message.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
        headers=dict(headers) if headers else None,
    )


def ollama_stream_error_line(message: str) -> str:
    """Error line for a stream that has already started.

    Once the first chunk is on the wire the status code is fixed at 200, so a
    failure can only be reported inside the body. Ollama sends the same flat
    "error" key it uses elsewhere, and its Python client inspects every
    streamed object for that key and raises ResponseError when it appears.

    A status string that merely begins with the word error does not work: the
    client only looks at the key.
    """
    return json.dumps({"error": message}) + "\n"


def describe_validation_error(exc: RequestValidationError) -> str:
    """Flatten a pydantic validation failure into one readable sentence.

    The Ollama error field is a string, so the structured list FastAPI
    produces has to collapse into something a person can act on.
    """
    described = []

    for error in exc.errors():
        location = ".".join(
            str(item) for item in error.get("loc", ()) if item != "body"
        )
        message = error.get("msg", "invalid value")
        described.append(f"{location}: {message}" if location else message)

    return "; ".join(described) or "invalid request body"


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> Response:
    if is_ollama_api_path(request.url.path):
        return ollama_error_response(
            exc.status_code,
            str(exc.detail),
            getattr(exc, "headers", None),
        )

    return await default_http_exception_handler(request, exc)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    if is_ollama_api_path(request.url.path):
        return ollama_error_response(
            OLLAMA_VALIDATION_STATUS,
            describe_validation_error(exc),
        )

    return await default_validation_exception_handler(request, exc)
