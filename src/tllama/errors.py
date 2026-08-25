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

# The OpenAI-compatible surface, which uses a different error shape.
OPENAI_API_PREFIX = "/v1/"

# OpenAI answers a malformed request with 400, and its Python client picks the
# exception class from the status alone: 400 is BadRequestError, 422 is
# UnprocessableEntityError.
OPENAI_VALIDATION_STATUS = 400

OPENAI_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def is_openai_api_path(path: str) -> bool:
    """Whether a request belongs to the OpenAI-compatible surface."""
    return path.startswith(OPENAI_API_PREFIX)


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


def openai_error_type(status_code: int) -> str:
    """Map an HTTP status to an OpenAI error type.

    The OpenAI schema types this field as a plain string and does not
    enumerate it, so this is the conventional vocabulary rather than a closed
    set. Ollama collapses everything outside 400 and 404 into api_error;
    reporting the actual category costs nothing and is strictly more
    informative to a client that looks.
    """
    if status_code >= 500:
        return "server_error"

    return OPENAI_ERROR_TYPES.get(status_code, "invalid_request_error")


def openai_error_response(
    status_code: int,
    message: str,
    param: str | None = None,
    code: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build an error body in the shape the OpenAI schema defines.

    ErrorResponse wraps an Error object whose type, message, param and code
    are all required, so param and code are emitted as null rather than
    omitted when there is nothing to say. The official Python client reads
    all three of type, param and code off the body.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": openai_error_type(status_code),
                "param": param,
                "code": code,
            }
        },
        headers=dict(headers) if headers else None,
    )


def validation_error_param(exc: RequestValidationError) -> str | None:
    """The offending field of the first validation failure, if identifiable.

    This is what the OpenAI param field is for. Ollama always reports null.
    """
    for error in exc.errors():
        location = ".".join(
            str(item) for item in error.get("loc", ()) if item != "body"
        )
        if location:
            return location

    return None


def openai_stream_error_frame(
    message: str,
    status_code: int = 500,
    param: str | None = None,
    code: str | None = None,
) -> str:
    """SSE frame for a stream that has already started.

    The headers are out by the time inference can fail, so the status is
    fixed at 200 and the failure can only travel as a frame. The official
    Python client inspects every plain data frame, one carrying no event
    line, for an "error" key and raises APIError with the message taken from
    that object.

    The frame has to arrive before [DONE], because the client stops reading
    there.
    """
    payload = {
        "error": {
            "message": message,
            "type": openai_error_type(status_code),
            "param": param,
            "code": code,
        }
    }

    return f"data: {json.dumps(payload)}\n\n"


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

    if is_openai_api_path(request.url.path):
        return openai_error_response(
            exc.status_code,
            str(exc.detail),
            headers=getattr(exc, "headers", None),
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

    if is_openai_api_path(request.url.path):
        return openai_error_response(
            OPENAI_VALIDATION_STATUS,
            describe_validation_error(exc),
            param=validation_error_param(exc),
        )

    return await default_validation_exception_handler(request, exc)
