import logging

import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from .routers import openai, ollama
from tllama.backend import model_manager
from tllama.config import load_app_config_from_env, resolve_cors_origins
from tllama.helpers.cors import split_origin_patterns
from tllama.middleware import UndeclaredJsonBodyMiddleware
from tllama.errors import (
    http_exception_handler,
    toml_model_exception_handler,
    validation_exception_handler,
)
from tllama.helpers.model_toml import TomlModelError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.start()
    try:
        yield
    finally:
        await model_manager.shutdown()

app = FastAPI(
    title="Multi AI Proxy Server",
    lifespan=lifespan
)

app.add_middleware(UndeclaredJsonBodyMiddleware)

# Added last, so it is outermost: add_middleware inserts at the front of the
# stack. Every response has to carry the CORS headers, including the ones
# UndeclaredJsonBodyMiddleware produces itself without reaching a route --
# a page that cannot read the refusal sees an opaque network failure and has
# nothing to work from.
#
# Without this TLlama cannot be used from a browser at all: a page on any
# origin gets its request through and is then refused the answer.
#
# allow_credentials stays off. TLlama has no cookie or session for a
# browser to send, so switching it on would grant nothing except the
# ability for a page to spend somebody else's credentials once there is
# something to spend -- and the CORS specification forbids it alongside a
# wildcard origin in any case.
#
# The header list is spelled out rather than opened with '*' so that
# adding one is a decision. Authorization is there for a reverse proxy in
# front, and the x-stainless-* set for the OpenAI SDK, which sends thirteen
# of its own headers and fails preflight without them -- and an OpenAI
# client in a browser is precisely the caller this exists for.
#
# Only the methods TLlama actually answers. There is no PUT or PATCH
# route, so there is nothing to allow.
#
# max_age is ten minutes, not the twelve hours Ollama inherits from its
# middleware's defaults. A cached preflight means a change to the origin
# list does not reach a browser that already has one, and half a day of
# that during setup costs more than one extra request per ten minutes ever
# will.
_cors_origins = resolve_cors_origins()
_cors_exact, _cors_regex = split_origin_patterns(_cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_exact,
    allow_origin_regex=_cors_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "HEAD", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "User-Agent",
        "X-Requested-With",
        "OpenAI-Beta",
        "x-stainless-arch",
        "x-stainless-async",
        "x-stainless-custom-poll-interval",
        "x-stainless-helper-method",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-poll-helper",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-timeout",
    ],
    max_age=600,
)


# FastAPI's HTTPException subclasses Starlette's, so one registration covers
# both.
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# TomlModelError subclasses ValueError, so registration order does not help
# here: an endpoint that catches ValueError itself gets to it first. Those
# endpoints re-raise it explicitly so this handler stays the single place
# that decides how a broken .toml is reported.
app.add_exception_handler(TomlModelError, toml_model_exception_handler)


app.include_router(openai.router)
app.include_router(ollama.router)


@app.head("/")
@app.get("/")
async def root_ping():
    """
    Ollama CLI expect return 200 OK from root.
    Method HEAD must not return any body, only headers.
    """
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def start_server():
    config = load_app_config_from_env()

    # Only the entry point configures logging. Importing the package must not
    # touch the root logger, because an embedder owns that decision.
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Said out loud because a browser reports a rejected origin as a bare
    # network failure, with nothing on the server side to look at. The list
    # is resolved at import, so this reports it rather than deciding it.
    logging.getLogger(__name__).info(
        "Browser origins allowed: %s", ", ".join(_cors_origins)
    )

    kwargs = {
        "host": config.host,
        "port": config.port,
    }

    if config.reload:
        kwargs["reload"] = True

    if config.debug:
        kwargs["log_level"] = "debug"

    uvicorn.run("tllama.main:app", **kwargs)

    return


def main() -> int:
    """The `tllama` entry point named in pyproject.toml.

    A thin forward: the command lives in cli.py so that adding a
    subcommand does not mean editing the module that builds the
    application.
    """
    from tllama.cli import main as cli_main

    return cli_main()


# Below main(), not above it. Running the module directly executes the
# file top to bottom, so a __main__ block placed before the function it
# calls raises NameError before anything else happens -- while the
# installed console script, which imports the module and then looks the
# name up, works perfectly and hides it.
if __name__ == "__main__":
    raise SystemExit(main())
