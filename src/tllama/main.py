import logging

import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routers import openai, ollama
from tllama.backend import model_manager
from tllama.config import load_app_config_from_env
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
