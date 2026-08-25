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
    validation_exception_handler,
)


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


if __name__ == "__main__":
    start_server()
