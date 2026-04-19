import os
import uvicorn

from fastapi import FastAPI, Response
from .routers import openai, ollama
from pydantic import TypeAdapter

app = FastAPI(title="Multi AI Proxy Server")

# OpenAI router
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
    if ":" in os.getenv('TLLAMA_HOST', '127.0.0.1'):
        host = os.getenv('TLLAMA_HOST', '127.0.0.1').split(':')[0]
        port = int(os.getenv('TLLAMA_HOST', '127.0.0.1').split(':')[1])
    else:
        host = '127.0.0.1'
        port = 8000
    kwargs = dict(
        host=host,
        port=port
    )
    if os.getenv('TLLAMA_UVICORN_RELOAD', False):
        kwargs['reload'] = TypeAdapter(bool).validate_python(os.getenv('TLLAMA_UVICORN_RELOAD'))
    if os.getenv('TLLAMA_DEBUG', False) and os.getenv('TLLAMA_DEBUG', '0').isdecimal() and os.getenv('TLLAMA_DEBUG', False):
        kwargs['log_level'] = "debug"

    uvicorn.run("tllama.main:app", **kwargs)

    return


if __name__ == "__main__":
    start_server()
