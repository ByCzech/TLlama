import uvicorn

from fastapi import FastAPI, Response

from .routers import openai, ollama
from tllama.backend import model_manager
from tllama.config import load_app_config_from_env

app = FastAPI(title="Multi AI Proxy Server")


@app.on_event("startup")
async def on_startup():
    await model_manager.start()


@app.on_event("shutdown")
async def on_shutdown():
    await model_manager.shutdown()

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

    if ":" in os.getenv('TLLAMA_HOST', '127.0.0.1'):
        host = os.getenv('TLLAMA_HOST', '127.0.0.1').split(':')[0]
        port = int(os.getenv('TLLAMA_HOST', '127.0.0.1').split(':')[1])
    else:
        host = '127.0.0.1'
        port = 54800
    kwargs = dict(
        host=host,
        port=port
    )

    uvicorn.run("tllama.main:app", **kwargs)

    return


if __name__ == "__main__":
    start_server()
