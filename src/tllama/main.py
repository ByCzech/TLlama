import uvicorn

from fastapi import FastAPI, Response
from .routers import openai
from .routers import ollama

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
    uvicorn.run(
        "tllama.main:app",
        host="127.0.0.1",
        port=8000,
        # reload=True,
        log_level="debug"
    )


if __name__ == "__main__":
    start_server()
