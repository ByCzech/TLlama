import json
import time
import asyncio

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.backend import model_manager

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)


def get_iso_time():
    """Ollama wants specific time format."""
    return datetime.now(timezone.utc).isoformat()


@router.get("/version")
async def get_version():
    return {"version": "0.0.0"}  # Return version, that client expect


@router.get("/tags")
async def list_models_ollama():
    local_models = model_manager.list_local_models()

    formatted_models = []
    for m in local_models:
        metadata_info = model_manager.get_model_metadata(m['id'])
        if not metadata_info:
            continue

        p_size = "unknown"
        if isinstance(metadata_info["params"], (int, str)) and int(metadata_info["params"]) > 0:
            p_size = f"{round(int(metadata_info['params']) / 1e9)}b"

        formatted_models.append({
            "name": f"{m['id']}",
            "model": f"{m['id']}",
            "modified_at": datetime.fromtimestamp(m["mtime"], timezone.utc).isoformat(),
            "size": m["size"],
            "digest": f"{m['sha256']}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": metadata_info["arch"],
                "families": [metadata_info["arch"]],
                "parameter_size": p_size,
                "quantization_level": metadata_info["bits"]
            }
        })

    return {"models": formatted_models}


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest):
    # If client sent model name, which we don't have loaded, get_model try to load it
    llm = await model_manager.get_model(request.model)

    response_iter = llm.create_chat_completion(
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        stream=True
    )

    async def generate_ollama():
        for chunk in response_iter:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            yield f"{json.dumps({'model': request.model, 'message': {'role': 'assistant', 'content': content}, 'done': False})}\n"
        yield f"{json.dumps({'model': request.model, 'done': True})}\n"

    return StreamingResponse(generate_ollama(), media_type="application/x-ndjson")


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    try:
        # Dynamic model loading on request
        llm = await model_manager.get_model(request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Preload model is in action?
    if not request.prompt.strip():
        print(f"DEBUG: Preload model {request.model} done.")

        # Ollama expects information, that is done
        response_data = {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": "",
            "done": True
        }

        if request.stream:
            async def preload_stream():
                yield f"{json.dumps(response_data)}\n"
            return StreamingResponse(preload_stream(), media_type="application/x-ndjson")

        return response_data

    # Prompt Assembly: If client sent system prompt, instert it before main prompt
    full_prompt = request.prompt
    if request.system:
        full_prompt = f"{request.system}\n\n{request.prompt}"

    # Config params from 'options' array
    generation_kwargs = {
        "max_tokens": request.options.get("num_predict", 512),
        "temperature": request.options.get("temperature", 0.8),
        "top_p": request.options.get("top_p", 0.9),
        "stop": request.options.get("stop", []),
        "echo": False,
    }

    if request.stream:
        async def generate_stream():
            start_time = time.time_ns()
            # llama-cpp-python generator
            response_iter = llm.create_completion(
                prompt=full_prompt,
                stream=True,
                **generation_kwargs
            )

            for chunk in response_iter:
                text = chunk["choices"][0]["text"]
                data = {
                    "model": request.model,
                    "created_at": get_iso_time(),
                    "response": text,
                    "done": False
                }
                yield f"{json.dumps(data)}\n"

            # Final chunk with statistics
            end_time = time.time_ns()
            final_data = {
                "model": request.model,
                "created_at": get_iso_time(),
                "done": True,
                "total_duration": end_time - start_time,
                "prompt_eval_count": len(llm.tokenize(full_prompt.encode("utf-8"))),
                # Context (tokens) for next prompt, if client wants it
                "context": []
            }
            yield f"{json.dumps(final_data)}\n"

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    else:
        # Non-streaming variant
        start_time = time.time_ns()
        response = llm.create_completion(
            prompt=full_prompt,
            stream=False,
            **generation_kwargs
        )
        end_time = time.time_ns()

        return {
            "model": request.model,
            "created_at": get_iso_time(),
            "response": response["choices"][0]["text"],
            "done": True,
            "total_duration": end_time - start_time
        }


@router.post("/show")
async def show_model_info(request: dict):
    model_name = request.get("name", "")
    metadata_info = model_manager.get_model_metadata(model_name)

    if not metadata_info:
        raise HTTPException(status_code=404, detail="Model doesn't exist")

    template = metadata_info.get("template", None) or "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: "

    return {
        "modelfile": f"FROM {model_name}.gguf\nTEMPLATE \"\"\"{template}\"\"\"",
        "parameters": "stop                           \"<|end_of_text|>\"",
        "template": template,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": metadata_info["arch"],
            "families": [metadata_info["arch"]],
            "parameter_size": f"{round(int(metadata_info['params']) / 1e9)}B" if metadata_info["params"] else "unknown",
            "quantization_level": metadata_info["bits"]
        }
    }
