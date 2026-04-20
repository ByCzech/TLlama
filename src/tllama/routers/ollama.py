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


def _resolve_think_flag(request) -> bool | None:
    """
    Ollama-style precedence:
    1) top-level request.think
    2) compat fallback: options["think"]
    3) None = not explicitly set
    """
    raw = getattr(request, "think", None)

    if raw is None:
        opts = request.options or {}
        raw = opts.get("think", None)

    if raw is None:
        return None

    if isinstance(raw, bool):
        return raw

    # kompatibilita pro klienty, co pošlou string
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value == "true":
            return True
        if value in ("false", "none"):
            return False

    raise HTTPException(
        status_code=400,
        detail='invalid think value; expected true/false'
    )


def _build_messages_for_template(request):
    """
    Připraví message dicts i pro templaty, které umí pracovat s `thinking`.
    Když message model pole `thinking` nemá, nic se nestane.
    """
    result = []
    for m in request.messages:
        item = {
            "role": m.role,
            "content": m.content,
        }

        thinking = getattr(m, "thinking", None)
        if thinking:
            item["thinking"] = thinking

        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = tool_calls

        tool_name = getattr(m, "tool_name", None)
        if tool_name:
            item["tool_name"] = tool_name

        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            item["tool_call_id"] = tool_call_id

        result.append(item)

    return result


def _render_prompt_with_explicit_think(llm, metadata_info, request, think_enabled: bool) -> str:
    """
    Ollama-like prompt rendering:
    explicitní think=false se musí propsat do template vrstvy.
    """
    template = metadata_info.get("template")
    if not template:
        raise HTTPException(
            status_code=501,
            detail="model template is not available; cannot emulate ollama think=false for this model"
        )

    bos_id = llm.token_bos()
    eos_id = llm.token_eos()

    bos_token = (
        llm.detokenize([bos_id]).decode("utf-8", errors="ignore")
        if bos_id != -1 else ""
    )
    eos_token = (
        llm.detokenize([eos_id]).decode("utf-8", errors="ignore")
        if eos_id != -1 else ""
    )

    messages = _build_messages_for_template(request)

    # Kontext záměrně obsahuje jak "Ollama-like" názvy, tak běžné jinja/hf názvy.
    # Různé GGUF templaty používají různé konvence.
    context = {
        # běžné HF/Jinja názvy
        "messages": messages,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_generation_prompt": True,
        "tools": getattr(request, "tools", []) or [],
        "response": "",
        "enable_thinking": think_enabled,
        "thinking": think_enabled,

        # Ollama-like názvy
        "Messages": messages,
        "Tools": getattr(request, "tools", []) or [],
        "Response": "",
        "Think": think_enabled,
        "ThinkLevel": "",
        "IsThinkSet": True,
    }

    # Preferuj stejný formatter, který už jsi měl rozchozený.
    try:
        handler = lcf.Jinja2ChatFormatter(
            template=template,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        format_result = handler(**context)
        return format_result.prompt
    except Exception:
        # fallback přes čisté Jinja2, kdyby formatter nesežral extra kwargs
        try:
            from jinja2.sandbox import SandboxedEnvironment
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"failed to render template with explicit think flag: {e}"
            )

        env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string(template)
        return tmpl.render(**context)


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
    llm = await model_manager.get_model(request.model)
    metadata_info = model_manager.get_model_metadata(request.model) or {}

    opts = request.options or {}
    explicit_think = _resolve_think_flag(request)

    gen_params = {
        "max_tokens": opts.get("num_predict", 512),
        "temperature": opts.get("temperature", 0.8),
        "top_p": opts.get("top_p", 0.9),
        "stop": opts.get("stop", []),
    }

    if request.format == "json":
        gen_params["response_format"] = {"type": "json_object"}

    start_time = time.time_ns()

    # ------------------------------------------------------------------
    # PATH 1:
    # explicit think=false -> Ollama-like render prompt manually, then create_completion
    # ------------------------------------------------------------------
    if explicit_think is False:
        full_prompt = _render_prompt_with_explicit_think(
            llm=llm,
            metadata_info=metadata_info,
            request=request,
            think_enabled=False,
        )

        if request.stream:
            async def chat_stream_generator():
                response_iter = llm.create_completion(
                    prompt=full_prompt,
                    stream=True,
                    **gen_params
                )

                for chunk in response_iter:
                    text = chunk["choices"][0].get("text", "")
                    if not text:
                        continue

                    yield f"{json.dumps({
                        'model': request.model,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'message': {'role': 'assistant', 'content': text},
                        'done': False
                    })}\n"

                end_time = time.time_ns()
                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'done': True,
                    'total_duration': end_time - start_time,
                    'load_duration': 0,
                    'prompt_eval_count': len(request.messages),
                    'eval_count': None
                })}\n"

            return StreamingResponse(
                chat_stream_generator(),
                media_type="application/x-ndjson"
            )

        response = llm.create_completion(
            prompt=full_prompt,
            stream=False,
            **gen_params
        )
        end_time = time.time_ns()

        content = response["choices"][0].get("text", "")

        return {
            "model": request.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": content
            },
            "done": True,
            "total_duration": end_time - start_time,
            "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
            "eval_count": response.get("usage", {}).get("completion_tokens")
        }

    # ------------------------------------------------------------------
    # PATH 2:
    # default / explicit think=true -> necháme původní fungující create_chat_completion path
    # ------------------------------------------------------------------
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        async def chat_stream_generator():
            response_iter = llm.create_chat_completion(
                messages=messages,
                stream=True,
                **gen_params
            )

            for chunk in response_iter:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")

                yield f"{json.dumps({
                    'model': request.model,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'message': {'role': 'assistant', 'content': content},
                    'done': False
                })}\n"

            end_time = time.time_ns()
            yield f"{json.dumps({
                'model': request.model,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'done': True,
                'total_duration': end_time - start_time,
                'load_duration': 0,
                'prompt_eval_count': len(request.messages),
                'eval_count': 100
            })}\n"

        return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")

    response = llm.create_chat_completion(
        messages=messages,
        stream=False,
        **gen_params
    )
    end_time = time.time_ns()

    return {
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": response["choices"][0]["message"],
        "done": True,
        "total_duration": end_time - start_time,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens"),
        "eval_count": response.get("usage", {}).get("completion_tokens")
    }


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
    model_name = request.get("name", "") or request.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

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
