import asyncio
import json

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest
from tllama.backend import model_manager, HF_LOOKUP_MISSING, PullProgress
from tllama.helpers.model_toml import TomlModelError
from tllama.version import get_version as get_tllama_version
from tllama.helpers.runner import RUNNER, validate_runner
from tllama.helpers.common import format_ollama_parameters
from tllama.helpers.reasoning_split import detect_reasoning_format
from tllama.native_routes import serve_native, MEMORY_UNAVAILABLE
from tllama.errors import ollama_stream_error_line

router = APIRouter(
    prefix="/api",
    tags=["Ollama API"]
)

# How often the pull stream reports download progress to the client.
# Ollama updates its progress bar far more often than this; a request every
# few hundred milliseconds is plenty for a status line and keeps the ndjson
# stream from being dominated by progress noise.
_PULL_PROGRESS_INTERVAL_SECONDS = 0.5


def _validate_runner(value) -> None:
    validate_runner(
        value, ignore_unsupported=model_manager.config.ignore_unsupported_runner
    )


@router.get("/version")
async def get_version():
    return {"version": get_tllama_version()}


@router.get("/tags")
async def list_models_ollama():
    local_models = await model_manager.list_local_models_with_metadata()

    formatted_models = []
    for m in local_models:
        family = m.get("arch", "unknown")

        formatted_models.append({
            "name": f"{m['id']}",
            "model": f"{m['id']}",
            "modified_at": datetime.fromtimestamp(m["mtime"], timezone.utc).isoformat(),
            "size": m["size"],
            "digest": m.get("digest", ""),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": family,
                "families": [family],
                "parameter_size": m.get("parameter_size", "unknown"),
                "quantization_level": m.get("bits", "unknown"),
                "runner": RUNNER,
            }
        })

    return {"models": formatted_models}


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest):
    _validate_runner(request.runner)
    return await serve_native(model_manager, request, api="ollama")


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    _validate_runner(request.runner)
    return await serve_native(model_manager, request, api="ollama", kind="completion")


@router.post("/show")
async def show_model_info(request: dict):
    _validate_runner(request.get("runner"))
    model_name = request.get("name", "") or request.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    metadata_info = await model_manager.get_model_metadata(model_name) or {}

    if not metadata_info:
        model_info = model_manager._build_model_file_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model doesn't exist")

    template = metadata_info.get("template") or "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: "

    family = metadata_info.get("arch", "unknown")

    # Real Ollama uses this list to tell a client what a model can do, and
    # clients built against it (any bot using the ollama Python client, for
    # instance) act on it directly -- an absent or empty list reads as "this
    # model can't do any of that", including things TLlama actually
    # supports. "completion" always holds; "thinking" is reported only when
    # the model's own template is one detect_reasoning_format recognizes as
    # having a thinking format, so a plain non-reasoning model is not
    # claimed to think just because it exists.
    #
    # Tools capability awaits its model/runtime checkpoint. Vision remains
    # unavailable on the public native text path until block 06.
    capabilities = ["completion"]
    if detect_reasoning_format(model_name, metadata_info) != "none":
        capabilities.append("thinking")
    # Vision capability is restored with native media inference in block 06.


    # Ollama's own `show --modelfile` hands back the definition a model
    # was built from, and the workflow that makes it worth having is
    # fetching it, editing it and putting it back. So this is the .toml
    # verbatim, comments and formatting intact, rather than the synthetic
    # FROM/TEMPLATE approximation that used to stand in for it and could
    # not be edited into anything.
    #
    # The fallback is not dead: a file can exist and still be unreadable,
    # and answering with something is better than failing a request that
    # only asked what a model is.
    modelfile = model_manager.read_model_definition(model_name)
    if modelfile is None:
        modelfile = f'FROM {model_name}\nTEMPLATE """{template}"""'

    return {
        "modelfile": modelfile,
        "parameters": format_ollama_parameters(metadata_info),
        "template": template,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": metadata_info.get("parameter_size", "unknown"),
            "quantization_level": metadata_info.get("bits", "unknown"),
            "runner": RUNNER,
        },
        # The official Ollama client's ShowResponse declares this field
        # Optional but without a pydantic default, which in practice makes
        # it required on the wire: a response missing the key fails
        # validation before the client ever looks at what's in it. Real
        # Ollama fills it with the model's raw GGUF key/value metadata;
        # metadata_raw is the same style of data, only over a narrower,
        # whitelisted set of keys, since that is what TLlama currently
        # reads from a GGUF header.
        "model_info": metadata_info.get("metadata_raw") or {},
        "capabilities": capabilities,
    }


@router.get("/ps")
async def list_running_models():
    raise HTTPException(status_code=503, detail=MEMORY_UNAVAILABLE)


@router.post("/pull")
async def pull_model_ollama(request: Request):
    payload = await request.json()
    _validate_runner(payload.get("runner"))

    model_ref = (payload.get("model") or payload.get("name") or "").strip()
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    stream = payload.get("stream", True)

    if not model_ref:
        raise HTTPException(status_code=400, detail="Missing model name")

    try:
        target_info = model_manager.resolve_hf_pull_target(model_ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Hugging Face uses token auth. For now, allow the password field
    # to act as a token placeholder if provided.
    hf_token = password or None

    # Published sha256 and size, looked up before the transfer. Best effort:
    # when unavailable the digest is simply computed from the file later.
    hf_file_info = await model_manager.fetch_hf_file_info(
        repo_id=target_info["repo_id"],
        filename=target_info["filename"],
        token=hf_token,
    )

    # hf_hub_download creates the whole directory chain implied by the file
    # name before it checks that the file exists, and leaves it behind when
    # the check fails. A reference built from a browser URL therefore litters
    # the model store with empty blob/main and resolve/main trees. Refuse here
    # instead, but only when the repository actually answered.
    if hf_file_info.status == HF_LOOKUP_MISSING:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{target_info['filename']}' not found in repository "
                f"'{target_info['repo_id']}'"
            ),
        )

    if stream is False:
        try:
            local_path = await model_manager.pull_hf_file(
                repo_id=target_info["repo_id"],
                filename=target_info["filename"],
                token=hf_token,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if await model_manager.store_hf_digest(local_path, hf_file_info) is None:
            await model_manager.get_model_digest(local_path)

        metadata = await model_manager.ensure_metadata_cache_for_path(local_path)
        await model_manager.ensure_virtual_model_toml(local_path, metadata)

        return JSONResponse({
            "status": "success",
            "path": local_path,
        })

    async def generate():
        yield json.dumps({"status": "resolving repository"}) + "\n"

        # Seed with what the pre-download lookup already told us (Content-
        # Length only arrives once the transfer itself starts), so the first
        # progress line does not have to wait for that.
        progress = PullProgress(total=hf_file_info.size or 0)

        download_task = asyncio.ensure_future(
            model_manager.pull_hf_file(
                repo_id=target_info["repo_id"],
                filename=target_info["filename"],
                token=hf_token,
                progress=progress,
            )
        )

        try:
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {download_task}, timeout=_PULL_PROGRESS_INTERVAL_SECONDS
                    )

                    completed, total = progress.snapshot()
                    yield json.dumps({
                        "status": "downloading model",
                        "completed": completed,
                        "total": total,
                    }) + "\n"

                    if done:
                        break

                local_path = await download_task
            except Exception as e:
                yield ollama_stream_error_line(str(e))
                return
        finally:
            # A client that disconnects mid-download (GeneratorExit) must not
            # leave this task dangling and unretrieved. The underlying thread
            # is not forcibly interrupted -- asyncio.to_thread offers no such
            # thing -- but the task itself is cleaned up promptly instead of
            # lingering until the transfer happens to finish on its own.
            if not download_task.done():
                download_task.cancel()

        if await model_manager.store_hf_digest(local_path, hf_file_info) is None:
            # The published digest was unusable. Hash the file here, while it
            # is still warm, rather than leaving the cost to a later listing.
            yield json.dumps({"status": "computing sha256 digest"}) + "\n"

            if await model_manager.get_model_digest(local_path) is None:
                yield json.dumps({"status": "digest unavailable"}) + "\n"

        yield json.dumps({"status": "creating metadata cache"}) + "\n"

        metadata = await model_manager.ensure_metadata_cache_for_path(local_path)
        if metadata is None:
            yield json.dumps({"status": "metadata cache unavailable"}) + "\n"
        else:
            yield json.dumps({"status": "metadata cache ready"}) + "\n"

        # Without this the bytes are on disk and nothing can see them: a
        # .gguf no .toml points at is not a model as far as any listing or
        # load is concerned.
        if await model_manager.ensure_virtual_model_toml(local_path, metadata) is None:
            yield json.dumps({"status": "no model definition written"}) + "\n"
        else:
            yield json.dumps({"status": "model definition ready"}) + "\n"

        yield json.dumps({
            "status": "success",
            "path": local_path,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.delete("/delete")
async def delete_model_ollama(request: Request):
    payload = await request.json()

    model_ref = (payload.get("name") or payload.get("model") or "").strip()
    if not model_ref:
        raise HTTPException(status_code=400, detail="Missing model name")

    try:
        result = await model_manager.delete_model(model_ref)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "deleted": result["deleted_path"],
        # The weights stay on disk. Reported so a client is not left to
        # infer from a bare success that several gigabytes just went away.
        "kept_model_file": result["kept_model_file"],
    }
