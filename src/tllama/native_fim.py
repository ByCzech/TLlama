"""Public suffix mapping using full, literal input and native output stops."""

from dataclasses import replace

from fastapi import HTTPException

from tllama.native_requests import prepare_completion
from tllama.worker import WorkerProtocolError


def validate_fim_input(request):
    if request.raw or request.template is not None or request.system is not None:
        raise ValueError("FIM suffix does not support raw, template or system; put code context in prompt/suffix")
    if request.images:
        raise ValueError("FIM suffix does not support images")
    if request.think not in (None, False):
        raise ValueError("FIM suffix does not support thinking")
    if request.options.get("tllama"):
        raise ValueError("FIM suffix does not use chat template variables")


async def prepare_fim(request, lease, metadata):
    inventory = metadata.get("fim")
    order = lease.resolved.fim_order
    if order == "disabled" or not inventory:
        raise HTTPException(501, "This model has no supported FIM recipe (or FIM is disabled)")
    if order == "auto":
        if lease.resolved.runtime.get("spm_infill"):
            raise ValueError("FIM suffix uses explicit token prompts; select [fim].order instead of runtime.spm_infill")
        order = "psm"
    worker = lease.worker
    pieces = inventory["tokens"]
    # Checking integer pieces also handles NORMAL delimiters whose spelling
    # tokenizes into several unrelated IDs. Never trust a stale GGUF/cache ID.
    actual = await worker.token_pieces(list(pieces.values()))
    if actual != list(pieces):
        raise WorkerProtocolError("GGUF FIM token pieces disagree with the loaded tokenizer")
    for key in ("prefix", "suffix", "middle"):
        piece = inventory[key]
        if await worker.tokenize(piece, add_special=False) != [pieces[piece]]:
            raise WorkerProtocolError("FIM control spelling is not a single loaded token")

    # Infer the active tokenizer's BOS policy, including runtime overrides.
    # Explicit token input must not contain an automatically appended EOS.
    ordinary = await worker.tokenize("x", add_special=False, parse_special=False)
    framed = await worker.tokenize("x", add_special=True, parse_special=False)
    bos_piece = (worker.props or {}).get("bos_token") or ""
    bos_ids = await worker.tokenize(bos_piece, add_special=False) if bos_piece else []
    bos = bos_ids if len(bos_ids) == 1 and framed != ordinary and framed[:1] == bos_ids else []
    if not ordinary or framed[:len(bos) + len(ordinary)] != bos + ordinary:
        raise WorkerProtocolError("Unknown FIM tokenizer framing")
    tail = framed[len(bos) + len(ordinary):]
    eos_piece = (worker.props or {}).get("eos_token") or ""
    eos_ids = await worker.tokenize(eos_piece, add_special=False) if eos_piece else []
    if tail and tail != eos_ids:
        raise WorkerProtocolError("Unknown FIM tokenizer suffix framing")

    prefix = await worker.tokenize(request.prompt, add_special=False, parse_special=False)
    suffix = await worker.tokenize(request.suffix, add_special=False, parse_special=False)
    pre, suf, mid = (pieces[inventory[key]] for key in ("prefix", "suffix", "middle"))
    if order == "psm":
        tokens = bos + [pre] + prefix + [suf] + suffix + [mid]
    elif order == "training_spm":
        tokens = bos + [pre, suf] + suffix + [mid] + prefix
    else:  # Explicit native_spm, distinct from the training SPM layout.
        tokens = bos + [suf] + suffix + [pre] + prefix + [mid]
    context = (worker.props or {}).get("default_generation_settings", {}).get("n_ctx")
    if type(context) is not int or context <= 0:
        raise WorkerProtocolError("FIM requires the effective native slot context")
    if len(tokens) >= context:
        raise ValueError(f"FIM input needs {len(tokens)} tokens; effective context is {context} "
                         "and must leave room for generation. Input was not truncated")

    # Use the same sampling/schema precedence as ordinary generate. Code is
    # literal completion output: no chat template or reasoning-marker splitting.
    prepared = prepare_completion("", request.options, resolved=lease.resolved,
                                  metadata_info=metadata, format_value=request.format)
    body = prepared.body()
    body.update(prompt=tokens, preserved_tokens=list(pieces))
    body["stop"] = list(dict.fromkeys([*body.get("stop", []), *pieces]))
    return replace(prepared, payload=body)
