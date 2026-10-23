"""Small immutable GGUF FIM inventory; never expose the whole vocabulary."""

from __future__ import annotations

FIM_ORDERS = frozenset({"auto", "psm", "training_spm", "native_spm", "disabled"})

# Recipes verified independently of model filenames and chat templates.
# Codestral's bracket tokens deliberately do not select a generic recipe.
_RECIPES = (
    (("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
     ("<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|file_separator|>")),
    (("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
     ("<fim_pad>", "<repo_name>", "<file_sep>")),
    (("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"), ()),
)


def read_fim_metadata(reader):
    """Resolve exact, unique vocabulary entries while scanning the GGUF header.

    Ordinary tokens are included: CodeGemma's file delimiter is NORMAL and
    cannot be recovered by tokenizing its spelling. Runtime piece verification
    is still required before using these IDs with a loaded worker.
    """
    import gguf

    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None or field.types != [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING]:
        return None
    wanted = {piece.encode("utf-8") for core, extra in _RECIPES for piece in (*core, *extra)}
    found, duplicates = {}, set()
    for token_id, part in enumerate(field.data):
        piece = field.parts[part].tobytes()
        if piece in wanted:
            if piece in found:
                duplicates.add(piece)
            found[piece] = token_id
    candidates = [(core, extra) for core, extra in _RECIPES
                  if all(piece.encode("utf-8") in found for piece in core)]
    if len(candidates) != 1:
        return None
    core, extra = candidates[0]
    pieces = [piece for piece in (*core, *extra) if piece.encode("utf-8") in found]
    if any(piece.encode("utf-8") in duplicates for piece in pieces):
        return None
    return {"prefix": core[0], "suffix": core[1], "middle": core[2],
            "tokens": {piece: found[piece.encode("utf-8")] for piece in pieces}}
