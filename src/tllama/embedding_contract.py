"""Pooled semantic output checks independent of a model name or worker process."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EmbeddingContract:
    pooling: int
    dimension: int
    vocab_size: int | None


def embedding_contract(metadata: Mapping, pooling: int = -1) -> EmbeddingContract:
    raw = metadata.get('metadata_raw') or {}
    arch = raw.get('general.architecture')
    if not isinstance(arch, str) or not arch:
        raise ValueError('Embedding output requires known GGUF architecture metadata')
    if raw.get(f'{arch}.decision.type'):
        raise ValueError('Decision-head scores are not semantic embedding vectors')
    if pooling == -1:
        pooling = raw.get(f'{arch}.pooling_type')
    if type(pooling) is not int or pooling not in (1, 2, 3):
        raise ValueError('Semantic embeddings require known mean, CLS or last pooling')
    # Native n_embd_out(): a positive output width overrides the hidden
    # width; absent/zero means use the hidden width. Never infer it from
    # a response or silently ignore a malformed explicit declaration.
    output_width = raw.get(f'{arch}.embedding_length_out', 0)
    if type(output_width) is not int or output_width < 0:
        raise ValueError('Embedding output requires a nonnegative integer GGUF embedding_length_out')
    dimension = output_width or raw.get(f'{arch}.embedding_length')
    if type(dimension) is not int or dimension <= 0:
        raise ValueError('Embedding output requires a known positive GGUF output embedding length')
    vocab_size = (metadata.get('embedding') or {}).get('vocab_size')
    if type(vocab_size) is not int or vocab_size <= 0:
        vocab_size = None
    return EmbeddingContract(pooling, dimension, vocab_size)


def embedding_capable(metadata: Mapping) -> bool:
    if metadata.get('embedding_configured') is not True or metadata.get('has_projector'):
        return False
    try:
        embedding_contract(metadata, metadata.get('embedding_pooling_type', -1))
    except ValueError:
        return False
    return True
