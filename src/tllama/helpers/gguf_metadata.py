from __future__ import annotations

from typing import Any, Dict

import gguf


__all__ = (
    "read_gguf_metadata",
    "build_model_metadata_payload",
)


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _field_to_python_value(field: Any) -> Any:
    """Convert a GGUF reader field into plain Python values."""
    parts = getattr(field, "parts", None)
    data = getattr(field, "data", None)

    if parts is None or data is None:
        return None

    try:
        indexes = list(data)
    except TypeError:
        return None

    values = []
    for index in indexes:
        try:
            values.append(_decode_bytes(parts[index]))
        except Exception:
            return None

    if len(values) == 1:
        return values[0]

    return values


def _normalize_metadata_value(value: Any) -> Any:
    value = _decode_bytes(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, list):
        normalized = []
        for item in value:
            normalized_item = _decode_bytes(item)
            if isinstance(normalized_item, (str, int, float, bool)) or normalized_item is None:
                normalized.append(normalized_item)
            else:
                normalized.append(str(normalized_item))
        return normalized

    return str(value)


def read_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """Read GGUF key-value metadata without initializing a llama.cpp model."""
    reader = gguf.GGUFReader(model_path)
    metadata: Dict[str, Any] = {}

    for key, field in reader.fields.items():
        metadata[str(key)] = _normalize_metadata_value(_field_to_python_value(field))

    return metadata


def build_model_metadata_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build the normalized metadata payload used by TLlama routes and caches."""
    arch = meta.get("general.architecture") or "unknown"
    template = meta.get("tokenizer.chat_template") or ""
    params = meta.get("general.parameter_count") or 0
    bits = meta.get("general.quantization_version") or "unknown"
    context_length = (
        meta.get("llama.context_length")
        or meta.get("deepseek2.context_length")
        or meta.get("qwen2.context_length")
        or meta.get("gemma.context_length")
        or 0
    )
    display_name = (
        meta.get("general.name")
        or meta.get("general.basename")
        or ""
    )

    return {
        "arch": arch,
        "params": params,
        "bits": bits,
        "template": template,
        "context_length": context_length,
        "display_name": display_name,
        "metadata_raw": meta,
    }
