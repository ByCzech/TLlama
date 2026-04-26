from __future__ import annotations

import gguf
from typing import Any, Dict, Iterable, Mapping
from enum import IntEnum

try:
    import numpy as np
except Exception:  # pragma: no cover - gguf normally depends on numpy
    np = None


__all__ = (
    "read_gguf_metadata",
    "build_model_metadata_payload",
)


BASE_METADATA_KEYS = (
    "general.architecture",
    "general.name",
    "general.basename",
    "general.size_label",
    "general.parameter_count",
    "general.file_type",
    "general.quantization_version",
    "tokenizer.chat_template",
)


KNOWN_CONTEXT_LENGTH_KEYS = (
    "llama.context_length",
    "deepseek2.context_length",
    "qwen2.context_length",
    "qwen3.context_length",
    "qwen35moe.context_length",
    "gemma.context_length",
    "gemma3.context_length",
    "mistral.context_length",
    "gptoss.context_length",
)


def _as_int(value: Any, default: int = 0) -> int:
    """Convert a normalized metadata value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    """Convert a normalized metadata value to str."""
    if value is None:
        return default
    return str(value)


def _clean_llama_file_type_name(name: str) -> str:
    """Convert GGUF/Llama file type enum names to Ollama-style quantization labels."""
    for prefix in ("MOSTLY_", "ALL_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _enum_name_from_value(enum_cls: type[IntEnum], value: int) -> str | None:
    """Return enum member name for a numeric value if the value is known."""
    try:
        return enum_cls(value).name
    except Exception:
        return None


def _quantization_level_from_metadata(meta: Mapping[str, Any]) -> str:
    """
    Resolve a string quantization level suitable for Ollama-compatible details.

    GGUF stores the useful model file type in general.file_type. When possible,
    it is converted through gguf-py's enum to labels such as Q4_K_M or IQ3_S.
    """
    file_type = meta.get("general.file_type")
    file_type_int = None

    if file_type is not None:
        try:
            file_type_int = int(file_type)
        except (TypeError, ValueError):
            file_type_int = None

    if file_type_int is not None:
        for enum_name in ("LlamaFileType", "GGMLQuantizationType"):
            enum_cls = getattr(gguf, enum_name, None)
            if enum_cls is None:
                continue

            resolved_name = _enum_name_from_value(enum_cls, file_type_int)
            if resolved_name:
                return _clean_llama_file_type_name(resolved_name)

        return str(file_type_int)

    quantization_version = meta.get("general.quantization_version")
    if quantization_version is not None:
        return str(quantization_version)

    return "unknown"


def _decode_bytes(value: bytes) -> str:
    """Decode UTF-8 bytes from GGUF metadata in a tolerant way."""
    return value.decode("utf-8", errors="replace")


def _normalize_numpy_value(value: Any) -> Any:
    """Convert numpy values used by gguf-py into plain Python values."""
    if np is None:
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            return _decode_bytes(value.tobytes())

        if value.ndim == 0:
            return value.item()

        if value.size == 1:
            return value.reshape(-1)[0].item()

        return value.tolist()

    return value


def _normalize_value(value: Any) -> Any:
    """Convert GGUF metadata values into JSON-friendly plain Python values."""
    value = _normalize_numpy_value(value)

    if isinstance(value, bytes):
        return _decode_bytes(value)

    if isinstance(value, memoryview):
        return _decode_bytes(value.tobytes())

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        normalized = [_normalize_value(item) for item in value]

        if len(normalized) == 1:
            return normalized[0]

        return normalized

    return str(value)


def _iter_field_data_indexes(field: Any) -> Iterable[int]:
    """Yield indexes from a GGUF field data list."""
    data = getattr(field, "data", None)
    if data is None:
        return ()

    try:
        return tuple(int(index) for index in data)
    except TypeError:
        return ()


def _field_to_python_value(field: Any) -> Any:
    """Convert one GGUF field into a plain Python value."""
    parts = getattr(field, "parts", None)
    if parts is None:
        return None

    indexes = tuple(_iter_field_data_indexes(field))
    if not indexes:
        return None

    values = []
    for index in indexes:
        try:
            values.append(_normalize_value(parts[index]))
        except Exception:
            return None

    if len(values) == 1:
        return values[0]

    return values


def _read_selected_fields(reader: Any, keys: Iterable[str]) -> Dict[str, Any]:
    """Read selected GGUF metadata keys from an already-created reader."""
    metadata: Dict[str, Any] = {}

    for key in keys:
        field = reader.fields.get(key)
        if field is None:
            continue

        metadata[key] = _field_to_python_value(field)

    return metadata


def _build_metadata_key_set(reader: Any) -> tuple[str, ...]:
    """
    Build the small GGUF metadata key set needed by TLlama.

    The architecture-specific context length key is added dynamically because
    GGUF architectures do not all use the same prefix.
    """
    keys = set(BASE_METADATA_KEYS)
    keys.update(KNOWN_CONTEXT_LENGTH_KEYS)

    arch_field = reader.fields.get("general.architecture")
    if arch_field is not None:
        arch = _field_to_python_value(arch_field)
        if isinstance(arch, str) and arch:
            keys.add(f"{arch}.context_length")

    return tuple(sorted(keys))


def read_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """
    Read TLlama-relevant GGUF metadata without initializing llama.cpp inference.

    This function intentionally reads only a small whitelist of keys. It avoids
    returning large tokenizer metadata such as tokens, token types and merges.
    """
    reader = gguf.GGUFReader(model_path)
    keys = _build_metadata_key_set(reader)
    return _read_selected_fields(reader, keys)


def _first_present(meta: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first non-empty value found in metadata."""
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return default


def _context_length_from_metadata(meta: Mapping[str, Any], arch: str) -> Any:
    """Resolve context length from architecture-specific GGUF metadata."""
    keys = [f"{arch}.context_length"]
    keys.extend(KNOWN_CONTEXT_LENGTH_KEYS)
    return _first_present(meta, keys, 0)


def build_model_metadata_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build the normalized metadata payload used by TLlama routes and caches."""
    arch = _as_str(meta.get("general.architecture"), "unknown")
    template = _as_str(meta.get("tokenizer.chat_template"), "")

    params = _as_int(meta.get("general.parameter_count"), 0)
    bits = _quantization_level_from_metadata(meta)

    display_name = _as_str(
        meta.get("general.name")
        or meta.get("general.basename"),
        "",
    )

    context_length = _as_int(_context_length_from_metadata(meta, arch), 0)

    metadata_raw = {
        key: value
        for key, value in meta.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }

    return {
        "arch": arch,
        "params": params,
        "bits": bits,
        "template": template,
        "context_length": context_length,
        "display_name": display_name,
        "metadata_raw": metadata_raw,
    }
