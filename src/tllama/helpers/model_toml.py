from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomlkit
import tomlkit.exceptions


__all__ = (
    "TomlModelError",
    "VirtualModelSpec",
    "parse_model_toml",
    "resolve_kv_cache_types",
    "resolve_repo_relative_path",
)


class TomlModelError(ValueError):
    """A .toml virtual-model definition is malformed or inconsistent.

    Raised for both syntax errors (invalid TOML) and semantic ones (e.g. an
    [llm] section with neither `model` nor `from`) so callers have one
    exception type to catch regardless of which stage rejected the file.
    """


@dataclass
class VirtualModelSpec:
    """The parsed contents of one virtual-model .toml file.

    One instance always corresponds to one .toml file (see spec doc §4: one
    virtual model per file). `document` keeps the original parsed
    tomlkit.TOMLDocument so a later caller can round-trip an edit (e.g.
    rewriting `from =` to `model =` after an import) without losing the
    comments and formatting a person wrote into the file -- see spec §2 for
    why that ruled out both configparser and tomli_w.
    """

    llm_model: Optional[str] = None
    llm_from: Optional[str] = None
    mmproj_model: Optional[str] = None
    mmproj_from: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    stop: List[str] = field(default_factory=list)
    template: Optional[str] = None
    system_prompt: Optional[str] = None
    document: Any = field(default=None, repr=False, compare=False)


def _require_xor(section: str, model_value: Any, from_value: Any, source: str) -> None:
    have_model = model_value is not None
    have_from = from_value is not None

    if have_model and have_from:
        raise TomlModelError(
            f"{source}: [{section}] has both 'model' and 'from' set; "
            "'from' is only for importing a file from outside the repo and "
            "is expected to be replaced by 'model' once that import runs, "
            "so having both at once is ambiguous."
        )
    if not have_model and not have_from:
        raise TomlModelError(
            f"{source}: [{section}] needs either 'model' (a path already "
            "inside the repo) or 'from' (a path to import)."
        )


def _as_optional_str(value: Any, *, field_name: str, source: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TomlModelError(
            f"{source}: '{field_name}' must be a string, got {type(value).__name__}."
        )
    return str(value)


def parse_model_toml(text: str, *, source: str = "<toml>") -> VirtualModelSpec:
    """Parse the contents of one virtual-model .toml file.

    Pure function: no filesystem access, no knowledge of models_dir, no
    llama_cpp import. `source` is only used to make error messages point at
    the right file; pass the actual path when you have one.
    """
    try:
        document = tomlkit.parse(text)
    except tomlkit.exceptions.ParseError as e:
        raise TomlModelError(f"{source}: invalid TOML: {e}") from e

    llm = document.get("llm")
    if llm is None:
        raise TomlModelError(f"{source}: missing required [llm] section.")

    llm_model = _as_optional_str(llm.get("model"), field_name="llm.model", source=source)
    llm_from = _as_optional_str(llm.get("from"), field_name="llm.from", source=source)
    _require_xor("llm", llm_model, llm_from, source)

    mmproj = document.get("mmproj")
    mmproj_model: Optional[str] = None
    mmproj_from: Optional[str] = None
    if mmproj is not None:
        mmproj_model = _as_optional_str(mmproj.get("model"), field_name="mmproj.model", source=source)
        mmproj_from = _as_optional_str(mmproj.get("from"), field_name="mmproj.from", source=source)
        _require_xor("mmproj", mmproj_model, mmproj_from, source)

    runtime = dict(document.get("runtime") or {})
    sampling_table = dict(document.get("sampling") or {})

    stop_value = sampling_table.pop("stop", [])
    if stop_value and not isinstance(stop_value, list):
        raise TomlModelError(f"{source}: [sampling] 'stop' must be an array of strings.")
    stop: List[str] = []
    for item in stop_value:
        if not isinstance(item, str):
            raise TomlModelError(
                f"{source}: [sampling] 'stop' entries must all be strings, "
                f"got {type(item).__name__}."
            )
        stop.append(str(item))

    template_table = document.get("template")
    template = None
    if template_table is not None:
        template = _as_optional_str(template_table.get("jinja"), field_name="template.jinja", source=source)

    system_table = document.get("system")
    system_prompt = None
    if system_table is not None:
        system_prompt = _as_optional_str(system_table.get("prompt"), field_name="system.prompt", source=source)

    return VirtualModelSpec(
        llm_model=llm_model,
        llm_from=llm_from,
        mmproj_model=mmproj_model,
        mmproj_from=mmproj_from,
        runtime=runtime,
        sampling=sampling_table,
        stop=stop,
        template=template,
        system_prompt=system_prompt,
        document=document,
    )


def resolve_kv_cache_types(runtime: Dict[str, Any]) -> "tuple[Optional[int], Optional[int]]":
    """Resolve type_k/type_v from a [runtime] table into ggml type ints.

    type_kv sets both; an explicit type_k/type_v takes priority over it for
    that side only (spec §8). A value can be either a name ("q4_0"),
    resolved via the same GGML_TYPE_* constants llama_cpp.llama_cpp already
    defines -- no hand-maintained table, new quant types just work once the
    vendored llama.cpp gains them -- or a raw int, used as-is.

    llama_cpp is imported lazily, matching the existing pattern elsewhere in
    this codebase (_ext.py, backend.py's HF pull path), so this module stays
    importable for standalone parsing/tests without llama_cpp installed.
    """
    type_kv = runtime.get("type_kv")
    type_k = runtime.get("type_k", type_kv)
    type_v = runtime.get("type_v", type_kv)

    return (
        _resolve_one_kv_cache_type(type_k, field_name="type_k"),
        _resolve_one_kv_cache_type(type_v, field_name="type_v"),
    )


def _resolve_one_kv_cache_type(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; reject before the int check
        raise TomlModelError(f"'{field_name}' must be a string or int, not a bool.")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        from llama_cpp import llama_cpp as llama_cpp_lib

        const_name = f"GGML_TYPE_{value.upper()}"
        resolved = getattr(llama_cpp_lib, const_name, None)
        if resolved is None:
            raise TomlModelError(
                f"'{field_name}' = \"{value}\" does not match any known ggml "
                f"type (looked for {const_name} in llama_cpp.llama_cpp)."
            )
        return int(resolved)
    raise TomlModelError(f"'{field_name}' must be a string or int, got {type(value).__name__}.")


def resolve_repo_relative_path(value: str, models_dir: Path) -> Path:
    """Resolve a .toml path value (e.g. "HuggingFace/ns/repo/model.gguf")
    against the repo root and confirm it does not escape it.

    Mirrors the existing Path.is_relative_to() idiom already used elsewhere
    in backend.py, so path-safety is checked the same way everywhere in the
    codebase. Raises TomlModelError (not the caller's problem to remember to
    check the return value) if the resolved path would land outside
    models_dir -- the "../../.." escape spec §5 calls out explicitly.
    """
    models_dir = models_dir.resolve()
    resolved = (models_dir / value).resolve()

    if not resolved.is_relative_to(models_dir):
        raise TomlModelError(
            f"'{value}' resolves outside the models repository ({models_dir})."
        )

    return resolved
