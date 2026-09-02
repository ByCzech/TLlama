from __future__ import annotations

import os
import tempfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomlkit
import tomlkit.exceptions


__all__ = (
    "TomlModelError",
    "VirtualModelSpec",
    "parse_model_toml",
    "render_model_toml",
    "resolve_kv_cache_types",
    "resolve_repo_relative_path",
    "write_model_toml",
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
    _reject_unknown_runtime_keys(runtime, source)

    sampling_table = dict(document.get("sampling") or {})
    _reject_unknown_sampling_keys(sampling_table, source)

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


# type_kv is TLlama's shorthand for setting type_k and type_v together; it
# is not a Llama() parameter and so has to be named here.
_RUNTIME_EXTRA_KEYS = {"type_kv"}

# model_path is a Llama() parameter, but the path comes from [llm]. A
# [runtime] naming it would point somewhere else and the model would stop
# being the one the file defines, so it is refused rather than ignored.
_RUNTIME_FORBIDDEN_KEYS = {"model_path": "the model path comes from [llm]"}


def _reject_unknown_runtime_keys(runtime: Dict[str, Any], source: str) -> None:
    """A [runtime] key that is not a Llama() parameter is a mistake.

    It used to be passed through and swallowed: Llama() takes **kwargs and
    never looks at what it did not expect, so a misspelt key reached the
    library, was ignored, and left a setting that appeared to have been
    applied.

    The valid names are Llama()'s own, read from its signature rather than
    listed, so this cannot fall behind the library it validates against.
    """
    from tllama.helpers.runtime_params import llama_parameters

    valid = set(llama_parameters()) | _RUNTIME_EXTRA_KEYS

    for key in runtime:
        if key in _RUNTIME_FORBIDDEN_KEYS:
            raise TomlModelError(
                f"{source}: [runtime] '{key}' cannot be set here: "
                f"{_RUNTIME_FORBIDDEN_KEYS[key]}."
            )
        if key not in valid:
            raise TomlModelError(
                f"{source}: [runtime] '{key}' is not a parameter of Llama() "
                "in the installed llama-cpp-python."
            )


def _reject_unknown_sampling_keys(sampling: Dict[str, Any], source: str) -> None:
    """A [sampling] key nothing reads is a mistake, not a no-op.

    build_sampling_kwargs consults a fixed set of names; anything else in
    the table was read from the file, carried through the parse and then
    dropped without a word.

    The set here is what that function actually consumes. It is smaller
    than what llama-cpp-python's completion calls accept -- logit_bias and
    grammar among the absentees -- which is a separate gap, not something
    this should paper over by accepting names that would still go nowhere.
    """
    from tllama.helpers.common import sampling_parameter_names

    valid = sampling_parameter_names()

    for key in sampling:
        if key not in valid:
            raise TomlModelError(
                f"{source}: [sampling] '{key}' is not a sampling parameter "
                "TLlama applies."
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


def _toml_string(value: str) -> str:
    """Render a Python string as a TOML basic string.

    tomlkit is the parser everywhere in this module, but generating a new
    file is plain text assembly: the point of the output is its comments
    and their placement, which is easier to control directly than by
    building a document object and decorating it afterwards. Everything
    rendered here is parsed back before it is written, so the two cannot
    drift apart silently.
    """
    return tomlkit.item(str(value)).as_string()


def _commented(line: str) -> str:
    return f"# {line}"


def render_model_toml(
    model_path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    mmproj_path: Optional[str] = None,
    suggested_mmproj: Optional[str] = None,
) -> str:
    """Render a new virtual-model .toml naming an existing repo file.

    `model_path` and `mmproj_path` are repo-relative, including the
    category prefix ("HuggingFace/ns/repo/file.gguf") -- the same values
    parse_model_toml reads back out of [llm]/[mmproj].

    Only [llm] is active. Everything the GGUF itself has an opinion about
    is written commented out, so the file states plainly what the model
    came with and a person can act on it by deleting a '#' rather than by
    going and reading the header themselves. A commented line changes
    nothing on its own: with it commented, the value in effect stays
    whatever the request, the global configuration and TLlama's own
    baseline decide, exactly as if the line were not there at all.

    Nothing is invented. A key whose value is not in this GGUF is not
    written, not even as an empty placeholder, so the absence of a line
    means the model did not say -- never that this function had nothing to
    put there. The chat template is deliberately excluded despite being
    available: real ones run from 2 to 17 kB (measured across a working
    model store), which is not something to put in a configuration file
    that a person is meant to read.

    `metadata` is a build_model_metadata_payload() result. Without it the
    output is just the [llm] section, which is a complete and valid file.
    """
    metadata = metadata or {}

    lines: List[str] = [
        "# TLlama virtual model.",
        "#",
        "# The file is the model: its name here is the name the model is",
        "# known by, and a .gguf nothing points at is not listed at all.",
        "# Commented lines below are what this GGUF reports about itself;",
        "# uncomment one to pin it, and it stops depending on anything else.",
        "",
        "[llm]",
        f"model = {_toml_string(model_path)}",
    ]

    if mmproj_path is not None:
        lines += [
            "",
            "[mmproj]",
            f"model = {_toml_string(mmproj_path)}",
        ]
    elif suggested_mmproj is not None:
        # A real projector found in this model's own repository, named
        # rather than guessed at. Whether it belongs to this model is not
        # something the files say, so it is offered commented out for a
        # person to decide -- an empty placeholder would be inventing a
        # section, which nothing else here does.
        lines += [
            "",
            "# A projector sits in this repository. Uncomment both lines",
            "# below if it belongs to this model.",
            "# [mmproj]",
            _commented(f"model = {_toml_string(suggested_mmproj)}"),
        ]

    context_length = metadata.get("context_length") or 0
    if context_length:
        lines += [
            "",
            "# [runtime] passes through to Llama() by the names it uses.",
            "[runtime]",
            _commented(f"n_ctx = {int(context_length)}    # the model's own trained maximum"),
        ]

    recommended = metadata.get("recommended_sampling") or {}
    if recommended:
        lines += [
            "",
            "# What the model's author recorded in the GGUF as their",
            "# recommended sampling. Not applied unless uncommented.",
            "[sampling]",
        ]
        for key in ("temperature", "top_k", "top_p"):
            if key in recommended:
                lines.append(_commented(f"{key} = {_render_number(recommended[key])}"))

    text = "\n".join(lines) + "\n"

    # Parsed before it is returned, so a rendering mistake surfaces here
    # rather than as an unreadable file discovered later by a listing.
    parse_model_toml(text, source="<generated>")

    return text


def _render_number(value: Any) -> str:
    """Render a numeric value for a comment line.

    Floats arrive from a GGUF as float32 widened to float64, so a top_p an
    author wrote as 0.95 reads back as 0.949999988079071. Rounded here
    because this is a comment offered to a person to uncomment, and the
    artefact of the storage format is not information about the model.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def write_model_toml(path: Path, text: str, *, overwrite: bool = False) -> Path:
    """Write a .toml, atomically, without clobbering by accident.

    The rename is atomic within the directory, so a reader scanning the
    repository at the same moment sees either no file or the whole one --
    never a half-written file it would then report as malformed. The scan
    runs on every listing, so that overlap is ordinary rather than rare.

    overwrite=False by default: a .toml is a file a person edits, and
    silently replacing one that already exists would discard their work.
    Callers that mean to replace one have to say so.
    """
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")

    path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise

    return path
