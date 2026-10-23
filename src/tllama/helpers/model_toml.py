from __future__ import annotations

import os
import tempfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomlkit
import tomlkit.exceptions
from tllama.helpers.fim import FIM_ORDERS


__all__ = (
    "TomlModelError",
    "VirtualModelSpec",
    "parse_model_toml",
    "render_model_toml",
    "resolve_repo_relative_path",
    "write_model_toml",
)


class TomlModelError(ValueError):
    """A .toml virtual-model definition is malformed or inconsistent.

    Raised for both syntax errors (invalid TOML) and semantic ones (e.g. an
    [llm] section with neither `model` nor `from`) so callers have one
    exception type to catch regardless of which stage rejected the file.
    """


def validate_definition_name(name):
    if (not isinstance(name, str) or not name or name != name.strip() or '\\' in name or '\0' in name
            or any(part in {'', '.', '..'} for part in name.split('/'))):
        raise ValueError('Invalid model name.')
    if len(name.split('/')) > 3:
        raise ValueError('Model names support one to three path components.')


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
    mmproj_context: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    stop: List[str] = field(default_factory=list)
    template: Optional[str] = None
    system_prompt: Optional[str] = None
    document: Any = field(default=None, repr=False, compare=False)
    fim_order: str = "auto"


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

    _reject_unknown_sections(document, source)

    llm = document.get("llm")
    if llm is None:
        raise TomlModelError(f"{source}: missing required [llm] section.")

    _reject_unknown_keys(llm, "llm", _LLM_KEYS, source)

    llm_model = _as_optional_str(llm.get("model"), field_name="llm.model", source=source)
    llm_from = _as_optional_str(llm.get("from"), field_name="llm.from", source=source)
    _require_xor("llm", llm_model, llm_from, source)

    mmproj = document.get("mmproj")
    mmproj_model: Optional[str] = None
    mmproj_from: Optional[str] = None
    mmproj_context: Dict[str, Any] = {}
    if mmproj is not None:
        _reject_unknown_mmproj_keys(mmproj, source)
        mmproj_context = _as_mmproj_context(mmproj, source)
        mmproj_model = _as_optional_str(mmproj.get("model"), field_name="mmproj.model", source=source)
        mmproj_from = _as_optional_str(mmproj.get("from"), field_name="mmproj.from", source=source)
        _require_xor("mmproj", mmproj_model, mmproj_from, source)

    runtime = dict(document.get("runtime") or {})
    _reject_unknown_runtime_keys(runtime, source)

    sampling_table = dict(document.get("sampling") or {})
    _reject_unknown_sampling_keys(sampling_table, source)

    if "logit_bias" in sampling_table:
        sampling_table["logit_bias"] = _as_logit_bias(
            sampling_table["logit_bias"], source
        )

    _reject_unusable_sampling_values(sampling_table, source)

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
        _reject_unknown_keys(template_table, "template", _TEMPLATE_KEYS, source)
        template = _as_optional_str(template_table.get("jinja"), field_name="template.jinja", source=source)

    system_table = document.get("system")
    system_prompt = None
    if system_table is not None:
        _reject_unknown_keys(system_table, "system", _SYSTEM_KEYS, source)
        system_prompt = _as_optional_str(system_table.get("prompt"), field_name="system.prompt", source=source)

    fim = document.get("fim", {})
    if not isinstance(fim, dict):
        raise TomlModelError(f"{source}: [fim] must be a table")
    _reject_unknown_keys(fim, "fim", {"order"}, source)
    fim_order = fim.get("order", "auto")
    if not isinstance(fim_order, str) or fim_order not in FIM_ORDERS:
        raise TomlModelError(f"{source}: [fim] order must be one of {', '.join(sorted(FIM_ORDERS))}")

    return VirtualModelSpec(
        llm_model=llm_model,
        llm_from=llm_from,
        mmproj_model=mmproj_model,
        mmproj_from=mmproj_from,
        mmproj_context=mmproj_context,
        runtime=runtime,
        sampling=sampling_table,
        stop=stop,
        template=template,
        system_prompt=system_prompt,
        fim_order=str(fim_order),
        document=document,
    )


# Every section parse_model_toml knows how to read. A name outside this
# set is refused rather than skipped, including one that is planned but
# not yet implemented: an unread [lora] means a model running without the
# adapter its file asked for, which is a wrong answer rather than a
# missing feature. The cost is that a .toml written for a later TLlama is
# refused by an earlier one, which is the intended outcome.
_KNOWN_SECTIONS = frozenset({
    "llm", "mmproj", "runtime", "sampling", "template", "system", "fim",
})

# The vocabularies parse_model_toml reads out of the three sections that
# had no check of their own. Each is short because each is asked for by
# name a few lines apart in the function above.
_LLM_KEYS = frozenset({"model", "from"})
_TEMPLATE_KEYS = frozenset({"jinja"})
_SYSTEM_KEYS = frozenset({"prompt"})


def _reject_unknown_sections(document: Any, source: str) -> None:
    """A section name nothing reads loses everything written under it.

    [mmproj], [runtime] and [sampling] each refuse a key they will not
    read. None of them can say anything about the name above it, so
    [smapling] parsed as valid TOML, was never looked at, and left
    sampling empty -- the file appearing to have set five values that
    reached nothing. That is the worse of the two mistakes: a misspelt
    key loses one setting, a misspelt section loses all of them at once,
    and neither the load nor the model shows any sign of it.

    Shape is checked here too, and for a sharper reason than tidiness.
    [[runtime]] is one bracket away from [runtime] and produced an array
    of tables, which dict() turned into a bare ValueError naming no file
    -- and get_model_metadata catches ValueError to mean "not a virtual
    model", so that typo discarded the entire .toml rather than being
    reported. A key at the top level with no section above it goes the
    same way: 'model = ...' written above [llm] belongs to nothing.
    """
    for name, value in document.items():
        if name not in _KNOWN_SECTIONS:
            raise TomlModelError(
                f"{source}: '{name}' is not a section TLlama reads. "
                f"The file takes {', '.join(f'[{s}]' for s in sorted(_KNOWN_SECTIONS))}."
            )

        if not isinstance(value, dict):
            raise TomlModelError(
                f"{source}: '{name}' must be a table written as [{name}], "
                f"not {type(value).__name__.lower()}."
            )


def _reject_unknown_keys(
    table: Any, section: str, allowed: frozenset, source: str
) -> None:
    """The same refusal the other sections make, for a fixed vocabulary.

    Each section uses TLlama's own vocabulary; these sections have no
    load/sampling-specific value interpretation.
    """
    for key in table:
        if key not in allowed:
            raise TomlModelError(
                f"{source}: [{section}] '{key}' is not a key TLlama reads. "
                f"The section takes {', '.join(sorted(allowed))}."
            )


# The whole vocabulary of [mmproj], which parse_model_toml reads by name.
_MMPROJ_PATH_KEYS = frozenset({"model", "from"})

# Fields of mtmd_context_params that _init_mtmd_context leaves at their
# defaults, so setting them here is the only way they can ever differ.
# The four upstream does set -- use_gpu, print_timings, n_threads and
# flash_attn_type -- are deliberately absent: it writes them after we do,
# so a value here would be overwritten and the file would lie.
_MMPROJ_CONTEXT_KEYS = {
    "warmup": bool,
    "image_min_tokens": int,
    "image_max_tokens": int,
    "batch_max_tokens": int,
}

_MMPROJ_KEYS = _MMPROJ_PATH_KEYS | frozenset(_MMPROJ_CONTEXT_KEYS)


def _reject_unknown_mmproj_keys(mmproj: Any, source: str) -> None:
    """An [mmproj] key nothing reads is a mistake, not a no-op.

    [runtime] and [sampling] already say so, each for its own reason.
    [mmproj] said nothing: parse_model_toml asks it for two names and
    never looks at the rest, so anything else was read from the file and
    dropped in silence, leaving a section that appeared to have been
    obeyed.

    Unlike the other two the vocabulary here is not derived from anything
    outside TLlama -- there is no foreign signature to read it off. It is
    the section's own, and it is written out because parse_model_toml
    asks for exactly these two by name a few lines up.
    """
    for key in mmproj:
        if key not in _MMPROJ_KEYS:
            raise TomlModelError(
                f"{source}: [mmproj] '{key}' is not a key TLlama reads. "
                f"The section takes {', '.join(sorted(_MMPROJ_KEYS))}."
            )


def _as_mmproj_context(mmproj: Any, source: str) -> Dict[str, Any]:
    """The mtmd context fields an [mmproj] asks to be set.

    Kept apart from the two path keys because they go somewhere else
    entirely: the paths select files, while these options become native
    projector arguments before the worker starts.

    Types are checked here so a failure names the configuration file.
    Bool is a subclass of int, so an unchecked 'image_max_tokens = true'
    would quietly mean 1.
    """
    context: Dict[str, Any] = {}

    for key, expected in _MMPROJ_CONTEXT_KEYS.items():
        if key not in mmproj:
            continue

        value = mmproj[key]
        if expected is bool:
            ok = isinstance(value, bool)
        else:
            ok = isinstance(value, int) and not isinstance(value, bool)

        if not ok:
            raise TomlModelError(
                f"{source}: [mmproj] '{key}' must be "
                f"{'a boolean' if expected is bool else 'an integer'}, "
                f"got {type(value).__name__}."
            )

        context[key] = expected(value)

    return context


def _reject_unknown_runtime_keys(runtime: Dict[str, Any], source: str) -> None:
    """Validate the pinned runtime contract without importing inference code."""
    from tllama.native_params import runtime_problem

    for key, value in runtime.items():
        problem = runtime_problem(key, value)
        if problem:
            raise TomlModelError(f"{source}: [runtime] {problem}.")


def _as_logit_bias(value: Any, source: str) -> Dict[int, float]:
    """Turn a [sampling] logit_bias table into what create_completion wants.

    Native token IDs are integers, but a TOML table's keys are always
    strings, so the token ids arrive as text and have to be converted here
    rather than at the call. Doing it at parse time also means a typo names
    the file it is in, which is not true of anything discovered later.

    A bias is deliberately unbounded: llama.cpp treats it as an addend on
    the logit and has no documented range, so a limit here would be one
    TLlama invented.
    """
    if not isinstance(value, dict):
        raise TomlModelError(
            f"{source}: [sampling] 'logit_bias' must be a table of "
            f"token id to bias, got {type(value).__name__}."
        )

    converted: Dict[int, float] = {}

    for token, bias in value.items():
        try:
            token_id = int(str(token), 10)
        except ValueError:
            raise TomlModelError(
                f"{source}: [sampling] 'logit_bias' key {token!r} is not a "
                "token id; keys must be integers."
            )

        # bool is an int subclass, so it would otherwise pass as a bias of
        # 1.0 or 0.0 and look deliberate.
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise TomlModelError(
                f"{source}: [sampling] 'logit_bias' value for token "
                f"{token_id} must be a number, got {type(bias).__name__}."
            )

        converted[token_id] = float(bias)

    return converted


def _reject_unusable_sampling_values(sampling: Dict[str, Any], source: str) -> None:
    """A value the key accepts but the sampler does not.

    Separate from _reject_unknown_sampling_keys: that one asks whether
    anything reads the name, this one whether the sampler can do what the
    value asks for. The judgement itself lives in helpers/common.py, next
    to the chain that applies these, so the environment layer reaches the
    same verdict on the same value.
    """
    from tllama.helpers.common import sampling_value_problem

    for name, value in sampling.items():
        problem = sampling_value_problem(name, value)
        if problem is not None:
            raise TomlModelError(f"{source}: [sampling] {problem}.")


def _reject_unknown_sampling_keys(sampling: Dict[str, Any], source: str) -> None:
    """Reject removed or unknown sampling names before interpreting values."""
    from tllama.native_params import SAMPLING_TYPES, sampling_problem

    for key in sampling:
        if key not in SAMPLING_TYPES:
            raise TomlModelError(f"{source}: [sampling] {sampling_problem(key, sampling[key])}.")


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
            "# [runtime] uses TLlama's supported native runtime settings.",
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
        newline="",
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

        if overwrite:
            os.replace(handle.name, path)
        else:
            # Unlike an exists/replace pair, link publishes without ever
            # clobbering a concurrent writer's newly created definition.
            os.link(handle.name, path)
            Path(handle.name).unlink()
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise

    return path
