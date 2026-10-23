"""A configuration field nothing reads is a setting that does nothing.

The environment layer already refuses a value it cannot use and a .toml
key nothing consumes, on the grounds that a setting which silently does
nothing is worse than one that fails loudly. A field on BackendConfig or
AppConfig that no other module ever reads is the same failure one level
up, and it is invisible from either side: the parsing is correct, the
documentation describing the variable is correct, and the wire between
them is missing.

The field names are taken from the dataclasses rather than listed here,
so a field added later is covered without a second edit.
"""

import re

from dataclasses import fields
from pathlib import Path

from tllama.config import AppConfig, BackendConfig


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

PACKAGE = REPOSITORY_ROOT / "src" / "tllama"

# Native integration documentation is explicitly deferred until the agreed
# integration and hardware checks are complete. TODO: Remove this finite exception
# list when publishing those docs; all other variables remain covered.
_DEFERRED_NATIVE_DOCS = {
    "TLLAMA_RUNTIME_EXECUTABLE", "TLLAMA_RUNTIME_VERSION_POLICY",
    "TLLAMA_MEMORY_SOURCE", "TLLAMA_MEMORY_COMPARE",
}


def _consumers(field_name: str) -> list:
    """Modules reading the field off a configuration object.

    Matching on `config.<name>` rather than on the bare name, because
    several of these are ordinary words -- `port` occurs inside `import`
    and `host` inside `hostname`, and a substring search would call every
    field consumed no matter what.

    config.py itself is excluded: it assigns these, which is the half
    already known to work.
    """
    pattern = re.compile(rf"config\.{field_name}\b")

    return [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name != "config.py"
        and pattern.search(path.read_text(encoding="utf-8"))
    ]


def test_every_backend_configuration_field_has_a_consumer():
    unread = [
        field.name for field in fields(BackendConfig) if not _consumers(field.name)
    ]

    assert not unread, (
        "BackendConfig fields parsed from the environment and read by "
        "nothing: " + ", ".join(unread)
    )


def test_every_app_configuration_field_has_a_consumer():
    unread = [field.name for field in fields(AppConfig) if not _consumers(field.name)]

    assert not unread, (
        "AppConfig fields parsed from the environment and read by "
        "nothing: " + ", ".join(unread)
    )


def test_the_check_can_tell_a_read_field_from_an_unread_one():
    """Both halves come from the same expression, so it has to be shown to
    separate them -- a pattern that matched everything would make the
    tests above pass no matter what the package contained."""
    assert _consumers("max_loaded_models")
    assert not _consumers("not_a_configuration_field")


def test_every_environment_variable_config_reads_is_documented():
    """The other direction: read by the code, described nowhere.

    This one is a documentation gap rather than a broken setting, and it
    is just as invisible from inside either half. TLLAMA_DEBUG had lost
    its heading in docs/configuration.md, leaving the sentence describing
    it attached to the section above and the name written nowhere at all,
    so it worked and could not be found.

    config.py is the only file that reads the environment, which is what
    makes the question answerable from one place.
    """
    source = (PACKAGE / "config.py").read_text(encoding="utf-8")
    documented = " ".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPOSITORY_ROOT / "docs").glob("*.md"))
    )

    undocumented = sorted(
        {
            name
            for name in re.findall(r"TLLAMA_[A-Z0-9_]+", source)
            if name not in documented and name not in _DEFERRED_NATIVE_DOCS
        }
    )

    assert not undocumented, (
        "read by config.py and described in no document: "
        + ", ".join(undocumented)
    )
