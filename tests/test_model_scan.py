"""Which files the model scan accepts, and the invariant behind it.

A reference has one segment for Local, two for TLlama and three or more for
HuggingFace. The scan has to agree with that, because anything it lists gets
a name, and a name that resolves elsewhere produces a model that appears in
/api/tags and then refuses to load.
"""

import pytest


USABLE_LAYOUTS = [
    "Local/model.gguf",
    "TLlama/ByCzech/model.gguf",
    "HuggingFace/unsloth/Repo-GGUF/model.gguf",
    "HuggingFace/unsloth/Repo-GGUF/UD-Q4_K_XL/model-00001-of-00005.gguf",
]

UNUSABLE_LAYOUTS = [
    "Local/nested/model.gguf",
    "Local/deeply/nested/model.gguf",
    "TLlama/model.gguf",
    "TLlama/ns/nested/model.gguf",
    "HuggingFace/model.gguf",
    "HuggingFace/unsloth/model.gguf",
]


def scanned(manager):
    return {str(path.relative_to(manager.models_dir)) for path in manager._iter_repository_model_files()}


@pytest.mark.parametrize("layout", USABLE_LAYOUTS)
def test_a_file_at_a_nameable_depth_is_listed(manager, gguf_file, layout):
    gguf_file(layout)

    assert scanned(manager) == {layout}


@pytest.mark.parametrize("layout", UNUSABLE_LAYOUTS)
def test_a_file_the_scheme_cannot_name_is_skipped(manager, gguf_file, layout):
    gguf_file(layout)

    assert scanned(manager) == set()


def test_everything_listed_resolves_back_to_its_own_path(manager, gguf_file):
    """The invariant that still holds: a listed path names consistently.

    _iter_repository_model_files() is the raw .gguf inventory (reserved for
    the future .toml migration tool, see TLlama_virtual_models_spec.md),
    no longer what /api/tags or get_model() use directly -- a name it
    yields is not expected to be loadable on its own anymore without a
    matching .toml (tests/test_virtual_model_scan.py covers that).
    """
    for layout in USABLE_LAYOUTS + UNUSABLE_LAYOUTS:
        gguf_file(layout)

    for path in manager._iter_repository_model_files():
        reference = manager._build_model_ref_from_path(path)

        assert manager.resolve_model_storage_path(reference) == path


def test_two_repositories_cannot_produce_the_same_name(manager, gguf_file):
    """At the permitted depths a Local and a TLlama file can never collide."""
    for layout in USABLE_LAYOUTS + UNUSABLE_LAYOUTS:
        gguf_file(layout)

    references = [
        manager._build_model_ref_from_path(path)
        for path in manager._iter_repository_model_files()
    ]

    assert len(references) == len(set(references))


def test_a_skipped_file_is_reported_once_not_on_every_listing(manager, gguf_file, caplog):
    """A listing runs on every ollama list, so the warning must not repeat."""
    gguf_file("Local/nested/model.gguf")

    with caplog.at_level("WARNING"):
        for _ in range(3):
            list(manager._iter_repository_model_files())

    assert len([r for r in caplog.records if "Ignoring" in r.getMessage()]) == 1


def test_a_non_gguf_file_is_ignored_without_complaint(manager, gguf_file, caplog):
    gguf_file("Local/notes.txt")
    gguf_file("HuggingFace/ns/repo/config.json")

    with caplog.at_level("WARNING"):
        assert scanned(manager) == set()

    assert caplog.records == []
