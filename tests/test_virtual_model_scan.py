"""Virtual model listing: .toml is the only source of truth (spec doc §3).

A .gguf without a matching .toml is not a model as far as /api/tags is
concerned -- it is just bytes on disk until some .toml names it. This is a
deliberate, strict policy (see TLlama_virtual_models_spec.md §3): it is what
makes multi-shard files, mmproj files, and deeply-nested HuggingFace quant
variants simply not show up as their own noisy entries, without needing any
special-case filtering for any of them.
"""

import pytest

from tllama.helpers.model_toml import parse_model_toml


def llm_toml(model_ref: str) -> str:
    return f'[llm]\nmodel = "{model_ref}"\n'


def listed_ids(manager):
    return {m["id"] for m in manager._list_local_models_sync()}


class TestBareGgufIsInvisible:
    """The core policy change: no .toml, no listing, regardless of depth."""

    def test_bare_gguf_at_a_normally_nameable_depth_is_not_listed(self, manager, gguf_file):
        gguf_file("Local/model.gguf")

        assert listed_ids(manager) == set()

    def test_a_multishard_file_without_its_own_toml_is_not_listed(self, manager, gguf_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model-00001-of-00003.gguf")
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model-00002-of-00003.gguf")

        assert listed_ids(manager) == set()

    def test_an_mmproj_file_without_its_own_toml_is_not_listed(self, manager, gguf_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/mmproj-model.gguf")

        assert listed_ids(manager) == set()


class TestFixedDepthPerCategory:
    """.toml depth is exact, not min/unbounded like raw .gguf depth."""

    def test_local_toml_at_depth_one_is_listed(self, manager, gguf_file, toml_file):
        gguf_file("Local/model.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/model.gguf"))

        assert listed_ids(manager) == {"MyModel"}

    def test_local_toml_nested_deeper_is_not_listed(self, manager, gguf_file, toml_file):
        gguf_file("Local/model.gguf")
        toml_file("Local/nested/MyModel.toml", llm_toml("Local/model.gguf"))

        assert listed_ids(manager) == set()

    def test_tllama_toml_at_depth_two_is_listed(self, manager, gguf_file, toml_file):
        gguf_file("TLlama/ByCzech/model.gguf")
        toml_file("TLlama/ByCzech/MyModel.toml", llm_toml("TLlama/ByCzech/model.gguf"))

        assert listed_ids(manager) == {"ByCzech/MyModel"}

    def test_tllama_toml_at_depth_one_is_not_listed(self, manager, gguf_file, toml_file):
        gguf_file("TLlama/model.gguf")
        toml_file("TLlama/MyModel.toml", llm_toml("TLlama/model.gguf"))

        assert listed_ids(manager) == set()

    def test_huggingface_toml_at_depth_three_is_listed(self, manager, gguf_file, toml_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model.gguf")
        toml_file(
            "HuggingFace/unsloth/Repo-GGUF/MyModel.toml",
            llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"),
        )

        assert listed_ids(manager) == {"unsloth/Repo-GGUF/MyModel"}

    def test_huggingface_toml_at_depth_four_is_not_listed(self, manager, gguf_file, toml_file):
        # Depth 3+ is fine for the raw .gguf itself, but a .toml is fixed at
        # exactly 3 -- deeper nesting of the underlying file is handled via
        # the model= path inside the .toml, not by nesting the .toml itself.
        gguf_file("HuggingFace/unsloth/Repo-GGUF/UD-Q4_K_XL/model.gguf")
        toml_file(
            "HuggingFace/unsloth/Repo-GGUF/UD-Q4_K_XL/MyModel.toml",
            llm_toml("HuggingFace/unsloth/Repo-GGUF/UD-Q4_K_XL/model.gguf"),
        )

        assert listed_ids(manager) == set()


class TestCrossReferences:
    """A .toml can point at a physical file anywhere in the repo (spec §5)."""

    def test_local_toml_can_reference_a_huggingface_file(self, manager, gguf_file, toml_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model.gguf")
        toml_file("Local/MyModel.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"))

        assert listed_ids(manager) == {"MyModel"}

    def test_two_virtual_models_can_share_one_physical_file(self, manager, gguf_file, toml_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model.gguf")
        toml_file("Local/Fast.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"))
        toml_file("Local/Quality.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"))

        assert listed_ids(manager) == {"Fast", "Quality"}

    def test_id_is_the_toml_name_not_the_underlying_filename(self, manager, gguf_file, toml_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/some-quant-file.gguf")
        toml_file("Local/FriendlyName.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/some-quant-file.gguf"))

        models = manager._list_local_models_sync()
        assert models[0]["id"] == "FriendlyName"
        assert models[0]["filename"] == "some-quant-file.gguf"

    def test_repository_reflects_the_toml_location_not_the_target(self, manager, gguf_file, toml_file):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model.gguf")
        toml_file("Local/MyModel.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"))

        models = manager._list_local_models_sync()
        assert models[0]["repository"] == "Local"


class TestBrokenTomlIsIsolated:
    """One bad .toml must not break the rest of the listing (spec §3)."""

    def test_unparseable_toml_is_skipped_not_fatal(self, manager, gguf_file, toml_file, caplog):
        gguf_file("Local/good.gguf")
        toml_file("Local/Good.toml", llm_toml("Local/good.gguf"))
        toml_file("Local/Bad.toml", "this is not [[[ valid toml")

        with caplog.at_level("WARNING"):
            assert listed_ids(manager) == {"Good"}

    def test_missing_llm_section_is_skipped_not_fatal(self, manager, gguf_file, toml_file):
        gguf_file("Local/good.gguf")
        toml_file("Local/Good.toml", llm_toml("Local/good.gguf"))
        toml_file("Local/Bad.toml", "[runtime]\nn_ctx = 8192\n")

        assert listed_ids(manager) == {"Good"}

    def test_model_pointing_nowhere_is_skipped_not_fatal(self, manager, gguf_file, toml_file):
        gguf_file("Local/good.gguf")
        toml_file("Local/Good.toml", llm_toml("Local/good.gguf"))
        toml_file("Local/Dangling.toml", llm_toml("Local/does-not-exist.gguf"))

        assert listed_ids(manager) == {"Good"}

    def test_path_escape_attempt_is_skipped_not_fatal(self, manager, gguf_file, toml_file):
        gguf_file("Local/good.gguf")
        toml_file("Local/Good.toml", llm_toml("Local/good.gguf"))
        toml_file("Local/Escape.toml", llm_toml("../../../../etc/passwd"))

        assert listed_ids(manager) == {"Good"}

    def test_unimported_from_is_skipped_not_fatal(self, manager, gguf_file, toml_file):
        gguf_file("Local/good.gguf")
        toml_file("Local/Good.toml", llm_toml("Local/good.gguf"))
        toml_file("Local/NotImportedYet.toml", '[llm]\nfrom = "/outside/repo/model.gguf"\n')

        assert listed_ids(manager) == {"Good"}


class TestIntegrationWithParser:
    def test_the_manager_uses_the_real_parser_not_a_reimplementation(self, manager, gguf_file, toml_file):
        # Sanity check that the manager's own listing agrees with
        # tllama.helpers.model_toml.parse_model_toml on a real file, rather
        # than reimplementing its own parsing.
        gguf_file("Local/model.gguf")
        text = llm_toml("Local/model.gguf")
        toml_file("Local/MyModel.toml", text)

        spec = parse_model_toml(text)
        models = manager._list_local_models_sync()

        assert models[0]["id"] == "MyModel"
        assert spec.llm_model == "Local/model.gguf"
