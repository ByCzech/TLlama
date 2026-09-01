"""A pulled .gguf gets a .toml, or it is not a model.

Under the strict policy (spec doc §3) a .gguf nothing points at is invisible
to every listing and cannot be loaded, so a pull that only fetches bytes
produces nothing usable. These cover what gets a definition, what
deliberately does not, and where the file lands.

Real GGUF files are written with gguf-py wherever the metadata matters, so
the projector and shard cases are decided by the same keys a real file
carries rather than by a dict saying so.
"""

import gguf
import pytest

from tllama.helpers.gguf_metadata import build_model_metadata_payload, read_gguf_metadata
from tllama.helpers.model_toml import parse_model_toml


@pytest.fixture
def place_gguf(manager):
    """Write a real GGUF at a path inside the model store."""
    def factory(relative_path, architecture="llama", **keys):
        path = manager.models_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = gguf.GGUFWriter(path=str(path), arch=architecture)
        for key, value in keys.items():
            key = key.replace("__", ".")
            if isinstance(value, bool):
                writer.add_bool(key, value)
            elif isinstance(value, int):
                writer.add_uint32(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            else:
                writer.add_string(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        return path

    return factory


@pytest.fixture
def define(manager, place_gguf):
    """Place a GGUF and run the .toml generator over it, as a pull does."""
    def factory(relative_path, **keys):
        path = place_gguf(relative_path, **keys)
        metadata = build_model_metadata_payload(read_gguf_metadata(str(path)))
        return path, manager._ensure_virtual_model_toml_sync(path, metadata)

    return factory


class TestWhereTheDefinitionLands:
    def test_a_huggingface_model_is_named_by_namespace_and_repo(self, manager, define):
        _, toml = define("HuggingFace/unsloth/Qwen3-GGUF/Qwen3-Q4_K_M.gguf")

        assert toml.relative_to(manager.models_dir).as_posix() == (
            "HuggingFace/unsloth/Qwen3-GGUF/Qwen3-Q4_K_M.toml"
        )

    def test_a_file_in_a_subdirectory_still_gets_a_three_segment_name(
        self, manager, define
    ):
        """Uploaders sort quantisations into subdirectories as an
        organisational habit with no protocol meaning. The .toml's depth is
        fixed regardless, so a model's name does not depend on how someone
        arranged their repository."""
        _, toml = define("HuggingFace/bartowski/Repo-GGUF/Q4_K_M/model.gguf")

        assert toml.relative_to(manager.models_dir).as_posix() == (
            "HuggingFace/bartowski/Repo-GGUF/model.toml"
        )

    def test_the_definition_points_at_the_deep_file_it_names(self, manager, define):
        """The indirection at work: the .toml is shallow, the .gguf is not,
        and the path in between is what connects them."""
        _, toml = define("HuggingFace/bartowski/Repo-GGUF/Q4_K_M/model.gguf")

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        assert spec.llm_model == "HuggingFace/bartowski/Repo-GGUF/Q4_K_M/model.gguf"

    def test_a_local_model_is_named_by_its_filename_alone(self, manager, define):
        _, toml = define("Local/MyModel-Q8_0.gguf")

        assert toml.relative_to(manager.models_dir).as_posix() == "Local/MyModel-Q8_0.toml"

    def test_a_tllama_model_keeps_its_namespace(self, manager, define):
        _, toml = define("TLlama/ByCzech/model.gguf")

        assert toml.relative_to(manager.models_dir).as_posix() == "TLlama/ByCzech/model.toml"

    def test_a_file_too_shallow_to_name_gets_nothing(self, manager, define):
        """A .gguf directly inside HuggingFace/ has no namespace and
        repository to take a name from."""
        _, toml = define("HuggingFace/loose.gguf")

        assert toml is None


class TestWhatIsNotAModel:
    def test_a_projector_gets_no_definition(self, manager, define):
        """It belongs to a model rather than being one. Listing it would
        offer the user something that cannot answer a prompt."""
        _, toml = define("HuggingFace/ns/repo/mmproj-model.gguf", architecture="clip")

        assert toml is None

    def test_a_projector_identified_only_by_general_type_is_also_skipped(
        self, manager, define
    ):
        _, toml = define("HuggingFace/ns/repo/mmproj.gguf", general__type="mmproj")

        assert toml is None

    def test_a_continuation_shard_gets_no_definition(self, manager, define):
        """llama.cpp loads the remaining parts from the first one, so a
        second shard is not a model anybody would ask for by name."""
        _, toml = define(
            "HuggingFace/ns/repo/model-00002-of-00003.gguf", split__no=1, split__count=3
        )

        assert toml is None

    def test_the_first_shard_does_get_one(self, manager, define):
        _, toml = define(
            "HuggingFace/ns/repo/model-00001-of-00003.gguf", split__no=0, split__count=3
        )

        assert toml is not None


class TestRepeatedPulls:
    def test_pulling_the_same_file_twice_leaves_one_definition(self, manager, define):
        """Re-pulling is ordinary -- a resumed download, a client that
        retries -- and must not accumulate duplicate definitions of the
        same model."""
        path, first = define("HuggingFace/ns/repo/model.gguf")
        second = manager._ensure_virtual_model_toml_sync(path, {})

        assert first is not None
        assert second is None
        assert len(list(first.parent.glob("*.toml"))) == 1

    def test_a_hand_edited_definition_is_not_overwritten(self, manager, define):
        path, toml = define("HuggingFace/ns/repo/model.gguf")
        toml.write_text(
            '# mine\n[llm]\nmodel = "HuggingFace/ns/repo/model.gguf"\n'
            '[runtime]\nn_ctx = 4096\n',
            encoding="utf-8",
        )

        manager._ensure_virtual_model_toml_sync(path, {})

        assert "# mine" in toml.read_text(encoding="utf-8")


class TestNameCollisions:
    def test_two_files_sharing_a_basename_get_distinct_names(self, manager, define):
        """Flattening to a fixed depth can put two files from different
        subdirectories on the same name. Rare, since quantisation is
        normally in the filename, but it must not silently replace one
        definition with the other."""
        _, first = define("HuggingFace/ns/repo/Q4_K_M/model.gguf")
        _, second = define("HuggingFace/ns/repo/Q8_0/model.gguf")

        assert first.name == "model.toml"
        assert second.name == "model_01.toml"

    def test_both_definitions_point_at_their_own_file(self, manager, define):
        _, first = define("HuggingFace/ns/repo/Q4_K_M/model.gguf")
        _, second = define("HuggingFace/ns/repo/Q8_0/model.gguf")

        assert parse_model_toml(first.read_text(encoding="utf-8")).llm_model.endswith(
            "Q4_K_M/model.gguf"
        )
        assert parse_model_toml(second.read_text(encoding="utf-8")).llm_model.endswith(
            "Q8_0/model.gguf"
        )

    def test_a_name_held_by_an_unreadable_file_is_still_taken(self, manager, define, place_gguf):
        """Repairing someone else's broken file is not this code's
        business; stepping around it is."""
        (manager.hf_models_dir / "ns" / "repo").mkdir(parents=True, exist_ok=True)
        (manager.hf_models_dir / "ns" / "repo" / "model.toml").write_text(
            "this is not toml at all\n", encoding="utf-8"
        )

        _, toml = define("HuggingFace/ns/repo/model.gguf")

        assert toml.name == "model_01.toml"


class TestTheModelIsThenVisible:
    async def test_a_defined_model_appears_in_the_listing(self, manager, define):
        """The end of the chain this exists for: bytes on disk, then a
        definition, then a model somebody can actually ask for."""
        define("HuggingFace/ns/repo/model.gguf")

        listed = await manager.list_local_models()

        assert [m["id"] for m in listed] == ["ns/repo/model"]

    async def test_an_undefined_file_stays_invisible(self, manager, place_gguf):
        place_gguf("HuggingFace/ns/repo/model.gguf")

        assert await manager.list_local_models() == []


class TestFailuresDoNotBreakThePull:
    def test_an_unwritable_directory_is_survived(self, manager, place_gguf, monkeypatch):
        """Several gigabytes arrived successfully. A small text file that
        could not be created afterwards is worth a warning, not a failed
        pull."""
        path = place_gguf("HuggingFace/ns/repo/model.gguf")

        def refuse(*args, **kwargs):
            raise PermissionError("read-only file system")

        monkeypatch.setattr("tllama.backend.write_model_toml", refuse)

        assert manager._ensure_virtual_model_toml_sync(path, {}) is None

    def test_a_file_outside_the_repository_is_declined(self, manager, tmp_path):
        stray = tmp_path / "elsewhere" / "model.gguf"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"x")

        assert manager._ensure_virtual_model_toml_sync(stray, {}) is None
