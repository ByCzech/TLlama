"""[mmproj]: checking that a named projector is one, and pointing out a
projector that is sitting there unnamed.

Both exist to move a failure earlier. A [mmproj] path that points at an
ordinary model, or at nothing, produces a virtual model that lists and
loads perfectly and then cannot see -- discovered whenever somebody
eventually sends an image. Saying so while the person still has the file
they just edited open is worth a header read.

Vision itself is not implemented yet. Nothing here loads a projector; it
only makes sure that what a definition names could be loaded when the
time comes.
"""

import gguf
import pytest

from tllama.helpers.model_toml import TomlModelError, parse_model_toml, render_model_toml


@pytest.fixture
def place_gguf(manager):
    def factory(relative_path, architecture="llama", **keys):
        path = manager.models_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = gguf.GGUFWriter(path=str(path), arch=architecture)
        for key, value in keys.items():
            writer.add_string(key.replace("__", "."), str(value))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        return path

    return factory


@pytest.fixture
def define(manager, toml_file):
    def factory(name, body):
        return toml_file(f"Local/{name}.toml", body)

    return factory


WITH_MMPROJ = '''[llm]
model = "Local/model.gguf"

[mmproj]
model = "{projector}"
'''


class TestAProjectorThatChecksOut:
    def test_the_model_lists_normally(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        place_gguf("Local/mmproj.gguf", architecture="clip")
        define("model", WITH_MMPROJ.format(projector="Local/mmproj.gguf"))

        assert [m["id"] for m in manager._list_local_models_sync()] == ["model"]

    def test_the_projector_resolves_to_its_file(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        projector = place_gguf("Local/mmproj.gguf", architecture="clip")
        toml = define("model", WITH_MMPROJ.format(projector="Local/mmproj.gguf"))

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        assert manager.resolve_mmproj_path(spec, "test") == projector

    def test_the_projector_itself_is_still_not_a_model(self, manager, place_gguf, define):
        """Named by a definition, it is a part of one. It does not become
        listable by being referred to."""
        place_gguf("Local/model.gguf")
        place_gguf("Local/mmproj.gguf", architecture="clip")
        define("model", WITH_MMPROJ.format(projector="Local/mmproj.gguf"))

        assert "mmproj" not in [m["id"] for m in manager._list_local_models_sync()]


class TestAProjectorThatDoesNot:
    def test_pointing_at_an_ordinary_model_is_rejected(self, manager, place_gguf, define):
        """The mistake this catches: a path that is plausible, exists, and
        is the wrong file."""
        place_gguf("Local/model.gguf")
        place_gguf("Local/other.gguf", architecture="qwen3")
        toml = define("model", WITH_MMPROJ.format(projector="Local/other.gguf"))

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        with pytest.raises(TomlModelError, match="not a projector"):
            manager.resolve_mmproj_path(spec, "test")

    def test_the_message_says_what_it_found_instead(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        place_gguf("Local/other.gguf", architecture="qwen3")
        toml = define("model", WITH_MMPROJ.format(projector="Local/other.gguf"))

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        with pytest.raises(TomlModelError, match="qwen3"):
            manager.resolve_mmproj_path(spec, "test")

    def test_pointing_at_nothing_is_rejected(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        toml = define("model", WITH_MMPROJ.format(projector="Local/absent.gguf"))

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        with pytest.raises(TomlModelError, match="existing file"):
            manager.resolve_mmproj_path(spec, "test")

    def test_escaping_the_repository_is_rejected(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        toml = define("model", WITH_MMPROJ.format(projector="../../../etc/passwd"))

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        with pytest.raises(TomlModelError, match="outside"):
            manager.resolve_mmproj_path(spec, "test")

    def test_such_a_model_is_dropped_from_the_listing(self, manager, place_gguf, define):
        """Same treatment a broken [llm] gets. A definition naming a
        projector it cannot use is not usable, and hiding that until
        somebody sends an image would be worse than saying it now."""
        place_gguf("Local/model.gguf")
        place_gguf("Local/other.gguf", architecture="qwen3")
        define("model", WITH_MMPROJ.format(projector="Local/other.gguf"))

        assert manager._list_local_models_sync() == []

    def test_asking_for_it_by_name_says_why(self, manager, place_gguf, define):
        """A name asked for by hand is one specific request, so it gets a
        reason rather than being told the model does not exist."""
        place_gguf("Local/model.gguf")
        place_gguf("Local/other.gguf", architecture="qwen3")
        define("model", WITH_MMPROJ.format(projector="Local/other.gguf"))

        with pytest.raises(TomlModelError):
            manager._build_model_file_info("model")


class TestNoMmprojAtAll:
    def test_a_plain_model_is_unaffected(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        define("model", '[llm]\nmodel = "Local/model.gguf"\n')

        assert [m["id"] for m in manager._list_local_models_sync()] == ["model"]

    def test_resolving_returns_nothing(self, manager, place_gguf, define):
        place_gguf("Local/model.gguf")
        toml = define("model", '[llm]\nmodel = "Local/model.gguf"\n')

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        assert manager.resolve_mmproj_path(spec, "test") is None


class TestSuggestingOne:
    def test_a_projector_beside_a_model_is_pointed_out(self, manager, place_gguf):
        """Named, not guessed: this is a file the scan actually found."""
        place_gguf("HuggingFace/ns/repo/model.gguf")
        place_gguf("HuggingFace/ns/repo/mmproj-model.gguf", architecture="clip")

        manager.rebuild_repository()

        text = (manager.hf_models_dir / "ns" / "repo" / "model.toml").read_text()
        assert "# model = \"HuggingFace/ns/repo/mmproj-model.gguf\"" in text

    def test_the_suggestion_is_inert_until_uncommented(self, manager, place_gguf):
        place_gguf("HuggingFace/ns/repo/model.gguf")
        place_gguf("HuggingFace/ns/repo/mmproj-model.gguf", architecture="clip")

        manager.rebuild_repository()

        toml = manager.hf_models_dir / "ns" / "repo" / "model.toml"
        spec = parse_model_toml(toml.read_text(encoding="utf-8"))

        assert spec.mmproj_model is None

    def test_uncommenting_it_produces_a_working_definition(self, manager, place_gguf):
        """A suggestion nobody can act on is decoration, so the two lines
        have to be valid exactly where they sit."""
        place_gguf("HuggingFace/ns/repo/model.gguf")
        place_gguf("HuggingFace/ns/repo/mmproj-model.gguf", architecture="clip")
        manager.rebuild_repository()

        toml = manager.hf_models_dir / "ns" / "repo" / "model.toml"
        toml.write_text(
            toml.read_text().replace("# [mmproj]", "[mmproj]").replace(
                '# model = "HuggingFace', 'model = "HuggingFace'
            ),
            encoding="utf-8",
        )

        spec = parse_model_toml(toml.read_text(encoding="utf-8"))
        assert spec.mmproj_model == "HuggingFace/ns/repo/mmproj-model.gguf"
        assert manager.resolve_mmproj_path(spec, "test") is not None

    def test_a_projector_elsewhere_is_not_offered(self, manager, place_gguf):
        """It was converted alongside a different model. Offering it would
        be noise a person has to recognise and dismiss."""
        place_gguf("HuggingFace/ns/repo/model.gguf")
        place_gguf("HuggingFace/other/repo/mmproj.gguf", architecture="clip")

        manager.rebuild_repository()

        text = (manager.hf_models_dir / "ns" / "repo" / "model.toml").read_text()
        assert "mmproj" not in text

    def test_two_projectors_are_a_question_not_a_suggestion(self, manager, place_gguf):
        place_gguf("HuggingFace/ns/repo/model.gguf")
        place_gguf("HuggingFace/ns/repo/mmproj-a.gguf", architecture="clip")
        place_gguf("HuggingFace/ns/repo/mmproj-b.gguf", architecture="clip")

        manager.rebuild_repository()

        text = (manager.hf_models_dir / "ns" / "repo" / "model.toml").read_text()
        assert "[mmproj]" not in text

    def test_nothing_is_offered_when_there_is_no_projector(self, manager, place_gguf):
        place_gguf("Local/model.gguf")

        manager.rebuild_repository()

        assert "mmproj" not in (manager.local_models_dir / "model.toml").read_text()


class TestTheRendererDirectly:
    def test_an_active_mmproj_is_still_active(self):
        text = render_model_toml("Local/m.gguf", mmproj_path="Local/p.gguf")

        assert parse_model_toml(text).mmproj_model == "Local/p.gguf"

    def test_a_suggestion_never_becomes_active(self):
        text = render_model_toml("Local/m.gguf", suggested_mmproj="Local/p.gguf")

        assert parse_model_toml(text).mmproj_model is None

    def test_an_explicit_projector_wins_over_a_suggestion(self):
        """Someone decided; there is nothing left to suggest."""
        text = render_model_toml(
            "Local/m.gguf", mmproj_path="Local/chosen.gguf", suggested_mmproj="Local/other.gguf"
        )

        assert parse_model_toml(text).mmproj_model == "Local/chosen.gguf"
        assert "other.gguf" not in text
