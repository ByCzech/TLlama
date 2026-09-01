"""Deleting a model deletes its definition, and only that.

The .toml is the manifest and the .gguf is the blob. Real Ollama's `rm`
works the same way and for the same reason: it removes the manifest and
leaves the blob alone for as long as anything else refers to it.

Here the reason is sharper still. More than one virtual model naming one
physical file is not an accident to tolerate but a stated purpose of the
indirection -- sharing weights instead of copying gigabytes -- so removing
the weights along with one of the names would quietly break the others.
"""

import pytest


MODEL_TOML = '[llm]\nmodel = "{target}"\n'


@pytest.fixture
def defined_model(manager, gguf_file, toml_file):
    """A model with a definition and a file, ready to be deleted."""
    def factory(toml_rel, gguf_rel):
        gguf = gguf_file(gguf_rel)
        toml = toml_file(toml_rel, MODEL_TOML.format(target=gguf_rel))
        return toml, gguf

    return factory


class TestWhatIsRemoved:
    def test_the_definition_is_gone(self, manager, defined_model):
        toml, _ = defined_model("Local/model.toml", "Local/model.gguf")

        manager.delete_model_definition("model")

        assert not toml.exists()

    def test_the_model_stops_being_listed(self, manager, defined_model):
        defined_model("Local/model.toml", "Local/model.gguf")

        manager.delete_model_definition("model")

        assert manager._list_local_models_sync() == []

    def test_the_deleted_path_reported_is_the_definition(self, manager, defined_model):
        toml, _ = defined_model("Local/model.toml", "Local/model.gguf")

        result = manager.delete_model_definition("model")

        assert result["deleted_path"] == str(toml)


class TestWhatIsKept:
    def test_the_weights_stay_on_disk(self, manager, defined_model):
        _, gguf = defined_model("Local/model.toml", "Local/model.gguf")

        manager.delete_model_definition("model")

        assert gguf.is_file()

    def test_the_kept_file_is_reported(self, manager, defined_model):
        """A bare success after asking to delete a model reads as several
        gigabytes having gone away. Saying which file stayed costs one
        field and removes the guess."""
        defined_model("Local/model.toml", "Local/model.gguf")

        result = manager.delete_model_definition("model")

        assert result["kept_model_file"] == "Local/model.gguf"

    def test_another_model_sharing_the_file_still_works(self, manager, defined_model, toml_file):
        """The case the old behaviour broke: two definitions, one file,
        and deleting either one used to take the file with it."""
        defined_model("Local/first.toml", "Local/shared.gguf")
        toml_file("Local/second.toml", MODEL_TOML.format(target="Local/shared.gguf"))

        manager.delete_model_definition("first")

        assert [m["id"] for m in manager._list_local_models_sync()] == ["second"]

    def test_the_survivor_can_still_be_resolved(self, manager, defined_model, toml_file):
        defined_model("Local/first.toml", "Local/shared.gguf")
        toml_file("Local/second.toml", MODEL_TOML.format(target="Local/shared.gguf"))

        manager.delete_model_definition("first")

        assert manager._build_model_file_info("second") is not None


class TestAcrossRepositories:
    def test_a_tllama_model_is_deleted_by_its_two_segment_name(self, manager, defined_model):
        toml, _ = defined_model("TLlama/ByCzech/model.toml", "TLlama/ByCzech/model.gguf")

        manager.delete_model_definition("ByCzech/model")

        assert not toml.exists()

    def test_a_huggingface_model_is_deleted_by_its_three_segment_name(
        self, manager, defined_model
    ):
        toml, _ = defined_model(
            "HuggingFace/ns/repo/model.toml", "HuggingFace/ns/repo/Q4/model.gguf"
        )

        manager.delete_model_definition("ns/repo/model")

        assert not toml.exists()

    def test_a_deep_file_named_by_a_shallow_definition_survives(self, manager, defined_model):
        _, gguf = defined_model(
            "HuggingFace/ns/repo/model.toml", "HuggingFace/ns/repo/Q4/model.gguf"
        )

        manager.delete_model_definition("ns/repo/model")

        assert gguf.is_file()


class TestRefusals:
    def test_an_unknown_name_is_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.delete_model_definition("no-such-model")

    def test_a_name_too_deep_to_hold_a_model_is_rejected(self, manager):
        with pytest.raises(ValueError):
            manager.delete_model_definition("a/b/c/d")

    def test_a_bare_gguf_with_no_definition_is_not_deletable(self, manager, gguf_file):
        """Nothing points at it, so it is not a model and there is no name
        by which to ask for its removal. Deleting stray files is separate
        work with its own decisions."""
        path = gguf_file("Local/orphan.gguf")

        with pytest.raises(FileNotFoundError):
            manager.delete_model_definition("orphan")

        assert path.is_file()


class TestABrokenDefinition:
    def test_it_can_still_be_deleted(self, manager, toml_file):
        """A file that will not parse is exactly when someone most wants
        to be able to get rid of it."""
        toml = toml_file("Local/broken.toml", "this is not toml at all\n")

        manager.delete_model_definition("broken")

        assert not toml.exists()

    def test_no_kept_file_is_claimed_when_none_could_be_read(self, manager, toml_file):
        toml_file("Local/broken.toml", "this is not toml at all\n")

        result = manager.delete_model_definition("broken")

        assert result["kept_model_file"] is None


class TestTheEndpoint:
    def test_delete_reports_success(self, client):
        from tllama.backend import model_manager

        gguf = model_manager.local_models_dir / "endpoint.gguf"
        toml = model_manager.local_models_dir / "endpoint.toml"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"weights")
        toml.write_text(MODEL_TOML.format(target="Local/endpoint.gguf"), encoding="utf-8")

        try:
            response = client.request("DELETE", "/api/delete", json={"model": "endpoint"})

            assert response.status_code == 200
            assert not toml.exists()
            assert gguf.is_file()
        finally:
            gguf.unlink(missing_ok=True)
            toml.unlink(missing_ok=True)

    def test_deleting_something_absent_is_a_404(self, client):
        response = client.request("DELETE", "/api/delete", json={"model": "absent"})

        assert response.status_code == 404
