"""/api/show says whether a model can see.

Real Ollama's capability list is acted on directly by clients: the ollama
Python client exposes it, and a bot deciding whether to attach a picture
reads it rather than trying and seeing. Leaving "vision" off a model that
has a working projector tells that client the wrong thing.

Answered from the definition rather than from a loaded model. A client
asking what a model can do has not loaded it, and should not have to.
"""

import gguf
import pytest

from tllama.helpers.model_toml import parse_model_toml


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


WITH_PROJECTOR = """[llm]
model = "Local/model.gguf"

[mmproj]
model = "Local/mmproj.gguf"
"""

WITHOUT_PROJECTOR = """[llm]
model = "Local/model.gguf"
"""

PROJECTOR_NOT_IMPORTED = """[llm]
model = "Local/model.gguf"

[mmproj]
from = "/mimo/repo/mmproj.gguf"
"""


class TestWhatTheMetadataSays:
    async def test_a_working_projector_is_reported(
        self, manager, place_gguf, define
    ):
        place_gguf("Local/model.gguf")
        place_gguf("Local/mmproj.gguf", architecture="clip")
        define("model", WITH_PROJECTOR)

        metadata = await manager.get_model_metadata("model")

        assert metadata["has_projector"] is True

    async def test_no_projector_is_reported_as_such(
        self, manager, place_gguf, define
    ):
        place_gguf("Local/model.gguf")
        define("model", WITHOUT_PROJECTOR)

        metadata = await manager.get_model_metadata("model")

        assert metadata["has_projector"] is False

    async def test_an_unimported_projector_is_not_claimed_yet(
        self, manager, place_gguf, define
    ):
        # 'from' has nothing inside the repo behind it until an import
        # runs, so there is no projector to load and nothing to claim.
        place_gguf("Local/model.gguf")
        define("model", PROJECTOR_NOT_IMPORTED)

        metadata = await manager.get_model_metadata("model")

        assert metadata["has_projector"] is False


class TestWhatTheEndpointReports:
    """Through /api/show itself. Rebuilding the list in the test would
    only check that the test agrees with itself."""

    @pytest.fixture
    def show(self, monkeypatch):
        from fastapi.testclient import TestClient

        from tllama.routers import ollama as ollama_router
        from tllama.main import app

        def factory(metadata):
            async def fake_metadata(*args, **kwargs):
                return metadata

            monkeypatch.setattr(
                ollama_router.model_manager,
                "get_model_metadata",
                fake_metadata,
            )
            monkeypatch.setattr(
                ollama_router.model_manager,
                "read_model_definition",
                lambda *a, **k: "[llm]\n",
            )

            response = TestClient(app).post("/api/show", json={"model": "m"})
            return response.json()["capabilities"]

        return factory

    def test_vision_is_listed_for_a_model_with_a_projector(self, show):
        assert "vision" in show({"has_projector": True})

    def test_vision_is_not_listed_without_one(self, show):
        assert "vision" not in show({"has_projector": False})

    def test_vision_is_not_listed_when_nothing_says_either_way(self, show):
        # A model whose metadata carries no has_projector at all: not
        # claimed, rather than guessed at. (Empty metadata is not usable
        # here -- the endpoint reads that as the model not existing.)
        assert "vision" not in show({"arch": "llama"})

    def test_completion_still_comes_first(self, show):
        assert show({"has_projector": True})[0] == "completion"
