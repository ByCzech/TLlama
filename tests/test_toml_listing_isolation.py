"""A listing describes many models. One .toml going bad must cost that one
model, not the answer.

The scan (_list_local_models_sync) already worked this way. The enrichment
step that runs after it did not, and it re-reads every .toml from disk --
so a file edited between the two reads is parsed twice with different
results, and the second one took /api/tags down with it.

Reproducing that needs the edit to land in the window between the two
reads, which is what the racing fixture below arranges deliberately. A
test that merely leaves a broken .toml lying around proves nothing here:
the scan drops such a file before enrichment is ever reached, so it passes
whether the isolation exists or not.

/api/ps deliberately behaves the opposite way and keeps the model.
"""

import pytest

from tllama.backend import model_manager


BROKEN_TOML = '[llm]\nmodel = "Local/racing.gguf"\nthis line is not valid toml\n'

HEALTHY_TOML = '[llm]\nmodel = "Local/{name}.gguf"\n'


def _place_model(name: str):
    """A .toml/.gguf pair the scan will accept."""
    gguf = model_manager.local_models_dir / f"{name}.gguf"
    toml = model_manager.local_models_dir / f"{name}.toml"
    toml.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"not-a-real-gguf")
    toml.write_text(HEALTHY_TOML.format(name=name), encoding="utf-8")
    return gguf, toml


@pytest.fixture
def two_healthy_models():
    made = [_place_model("racing"), _place_model("bystander")]
    try:
        yield
    finally:
        for gguf, toml in made:
            gguf.unlink(missing_ok=True)
            toml.unlink(missing_ok=True)
        model_manager._invalidate_metadata_cache_entry("racing")
        model_manager._invalidate_metadata_cache_entry("bystander")


@pytest.fixture
def broken_between_the_two_reads(monkeypatch, two_healthy_models):
    """Break one .toml in the window the isolation exists for.

    list_local_models() is the scan; whatever runs after it is the
    enrichment. Corrupting the file as the scan returns puts the edit
    exactly where a person's editor would have put it, without depending on
    timing.
    """
    original = model_manager.list_local_models

    async def scan_then_corrupt():
        models = await original()
        (model_manager.local_models_dir / "racing.toml").write_text(
            BROKEN_TOML, encoding="utf-8"
        )
        return models

    monkeypatch.setattr(model_manager, "list_local_models", scan_then_corrupt)
    return "racing"


class TestTags:
    def test_the_listing_still_answers(self, client, broken_between_the_two_reads):
        response = client.get("/api/tags")

        assert response.status_code == 200

    def test_the_broken_model_is_dropped(self, client, broken_between_the_two_reads):
        response = client.get("/api/tags")
        names = {m["name"] for m in response.json()["models"]}

        assert broken_between_the_two_reads not in names

    def test_the_other_models_survive_it(self, client, broken_between_the_two_reads):
        """The whole point: the bystander was listed by the same scan and
        has nothing wrong with it."""
        response = client.get("/api/tags")
        names = {m["name"] for m in response.json()["models"]}

        assert "bystander" in names


@pytest.fixture
def resident_model_with_a_broken_toml():
    """A model already loaded when its .toml turned unparseable.

    Registered directly rather than loaded for real: /api/ps reports from
    active_models, and no inference is involved in what is being tested.
    """
    gguf, toml = _place_model("resident")
    toml.write_text(BROKEN_TOML, encoding="utf-8")

    model_manager.active_models["resident"] = {
        "id": "resident",
        "model": "resident",
        "filename": gguf.name,
        "path": str(gguf),
        "size": gguf.stat().st_size,
        "mtime": int(gguf.stat().st_mtime),
        "sha256": "",
        "n_ctx": 4096,
        "expires_at": None,
        "processor": "100% CPU",
    }
    try:
        yield "resident"
    finally:
        model_manager.active_models.pop("resident", None)
        gguf.unlink(missing_ok=True)
        toml.unlink(missing_ok=True)
        model_manager._invalidate_metadata_cache_entry("resident")


class TestPs:
    def test_ps_still_answers(self, client, resident_model_with_a_broken_toml):
        response = client.get("/api/ps")

        assert response.status_code == 200

    def test_a_resident_model_is_kept_not_dropped(
        self, client, resident_model_with_a_broken_toml
    ):
        """It is still loaded and still holding memory whatever state its
        .toml is in. Dropping it would describe the machine less accurately
        than listing it."""
        response = client.get("/api/ps")
        names = {m["name"] for m in response.json()["models"]}

        assert resident_model_with_a_broken_toml in names

    def test_its_metadata_degrades_to_unknown(
        self, client, resident_model_with_a_broken_toml
    ):
        response = client.get("/api/ps")
        entry = next(
            m for m in response.json()["models"]
            if m["name"] == resident_model_with_a_broken_toml
        )

        assert entry["details"]["family"] == "unknown"
