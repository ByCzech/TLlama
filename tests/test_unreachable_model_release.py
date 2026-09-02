"""A model nothing can ask for should not go on holding memory.

get_model() re-reads the .toml before it will hand back a resident model,
so a definition that will not parse, or that names something that is not
there, already fails every request for that model. What it did not do was
let go of the model, which stayed in memory -- unreachable and resident at
once -- until the process ended or a capacity limit evicted it.
"""

import pytest

from tllama.helpers.model_toml import TomlModelError


class FakeLlama:
    def __init__(self, n_ctx=4096):
        self._n_ctx = n_ctx

    def n_ctx(self):
        return self._n_ctx


@pytest.fixture
def resident(manager, gguf_file, toml_file, monkeypatch):
    """Get alpha loaded, and return the handle for editing its definition."""

    def load(model_path, requested_n_ctx, virtual_spec=None):
        return FakeLlama(requested_n_ctx or 4096), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)
    gguf_file("Local/alpha.gguf")
    path = toml_file("Local/alpha.toml", '[llm]\nmodel = "Local/alpha.gguf"\n')
    return path


async def test_a_definition_that_will_not_parse_frees_the_model(manager, resident):
    await manager.get_model("alpha")
    assert manager.is_model_loaded("alpha")

    resident.write_text("[llm\nmodel =", encoding="utf-8")

    with pytest.raises(TomlModelError):
        await manager.get_model("alpha")

    assert not manager.is_model_loaded("alpha")
    assert manager.get_loaded_model_info("alpha") is None


async def test_a_deleted_definition_frees_the_model(manager, resident):
    """The same event, arriving physically instead of logically.

    /api/delete unloads before it removes the .toml. Someone removing the
    file directly means the same thing and has to end the same way, or the
    two routes disagree for no reason anyone could explain.
    """
    await manager.get_model("alpha")

    resident.unlink()

    with pytest.raises(FileNotFoundError):
        await manager.get_model("alpha")

    assert not manager.is_model_loaded("alpha")


async def test_a_definition_pointing_at_nothing_frees_the_model(manager, resident, models_root):
    """The definition parses; what it names is gone."""
    await manager.get_model("alpha")

    (models_root / "Local/alpha.gguf").unlink()

    with pytest.raises(FileNotFoundError):
        await manager.get_model("alpha")

    assert not manager.is_model_loaded("alpha")


async def test_a_broken_definition_for_a_model_nobody_loaded_changes_nothing(
    manager, resident, gguf_file, toml_file
):
    """One model's broken definition must not disturb another.

    The unload is keyed on the name that was asked for, so a model that
    happens to be resident under a different name has nothing to do with
    it.
    """
    await manager.get_model("alpha")

    gguf_file("Local/beta.gguf")
    toml_file("Local/beta.toml", "[llm\nmodel =")

    with pytest.raises(TomlModelError):
        await manager.get_model("beta")

    assert manager.is_model_loaded("alpha")


async def test_asking_for_a_model_that_was_never_there_is_not_an_unload(manager, resident, monkeypatch):
    """Nothing resident means nothing to free, and no work to do for it.

    This path runs on every request naming a model that does not exist,
    including a plain typo, so it must not reach the unload machinery at
    all.
    """
    await manager.get_model("alpha")

    unloads = []
    monkeypatch.setattr(manager, "unload_model", lambda name: unloads.append(name))

    with pytest.raises(FileNotFoundError):
        await manager.get_model("does-not-exist")

    assert unloads == []
    assert manager.is_model_loaded("alpha")


async def test_a_repaired_definition_loads_again(manager, resident):
    """Freeing the model is not a tombstone: fixing the file is enough."""
    await manager.get_model("alpha")

    original = resident.read_text(encoding="utf-8")
    resident.write_text("[llm\nmodel =", encoding="utf-8")

    with pytest.raises(TomlModelError):
        await manager.get_model("alpha")

    resident.write_text(original, encoding="utf-8")

    assert await manager.get_model("alpha") is not None
    assert manager.is_model_loaded("alpha")
