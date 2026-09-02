"""An edited definition has to reach the model it defines.

A .toml is an ordinary file a person edits in place, and most of what it
says is consumed while Llama() is being constructed. Until this was wired
up, the only thing that could bring a model back down was a client sending
num_ctx explicitly with a different value -- which `ollama run` never does,
so an edited file was answered by the model built from the previous one for
as long as the model stayed resident.
"""

import os

import pytest


class FakeLlama:
    def __init__(self, n_ctx=4096):
        self._n_ctx = n_ctx

    def n_ctx(self):
        return self._n_ctx


@pytest.fixture
def loads(manager, gguf_file, monkeypatch):
    """Record what each load was actually given to build a model from."""
    calls = []

    def load(model_path, requested_n_ctx, virtual_spec=None):
        calls.append({"n_ctx": requested_n_ctx, "spec": virtual_spec})
        return FakeLlama(requested_n_ctx or 4096), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)
    gguf_file("Local/alpha.gguf")
    return calls


@pytest.fixture
def definition(models_root, toml_file):
    """Write alpha's .toml, with a distinct mtime every time."""
    written = {"stamp": 1_700_000_000}

    def write(text):
        path = toml_file("Local/alpha.toml", text)
        # Set explicitly rather than relying on the clock: two writes in the
        # same test are microseconds apart and the filesystem's timestamp
        # granularity is not something a test should be betting on.
        written["stamp"] += 60
        os.utime(path, ns=(written["stamp"] * 10**9, written["stamp"] * 10**9))
        return path

    return write


BASE = '[llm]\nmodel = "Local/alpha.gguf"\n'


async def test_an_untouched_definition_does_not_reload(manager, loads, definition):
    definition(BASE)

    first = await manager.get_model("alpha")
    second = await manager.get_model("alpha")

    assert second is first
    assert len(loads) == 1


async def test_an_edited_definition_brings_the_model_back_down(manager, loads, definition):
    definition(BASE)
    first = await manager.get_model("alpha")

    definition(BASE + "\n[runtime]\nn_ctx = 8192\n")
    second = await manager.get_model("alpha")

    assert second is not first
    assert len(loads) == 2


async def test_the_reloaded_model_is_built_from_the_new_definition(manager, loads, definition):
    """The point of reloading, checked against what the load was handed.

    Asserting only that a second load happened would pass just as well if
    it were handed the old file's contents.
    """
    definition(BASE + "\n[runtime]\nn_ctx = 2048\nflash_attn = false\n")
    await manager.get_model("alpha")

    definition(BASE + "\n[runtime]\nn_ctx = 8192\nflash_attn = true\n")
    await manager.get_model("alpha")

    assert loads[0]["spec"].runtime["flash_attn"] is False
    assert loads[1]["spec"].runtime["flash_attn"] is True
    assert (loads[0]["n_ctx"], loads[1]["n_ctx"]) == (2048, 8192)


async def test_touching_the_definition_is_enough(manager, loads, definition):
    """`touch model.toml` as a way to force a reload, deliberately kept.

    The fingerprint is the file's revision, not its contents: a definition
    whose bytes did not change still counts as changed. Hashing instead
    would take this away.
    """
    definition(BASE)
    first = await manager.get_model("alpha")

    definition(BASE)
    second = await manager.get_model("alpha")

    assert second is not first
    assert len(loads) == 2


async def test_an_explicit_num_ctx_still_reloads(manager, loads, definition):
    """The behaviour that already existed, kept alongside the new one."""
    definition(BASE)
    first = await manager.get_model("alpha", num_ctx=4096)

    second = await manager.get_model("alpha", num_ctx=8192)

    assert second is not first
    assert (loads[0]["n_ctx"], loads[1]["n_ctx"]) == (4096, 8192)


async def test_a_definition_edited_mid_load_is_noticed_afterwards(manager, gguf_file, definition, monkeypatch):
    """A file edited while the load runs did not shape what came out of it.

    Recording the revision seen after the load would mark the resident
    model as current when it was built from something else, and the edit
    would never be picked up.
    """
    gguf_file("Local/alpha.gguf")
    definition(BASE)

    loads = []

    def load(model_path, requested_n_ctx, virtual_spec=None):
        loads.append(requested_n_ctx)
        definition(BASE + "\n[runtime]\nn_ctx = 8192\n")
        return FakeLlama(requested_n_ctx or 4096), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)

    first = await manager.get_model("alpha")
    second = await manager.get_model("alpha")

    assert second is not first
    assert len(loads) == 2
