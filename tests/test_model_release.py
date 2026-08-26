"""Unloading a model must not free memory a request is still using.

The weights are in memory llama.cpp allocated, and they are freed when the
last reference to the Llama goes. A request generating on the model holds
one, so an unload asked for mid-generation waits for it to finish.

That deferral was once mistaken for a defect and replaced with an explicit
close() on unload. It segfaulted the server: ollama stop is a /api/generate
with keep_alive 0, the unload it triggers runs outside the generation slot,
and freeing there destroyed the llama_context under a request that was still
decoding. These tests exist so the deferral is not mistaken for a defect
again.
"""


class RecordingLlama:
    def __init__(self):
        self.closed = 0

    def n_ctx(self):
        return 4096

    def close(self):
        self.closed += 1


def register(manager, name, llm):
    manager.models[name] = llm
    manager.active_models[name] = {"id": name, "model": name}


def test_unloading_forgets_the_model(manager):
    register(manager, "alpha", RecordingLlama())

    manager.unload_model("alpha")

    assert "alpha" not in manager.models
    assert "alpha" not in manager.active_models


def test_unloading_does_not_free_a_model_someone_still_holds(manager):
    """The crash this prevents: a running generation loses its context."""
    llm = RecordingLlama()
    register(manager, "alpha", llm)
    still_generating = llm

    manager.unload_model("alpha")

    assert still_generating.closed == 0


def test_unloading_everything_frees_nothing_early_either(manager):
    models = {name: RecordingLlama() for name in ("alpha", "beta", "gamma")}
    for name, llm in models.items():
        register(manager, name, llm)

    manager.unload_all_models()

    assert manager.models == {}
    assert all(llm.closed == 0 for llm in models.values())


def test_unloading_an_absent_model_is_harmless(manager):
    manager.unload_model("no-such-model")

    assert manager.models == {}


def test_a_reloaded_model_replaces_the_previous_entry(manager):
    first = RecordingLlama()
    register(manager, "alpha", first)

    manager.unload_model("alpha")
    second = RecordingLlama()
    register(manager, "alpha", second)

    assert manager.models["alpha"] is second
