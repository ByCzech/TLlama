"""Unloading a model has to free its memory now, not eventually.

The weights are in memory llama.cpp allocated. Dropping the last Python
reference and calling the garbage collector reaches close() only when the
last reference actually goes, and a request still streaming holds one. That
turns an unload into a promise rather than an act, which on a machine
juggling multi-gigabyte models is the difference between working and running
out of memory.
"""

import pytest


class RecordingLlama:
    def __init__(self):
        self.closed = 0

    def n_ctx(self):
        return 4096

    def close(self):
        self.closed += 1


class UnclosableLlama(RecordingLlama):
    def close(self):
        raise RuntimeError("llama_model_free failed")


class OpaqueLlama:
    """An object with no close at all, as an older binding might hand back."""

    def n_ctx(self):
        return 4096


def register(manager, name, llm):
    manager.models[name] = llm
    manager.active_models[name] = {"id": name, "model": name}


def test_unloading_releases_the_model(manager):
    llm = RecordingLlama()
    register(manager, "alpha", llm)

    manager.unload_model("alpha")

    assert llm.closed == 1
    assert "alpha" not in manager.models


def test_a_model_still_referenced_elsewhere_is_released_anyway(manager):
    """A streaming request holds a reference; the unload must not wait for it."""
    llm = RecordingLlama()
    register(manager, "alpha", llm)
    still_in_use = llm

    manager.unload_model("alpha")

    assert still_in_use.closed == 1


def test_unloading_twice_does_not_release_twice(manager):
    llm = RecordingLlama()
    register(manager, "alpha", llm)

    manager.unload_model("alpha")
    manager.unload_model("alpha")

    assert llm.closed == 1


def test_a_failed_release_is_reported_rather_than_swallowed(manager, caplog):
    """Silence here would surface an hour later as an unexplained load failure."""
    register(manager, "alpha", UnclosableLlama())

    with caplog.at_level("WARNING"):
        manager.unload_model("alpha")

    assert any("may stay allocated" in record.getMessage() for record in caplog.records)


def test_a_failed_release_still_removes_the_model(manager):
    register(manager, "alpha", UnclosableLlama())

    manager.unload_model("alpha")

    assert "alpha" not in manager.models
    assert "alpha" not in manager.active_models


def test_a_model_without_close_is_left_alone(manager, caplog):
    with caplog.at_level("WARNING"):
        register(manager, "alpha", OpaqueLlama())
        manager.unload_model("alpha")

    assert "alpha" not in manager.models
    assert caplog.records == []


def test_unloading_everything_releases_everything(manager):
    models = {name: RecordingLlama() for name in ("alpha", "beta", "gamma")}
    for name, llm in models.items():
        register(manager, name, llm)

    manager.unload_all_models()

    assert all(llm.closed == 1 for llm in models.values())
    assert manager.models == {}
