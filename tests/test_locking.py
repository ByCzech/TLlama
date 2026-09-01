"""One lock per piece of state, and never two at once.

A single lock made a model load block everything: a listing, a metadata
read, even a request for a different model that was already resident. These
tests pin the separation, and the invariant that keeps it deadlock-free.
"""

import asyncio
import inspect
import re

import pytest

from tllama import backend as backend_module


LOCKS = ("_models_lock", "_metadata_lock", "_slots_lock")


def test_each_lock_guards_its_own_state(manager):
    for name in LOCKS:
        assert isinstance(getattr(manager, name), asyncio.Lock)


def test_no_path_holds_one_lock_while_taking_another():
    """The rule that makes three locks safe: they never nest, so they cannot deadlock."""
    source = inspect.getsource(backend_module).splitlines()

    held = []
    nested = []
    for line in source:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        held = [entry for entry in held if entry[1] < indent]
        for name in LOCKS:
            if f"async with self.{name}" in stripped:
                if held:
                    nested.append((name, [entry[0] for entry in held]))
                held.append((name, indent))

    assert nested == []


def test_the_metadata_cache_is_not_guarded_by_the_model_lock():
    """A listing must not queue behind a load; that was the original defect.

    get_model_metadata() is a thin wrapper around _get_raw_model_metadata()
    (the GGUF-cache-backed part that actually needs the lock) plus a
    virtual model's [template] override applied fresh on every call -- see
    that split in backend.py. The invariant still has to hold across both.
    """
    source = inspect.getsource(backend_module.ModelManager.get_model_metadata)
    source += inspect.getsource(backend_module.ModelManager._get_raw_model_metadata)

    assert "self._metadata_lock" in source
    assert "self._models_lock" not in source


async def test_a_held_model_lock_does_not_block_a_metadata_read(manager, gguf_file, toml_file):
    """The original defect: ollama list queued behind a model load."""
    path = gguf_file("Local/model.gguf")
    toml_file("Local/model.toml", '[llm]\nmodel = "Local/model.gguf"\n')
    fingerprint = manager._build_model_file_info("model")["sha256"]
    manager._set_cached_metadata_entry("model", fingerprint, {"arch": "qwen3"})

    async with manager._models_lock:
        metadata = await asyncio.wait_for(manager.get_model_metadata("model"), timeout=1.0)

    # The cached entry is what came back, which is the point here.
    # Metadata carries more than the cache put in it -- a .toml's own
    # overrides are layered on every call -- so this asks about the value
    # under test rather than about everything beside it.
    assert metadata["arch"] == "qwen3"


async def test_a_held_model_lock_does_not_block_taking_a_generation_slot(manager):
    async with manager._models_lock:
        async with asyncio.timeout(1.0):
            async with manager.acquire_inference_slot("m"):
                pass


async def test_a_held_metadata_lock_does_not_block_a_generation_slot(manager):
    async with manager._metadata_lock:
        async with asyncio.timeout(1.0):
            async with manager.acquire_inference_slot("m"):
                pass
