import hashlib
import os

import pytest

from tllama.backend import (
    HF_LOOKUP_FOUND,
    HF_LOOKUP_MISSING,
    HF_LOOKUP_UNKNOWN,
    HfFileLookup,
)
from tllama.helpers import metadata_cache


@pytest.mark.parametrize("size", [0, 1, 8 * 1024 * 1024 - 1, 8 * 1024 * 1024 + 1234])
def test_digest_matches_a_plain_sha256_across_the_read_boundary(
    manager, gguf_file, size
):
    path = gguf_file(f"Local/size-{size}.gguf", os.urandom(size))

    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert manager._compute_content_sha256(path) == expected


async def test_identical_bytes_get_the_same_digest_wherever_they_live(
    manager, gguf_file
):
    """The digest is the model's identity, so it must not encode its location."""
    payload = os.urandom(2048)
    paths = [
        gguf_file("Local/one.gguf", payload),
        gguf_file("TLlama/ns/two.gguf", payload),
        gguf_file("HuggingFace/ns/repo/three.gguf", payload),
    ]

    digests = {(await manager.get_model_digest(path))["content_sha256"] for path in paths}

    assert len(digests) == 1


async def test_a_single_changed_byte_changes_the_digest(manager, gguf_file):
    payload = bytearray(os.urandom(4096))
    path = gguf_file("Local/model.gguf", bytes(payload))
    before = (await manager.get_model_digest(path))["content_sha256"]

    payload[1000] ^= 0x01
    path.write_bytes(bytes(payload))

    assert (await manager.get_model_digest(path))["content_sha256"] != before


async def test_a_locally_computed_digest_is_marked_as_such(manager, gguf_file):
    path = gguf_file("Local/model.gguf")

    assert (await manager.get_model_digest(path))["source"] == "local"


async def test_a_cached_digest_is_not_recomputed(manager, gguf_file, monkeypatch):
    path = gguf_file("Local/model.gguf")
    first = await manager.get_model_digest(path)

    def fail(*args, **kwargs):
        raise AssertionError("the file was read again")

    monkeypatch.setattr(manager, "_compute_content_sha256", fail)

    assert await manager.get_model_digest(path) == first


async def test_touching_the_file_invalidates_the_cached_digest(manager, gguf_file):
    path = gguf_file("Local/model.gguf")
    await manager.get_model_digest(path)

    os.utime(path, ns=(0, 12345))

    assert metadata_cache.load_digest_cache(manager.metadata_cache_dir, path) is None


async def test_an_unreadable_file_yields_no_digest_rather_than_a_substitute(
    manager, models_root
):
    assert await manager.get_model_digest(models_root / "Local/absent.gguf") is None


def test_the_two_cache_documents_are_independent(manager, gguf_file, monkeypatch):
    """Rehashing a multi-gigabyte file must not follow a metadata layout change."""
    path = gguf_file("Local/model.gguf")
    metadata_cache.save_metadata_cache(
        manager.metadata_cache_dir, "model", path, {"arch": "qwen3"}
    )
    metadata_cache.save_digest_cache(
        manager.metadata_cache_dir, "model", path, {"content_sha256": "x", "source": "local"}
    )

    monkeypatch.setattr(metadata_cache, "SCHEMA_VERSION", metadata_cache.SCHEMA_VERSION + 1)

    assert metadata_cache.load_metadata_cache(manager.metadata_cache_dir, path) is None
    assert metadata_cache.load_digest_cache(manager.metadata_cache_dir, path) is not None


def test_deleting_a_model_removes_both_cache_documents(manager, gguf_file):
    path = gguf_file("Local/model.gguf")
    metadata_cache.save_metadata_cache(manager.metadata_cache_dir, "model", path, {"a": 1})
    metadata_cache.save_digest_cache(manager.metadata_cache_dir, "model", path, {"b": 2})

    manager.delete_model_file("model")

    assert not metadata_cache.get_metadata_cache_path(manager.metadata_cache_dir, path).exists()
    assert not metadata_cache.get_digest_cache_path(manager.metadata_cache_dir, path).exists()


async def test_a_published_digest_is_recorded_without_reading_the_file(
    manager, gguf_file, monkeypatch
):
    payload = os.urandom(512)
    path = gguf_file("HuggingFace/ns/repo/model.gguf", payload)
    published = hashlib.sha256(payload).hexdigest()

    def fail(*args, **kwargs):
        raise AssertionError("the file was hashed despite a usable published digest")

    monkeypatch.setattr(manager, "_compute_content_sha256", fail)

    stored = await manager.store_hf_digest(
        path, HfFileLookup(HF_LOOKUP_FOUND, published, len(payload))
    )

    assert stored == {"content_sha256": published, "source": "hf"}


@pytest.mark.parametrize(
    "lookup",
    [
        None,
        HfFileLookup(HF_LOOKUP_UNKNOWN),
        HfFileLookup(HF_LOOKUP_MISSING),
        HfFileLookup(HF_LOOKUP_FOUND, "a" * 64, 999999),
    ],
    ids=["absent", "unknown", "missing", "size-mismatch"],
)
async def test_an_unusable_published_digest_is_not_recorded(manager, gguf_file, lookup):
    """A published sha256 is a claim; a size that disagrees withdraws it."""
    path = gguf_file("HuggingFace/ns/repo/model.gguf")

    assert await manager.store_hf_digest(path, lookup) is None
    assert metadata_cache.load_digest_cache(manager.metadata_cache_dir, path) is None
