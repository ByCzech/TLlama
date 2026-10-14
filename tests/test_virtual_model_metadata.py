"""Virtual definitions apply metadata from the underlying GGUF cache."""

import asyncio

def llm_toml(model_ref: str, extra: str = "") -> str:
    return f'[llm]\nmodel = "{model_ref}"\n{extra}'


class TestGetModelMetadataUsesVirtualSpec:
    async def test_metadata_fingerprint_comes_from_the_underlying_file(
        self, manager, gguf_file, toml_file
    ):
        gguf_file("Local/m.gguf", payload=b"some particular bytes")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf"))

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry("MyModel", fingerprint, {"arch": "qwen3"})

        metadata = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        # Only the cache, keyed by the .gguf's fingerprint, could have put
        # this here: the file itself is not a readable GGUF. What else the
        # .toml layer adds on top is not what this is measuring.
        assert metadata["arch"] == "qwen3"

    async def test_no_toml_means_no_metadata(self, manager, gguf_file):
        gguf_file("Local/orphan.gguf")

        metadata = await manager.get_model_metadata("orphan")

        assert metadata is None
