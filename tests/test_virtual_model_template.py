"""[template] from a virtual model's .toml overrides the GGUF's own template.

The override must apply fresh on every call, not get baked into the
GGUF-fingerprint-keyed metadata cache -- editing the .toml has to take
effect immediately, without needing the underlying .gguf to also change.
"""

import asyncio


def llm_toml(model_ref: str, extra: str = "") -> str:
    return f'[llm]\nmodel = "{model_ref}"\n{extra}'


class TestTemplateOverride:
    async def test_toml_template_overrides_cached_gguf_metadata(
        self, manager, gguf_file, toml_file
    ):
        gguf_file("Local/m.gguf")
        toml_file(
            "Local/MyModel.toml",
            llm_toml("Local/m.gguf", '\n[template]\njinja = "custom template text"\n'),
        )

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry(
            "MyModel", fingerprint, {"arch": "qwen3", "template": "gguf's own baked-in template"}
        )

        metadata = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        assert metadata["template"] == "custom template text"
        assert metadata["arch"] == "qwen3"

    async def test_no_template_section_leaves_gguf_template_untouched(
        self, manager, gguf_file, toml_file
    ):
        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf"))

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry(
            "MyModel", fingerprint, {"arch": "qwen3", "template": "gguf's own baked-in template"}
        )

        metadata = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        assert metadata["template"] == "gguf's own baked-in template"

    async def test_override_does_not_mutate_the_cached_entry(self, manager, gguf_file, toml_file):
        # A second, unrelated read of the same cached entry must still see
        # the GGUF's own template -- the override must not have leaked into
        # the cache itself.
        gguf_file("Local/m.gguf")
        toml_file(
            "Local/MyModel.toml",
            llm_toml("Local/m.gguf", '\n[template]\njinja = "custom template text"\n'),
        )

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        cached = {"arch": "qwen3", "template": "gguf's own baked-in template"}
        manager._set_cached_metadata_entry("MyModel", fingerprint, cached)

        await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        assert cached["template"] == "gguf's own baked-in template"

    async def test_editing_the_toml_takes_effect_without_touching_the_gguf(
        self, manager, gguf_file, toml_file
    ):
        # This is the whole point of applying the override outside the
        # cache: the .gguf's own fingerprint never changes here, only the
        # .toml's [template] value between the two reads.
        gguf_file("Local/m.gguf")
        toml_path = toml_file(
            "Local/MyModel.toml",
            llm_toml("Local/m.gguf", '\n[template]\njinja = "version one"\n'),
        )

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry("MyModel", fingerprint, {"template": "gguf default"})

        first = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)
        assert first["template"] == "version one"

        toml_path.write_text(llm_toml("Local/m.gguf", '\n[template]\njinja = "version two"\n'))

        second = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)
        assert second["template"] == "version two"

    async def test_no_toml_at_all_is_unaffected(self, manager, gguf_file):
        # A plain reference with no .toml already returns None from
        # get_model_metadata (step 3a); confirm the override logic does not
        # change that.
        gguf_file("Local/orphan.gguf")

        metadata = await manager.get_model_metadata("orphan")

        assert metadata is None


class TestSystemPromptOverride:
    async def test_toml_system_prompt_surfaces_as_default_system_prompt(
        self, manager, gguf_file, toml_file
    ):
        gguf_file("Local/m.gguf")
        toml_file(
            "Local/MyModel.toml",
            llm_toml("Local/m.gguf", '\n[system]\nprompt = "be nice"\n'),
        )

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry("MyModel", fingerprint, {"arch": "qwen3"})

        metadata = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        assert metadata["default_system_prompt"] == "be nice"

    async def test_no_system_section_means_no_default_key(self, manager, gguf_file, toml_file):
        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf"))

        fingerprint = manager._build_model_file_info("MyModel")["sha256"]
        manager._set_cached_metadata_entry("MyModel", fingerprint, {"arch": "qwen3"})

        metadata = await asyncio.wait_for(manager.get_model_metadata("MyModel"), timeout=1.0)

        assert "default_system_prompt" not in metadata
