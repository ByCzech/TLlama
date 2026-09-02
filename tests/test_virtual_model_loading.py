"""[runtime] from a virtual model's .toml reaches the Llama() construction.

[sampling]/[template]/[system] are a different layer (generation-time, not
load-time) and are deliberately out of scope here -- see the step split in
TLlama_virtual_models_spec.md sec 14 and the conversation that confirmed it.
"""

import asyncio

import pytest

from tllama.config import ConfigError
from tllama.helpers.model_toml import parse_model_toml, resolve_kv_cache_types


class FakeLlama:
    def __init__(self, name, n_ctx=4096):
        self.name = name
        self._n_ctx = n_ctx

    def n_ctx(self):
        return self._n_ctx


def llm_toml(model_ref: str, extra: str = "") -> str:
    return f'[llm]\nmodel = "{model_ref}"\n{extra}'


class TestBuildLlamaLoadKwargs:
    """Pure function: no I/O, no lock, the cheapest place to pin the mapping."""

    def test_no_virtual_spec_keeps_today_s_behavior(self, manager):
        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 8192, None)

        assert kwargs["model_path"] == "Local/model.gguf"
        assert kwargs["n_ctx"] == 8192
        assert kwargs["n_gpu_layers"] == -1

        # Absent, not False: llama-cpp-python defaults use_mmap to True and
        # turns it off by itself when a LoRA is loaded. Passing anything
        # here would override both, and passing True would restate a value
        # TLlama has no reason to own.
        assert "use_mmap" not in kwargs

    def test_runtime_keys_pass_through_generically(self, manager):
        spec = parse_model_toml(llm_toml("Local/m.gguf", '\n[runtime]\nn_gpu_layers = 20\nuse_mmap = true\n'))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["n_gpu_layers"] == 20
        assert kwargs["use_mmap"] is True

    def test_runtime_flash_attn_overrides_global_config_default(self, make_manager):
        manager = make_manager(runtime_overrides={"flash_attn": "false"})
        spec = parse_model_toml(llm_toml("Local/m.gguf", "\n[runtime]\nflash_attn = true\n"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["flash_attn"] is True

    def test_n_ctx_in_runtime_is_not_blindly_passed_through(self, manager):
        # n_ctx priority (client > toml > global config) is resolved by the
        # caller (get_model) before this function ever runs; a stray n_ctx
        # left in the runtime table here must not silently clobber the
        # n_ctx that was already decided.
        spec = parse_model_toml(llm_toml("Local/m.gguf", "\n[runtime]\nn_ctx = 999\n"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["n_ctx"] == 8192

    def test_model_path_in_runtime_cannot_hijack_the_real_path(self, manager):
        spec = parse_model_toml(llm_toml("Local/m.gguf", '\n[runtime]\nmodel_path = "Local/other.gguf"\n'))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["model_path"] == "Local/m.gguf"

    def test_type_k_string_resolves_to_ggml_constant(self, manager):
        spec = parse_model_toml(llm_toml("Local/m.gguf", '\n[runtime]\ntype_k = "q4_0"\n'))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == 2  # GGML_TYPE_Q4_0, verified against the real library

    def test_type_kv_sets_both_type_k_and_type_v(self, manager):
        spec = parse_model_toml(llm_toml("Local/m.gguf", '\n[runtime]\ntype_kv = "f16"\n'))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == kwargs["type_v"] == 1  # GGML_TYPE_F16

    def test_explicit_type_v_overrides_type_kv_for_that_side_only(self, manager):
        spec = parse_model_toml(
            llm_toml("Local/m.gguf", '\n[runtime]\ntype_kv = "f16"\ntype_v = "q4_0"\n')
        )

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == 1  # from type_kv
        assert kwargs["type_v"] == 2  # explicit type_v wins

    def test_toml_type_k_overrides_the_global_config_default(self, make_manager):
        manager = make_manager(kv_cache_type="f16")
        spec = parse_model_toml(llm_toml("Local/m.gguf", '\n[runtime]\ntype_k = "q4_0"\ntype_v = "q4_0"\n'))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == kwargs["type_v"] == 2

    def test_no_runtime_section_falls_back_to_global_config(self, make_manager):
        manager = make_manager(kv_cache_type="f16")
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == kwargs["type_v"] == 1

    def test_global_config_accepts_every_type_the_toml_does(self, make_manager):
        # The global setting used to carry its own name table listing three
        # types, so a name [runtime] accepted was rejected here. q5_1 is one
        # of the types that difference excluded; the point is the two paths
        # now resolve through the same place, not this type specifically.
        manager = make_manager(kv_cache_type="q5_1")
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        from_toml, _ = resolve_kv_cache_types({"type_kv": "q5_1"})
        assert kwargs["type_k"] == kwargs["type_v"] == from_toml

    def test_a_name_no_ggml_type_matches_is_rejected_at_construction(self, make_manager):
        # Not at load: an unknown name is a misconfiguration, and waiting
        # for the first request that needs a model reports it as that
        # model failing to load, hours after the server came up.
        with pytest.raises(ConfigError, match="TLLAMA_KV_CACHE_TYPE"):
            make_manager(kv_cache_type="not_a_real_type")

    def test_the_two_sides_can_be_set_apart(self, make_manager):
        manager = make_manager(k_cache_type="q8_0", v_cache_type="q4_0")
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == resolve_kv_cache_types({"type_kv": "q8_0"})[0]
        assert kwargs["type_v"] == resolve_kv_cache_types({"type_kv": "q4_0"})[0]
        assert kwargs["type_k"] != kwargs["type_v"]

    def test_one_side_overrides_the_shorthand_and_the_other_keeps_it(self, make_manager):
        manager = make_manager(kv_cache_type="f16", k_cache_type="q4_0")
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["type_k"] == 2  # q4_0, the per-side setting
        assert kwargs["type_v"] == 1  # f16, still from the shorthand

    def test_the_precedence_matches_the_toml_it_mirrors(self, make_manager):
        # Not a restatement of the assertions above: this compares the two
        # layers against each other, which is the property that has to hold
        # if the same names are to mean the same thing in both places.
        manager = make_manager(kv_cache_type="f16", v_cache_type="q8_0")
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        from_toml = resolve_kv_cache_types({"type_kv": "f16", "type_v": "q8_0"})
        assert (kwargs["type_k"], kwargs["type_v"]) == from_toml

    @pytest.mark.parametrize(
        "field,variable",
        [
            ("kv_cache_type", "TLLAMA_KV_CACHE_TYPE"),
            ("k_cache_type", "TLLAMA_K_CACHE_TYPE"),
            ("v_cache_type", "TLLAMA_V_CACHE_TYPE"),
        ],
    )
    def test_a_rejection_names_the_variable_that_was_wrong(
        self, make_manager, field, variable
    ):
        with pytest.raises(ConfigError, match=variable):
            make_manager(**{field: "not_a_real_type"})


class TestGetModelUsesVirtualSpec:
    """End-to-end through get_model(), with _load_model_sync faked out."""

    @pytest.fixture
    def fake_load(self, manager, monkeypatch):
        calls = []

        def load(model_path, requested_n_ctx, virtual_spec=None):
            calls.append({"path": model_path, "n_ctx": requested_n_ctx, "spec": virtual_spec})
            return FakeLlama("m", requested_n_ctx or 4096), {}, ""

        monkeypatch.setattr(manager, "_load_model_sync", load)
        return calls

    async def test_toml_n_ctx_used_when_client_does_not_specify(
        self, manager, gguf_file, toml_file, fake_load
    ):
        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf", "\n[runtime]\nn_ctx = 16384\n"))

        await manager.get_model("MyModel")

        assert fake_load[0]["n_ctx"] == 16384

    async def test_client_num_ctx_overrides_toml_n_ctx(self, manager, gguf_file, toml_file, fake_load):
        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf", "\n[runtime]\nn_ctx = 16384\n"))

        await manager.get_model("MyModel", num_ctx=4096)

        assert fake_load[0]["n_ctx"] == 4096

    async def test_global_config_used_when_neither_client_nor_toml_specify(
        self, make_manager, gguf_file, toml_file, monkeypatch
    ):
        manager = make_manager(context_length=2048)
        calls = []

        def load(model_path, requested_n_ctx, virtual_spec=None):
            calls.append(requested_n_ctx)
            return FakeLlama("m", requested_n_ctx or 4096), {}, ""

        monkeypatch.setattr(manager, "_load_model_sync", load)

        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf"))

        await manager.get_model("MyModel")

        assert calls[0] == 2048

    async def test_virtual_spec_is_passed_to_load(self, manager, gguf_file, toml_file, fake_load):
        gguf_file("Local/m.gguf")
        toml_file("Local/MyModel.toml", llm_toml("Local/m.gguf", "\n[runtime]\nn_gpu_layers = 5\n"))

        await manager.get_model("MyModel")

        assert fake_load[0]["spec"] is not None
        assert fake_load[0]["spec"].runtime["n_gpu_layers"] == 5

    async def test_a_bare_gguf_with_no_toml_cannot_be_loaded(self, manager, gguf_file, fake_load):
        gguf_file("Local/orphan.gguf")

        with pytest.raises(FileNotFoundError):
            await manager.get_model("orphan")

        assert fake_load == []

    async def test_cross_repository_reference_loads_the_right_file(
        self, manager, gguf_file, toml_file, fake_load
    ):
        gguf_file("HuggingFace/unsloth/Repo-GGUF/model.gguf")
        toml_file("Local/MyModel.toml", llm_toml("HuggingFace/unsloth/Repo-GGUF/model.gguf"))

        await manager.get_model("MyModel")

        assert fake_load[0]["path"].endswith("HuggingFace/unsloth/Repo-GGUF/model.gguf")


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
