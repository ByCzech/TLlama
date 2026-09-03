"""Parsing a virtual-model .toml file into VirtualModelSpec.

Pure parsing only -- no filesystem scan, no llama_cpp Llama() construction.
That comes in later steps (spec doc TLlama_virtual_models_spec.md §14).
"""

import pytest

from tllama.helpers.model_toml import (
    TomlModelError,
    parse_model_toml,
    resolve_kv_cache_types,
    resolve_repo_relative_path,
)


MINIMAL = '''
[llm]
model = "HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
'''


class TestLlmSection:
    def test_model_only_is_valid(self):
        spec = parse_model_toml(MINIMAL)
        assert spec.llm_model == "HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
        assert spec.llm_from is None

    def test_from_only_is_valid(self):
        spec = parse_model_toml('[llm]\nfrom = "/outside/repo/model.gguf"\n')
        assert spec.llm_from == "/outside/repo/model.gguf"
        assert spec.llm_model is None

    def test_missing_llm_section_rejected(self):
        with pytest.raises(TomlModelError, match=r"\[llm\]"):
            parse_model_toml("[runtime]\nn_ctx = 8192\n")

    def test_both_model_and_from_rejected(self):
        text = '[llm]\nmodel = "a.gguf"\nfrom = "/b.gguf"\n'
        with pytest.raises(TomlModelError, match="both"):
            parse_model_toml(text)

    def test_neither_model_nor_from_rejected(self):
        with pytest.raises(TomlModelError, match="either"):
            parse_model_toml("[llm]\n")

    def test_non_string_model_rejected(self):
        with pytest.raises(TomlModelError, match="string"):
            parse_model_toml("[llm]\nmodel = 123\n")


class TestMmprojSection:
    def test_absent_mmproj_is_fine(self):
        spec = parse_model_toml(MINIMAL)
        assert spec.mmproj_model is None
        assert spec.mmproj_from is None

    def test_mmproj_model(self):
        text = MINIMAL + '\n[mmproj]\nmodel = "HuggingFace/ns/repo/mmproj-X.gguf"\n'
        spec = parse_model_toml(text)
        assert spec.mmproj_model == "HuggingFace/ns/repo/mmproj-X.gguf"

    def test_mmproj_both_model_and_from_rejected(self):
        text = MINIMAL + '\n[mmproj]\nmodel = "a.gguf"\nfrom = "/b.gguf"\n'
        with pytest.raises(TomlModelError, match="both"):
            parse_model_toml(text)

    def test_mmproj_neither_rejected(self):
        text = MINIMAL + "\n[mmproj]\n"
        with pytest.raises(TomlModelError, match="either"):
            parse_model_toml(text)


class TestRuntimeAndSampling:
    def test_runtime_keeps_native_types(self):
        text = MINIMAL + '''
[runtime]
n_ctx = 8192
flash_attn = true
n_gpu_layers = -1
type_k = "q4_0"
'''
        spec = parse_model_toml(text)
        assert spec.runtime["n_ctx"] == 8192
        assert isinstance(spec.runtime["n_ctx"], int)
        assert spec.runtime["flash_attn"] is True
        assert spec.runtime["n_gpu_layers"] == -1
        assert spec.runtime["type_k"] == "q4_0"

    def test_sampling_stop_extracted_as_plain_list(self):
        text = MINIMAL + '''
[sampling]
temperature = 0.7
stop = ["<|im_end|>", "<|endoftext|>"]
'''
        spec = parse_model_toml(text)
        assert spec.stop == ["<|im_end|>", "<|endoftext|>"]
        assert "stop" not in spec.sampling
        assert spec.sampling["temperature"] == 0.7

    def test_missing_stop_defaults_to_empty_list(self):
        spec = parse_model_toml(MINIMAL)
        assert spec.stop == []

    def test_stop_must_be_an_array(self):
        text = MINIMAL + '\n[sampling]\nstop = "<|im_end|>"\n'
        with pytest.raises(TomlModelError, match="array"):
            parse_model_toml(text)

    def test_stop_entries_must_be_strings(self):
        text = MINIMAL + "\n[sampling]\nstop = [1, 2]\n"
        with pytest.raises(TomlModelError, match="strings"):
            parse_model_toml(text)

    def test_absent_runtime_and_sampling_default_empty(self):
        spec = parse_model_toml(MINIMAL)
        assert spec.runtime == {}
        assert spec.sampling == {}


class TestTemplateAndSystem:
    def test_multiline_template_and_system(self):
        # TOML trims a newline immediately after the opening """, so the
        # blank line right under the header does not become a leading \n.
        text = MINIMAL + '''
[template]
jinja = """
{{ messages }}
line two
"""

[system]
prompt = """
Jsi uzitecny asistent.
"""
'''
        spec = parse_model_toml(text)
        assert spec.template == "{{ messages }}\nline two\n"
        assert spec.system_prompt == "Jsi uzitecny asistent.\n"

    def test_absent_template_and_system_are_none(self):
        spec = parse_model_toml(MINIMAL)
        assert spec.template is None
        assert spec.system_prompt is None

    def test_percent_sign_in_template_does_not_break_parsing(self):
        # The exact configparser gotcha (spec doc §2) that motivated TOML
        # over INI in the first place.
        text = MINIMAL + '\n[template]\njinja = "50% done"\n'
        spec = parse_model_toml(text)
        assert spec.template == "50% done"


class TestSyntaxErrors:
    def test_invalid_toml_raises_toml_model_error(self):
        with pytest.raises(TomlModelError, match="invalid TOML"):
            parse_model_toml("this is not [[[ valid")

    def test_error_message_includes_source(self):
        with pytest.raises(TomlModelError, match="my-model.toml"):
            parse_model_toml("[runtime]\n", source="my-model.toml")


class TestRoundTripDocument:
    def test_document_preserves_comments_for_later_rewrite(self):
        text = '# a note from a human\n[llm]\nmodel = "a.gguf"  # inline note\n'
        spec = parse_model_toml(text)
        import tomlkit

        assert "a note from a human" in tomlkit.dumps(spec.document)
        assert "inline note" in tomlkit.dumps(spec.document)


class TestResolveKvCacheTypes:
    def test_raw_ints_pass_through(self):
        type_k, type_v = resolve_kv_cache_types({"type_k": 2, "type_v": 8})
        assert (type_k, type_v) == (2, 8)

    def test_type_kv_sets_both(self):
        type_k, type_v = resolve_kv_cache_types({"type_kv": "q4_0"})
        assert type_k == type_v
        assert type_k is not None

    def test_explicit_type_k_overrides_type_kv_for_that_side_only(self):
        type_k, type_v = resolve_kv_cache_types({"type_kv": "q4_0", "type_k": "f16"})
        assert type_k != type_v

    def test_unknown_type_name_raises(self):
        with pytest.raises(TomlModelError, match="GGML_TYPE_NOT_A_REAL_TYPE"):
            resolve_kv_cache_types({"type_k": "not_a_real_type"})

    def test_absent_keys_resolve_to_none(self):
        assert resolve_kv_cache_types({}) == (None, None)

    def test_bool_is_rejected_despite_being_an_int_subclass(self):
        with pytest.raises(TomlModelError, match="bool"):
            resolve_kv_cache_types({"type_k": True})

    def test_wrong_type_rejected(self):
        with pytest.raises(TomlModelError, match="string or int"):
            resolve_kv_cache_types({"type_k": 3.5})


class TestResolveRepoRelativePath:
    def test_path_inside_repo_resolves(self, tmp_path):
        models_dir = tmp_path / "models"
        (models_dir / "HuggingFace" / "ns" / "repo").mkdir(parents=True)
        target = models_dir / "HuggingFace" / "ns" / "repo" / "model.gguf"
        target.touch()

        resolved = resolve_repo_relative_path("HuggingFace/ns/repo/model.gguf", models_dir)

        assert resolved == target.resolve()

    def test_escape_attempt_rejected(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(TomlModelError, match="outside"):
            resolve_repo_relative_path("../../../etc/passwd", models_dir)

    def test_escape_attempt_disguised_mid_path_rejected(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(TomlModelError, match="outside"):
            resolve_repo_relative_path("Local/../../outside.gguf", models_dir)


class TestUnknownKeysAreRefused:
    """A key nothing reads used to be carried through and dropped.

    Both tables did it, by different routes: Llama() takes **kwargs and
    never inspects what it did not expect, and build_sampling_kwargs
    consults a fixed set of names and ignores the rest. Either way the
    file said one thing and the server did another, with no way to tell
    from the outside which.
    """

    def test_a_misspelt_runtime_key_is_refused(self):
        with pytest.raises(TomlModelError, match="n_threds"):
            parse_model_toml(
                '[llm]\nmodel = "Local/m.gguf"\n[runtime]\nn_threds = 8\n'
            )

    def test_a_real_llama_parameter_is_accepted(self):
        # The counterweight: the check is against Llama()'s signature, so
        # anything the installed library takes has to keep working,
        # including names TLlama's own code never mentions.
        spec = parse_model_toml(
            '[llm]\nmodel = "Local/m.gguf"\n[runtime]\nyarn_beta_slow = 1.5\n'
        )

        assert spec.runtime["yarn_beta_slow"] == 1.5

    def test_type_kv_is_accepted_though_it_is_not_a_llama_parameter(self):
        # TLlama's own shorthand for setting both cache sides at once.
        spec = parse_model_toml(
            '[llm]\nmodel = "Local/m.gguf"\n[runtime]\ntype_kv = "q4_0"\n'
        )

        assert spec.runtime["type_kv"] == "q4_0"

    def test_a_misspelt_sampling_key_is_refused(self):
        with pytest.raises(TomlModelError, match="temperatur"):
            parse_model_toml(
                '[llm]\nmodel = "Local/m.gguf"\n[sampling]\ntemperatur = 0.7\n'
            )

    def test_max_tokens_and_stop_are_accepted(self):
        # Neither is in _SAMPLING_DEFAULTS; both are read directly, so a
        # check built only from that table would reject them.
        spec = parse_model_toml(
            '[llm]\nmodel = "Local/m.gguf"\n[sampling]\n'
            'max_tokens = 512\nstop = ["<|im_end|>"]\n'
        )

        assert spec.sampling["max_tokens"] == 512
        assert spec.stop == ["<|im_end|>"]

    def test_a_sampling_name_llama_takes_but_tllama_does_not_apply_is_refused(self):
        # grammar is a real create_completion parameter that
        # build_sampling_kwargs does not pass on. Accepting it here would
        # promise something that still goes nowhere -- and worse than
        # nowhere: GBNF llama.cpp cannot parse kills the process, and
        # nothing available parses it early enough to catch that.
        # logit_bias used to stand here and is applied as of this patch.
        with pytest.raises(TomlModelError, match="grammar"):
            parse_model_toml(
                '[llm]\nmodel = "Local/m.gguf"\n[sampling]\ngrammar = "root ::= \\"a\\""\n'
            )
