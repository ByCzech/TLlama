"""TLLAMA_RUNTIME_* reaches Llama(), typed by Llama()'s own signature.

Around three dozen load parameters had no server-wide setting at all: a
person wanting n_threads on every model had to write it into every .toml.
The set of names and the type of each is read off the signature rather
than listed, so these check the reading rather than a list someone typed.
"""

import inspect

import pytest

from llama_cpp import Llama

from tllama.config import ConfigError, load_backend_config_from_env
from tllama.helpers.model_toml import parse_model_toml
from tllama.helpers.runtime_params import (
    coerce_runtime_overrides,
    env_key_for,
    settable_parameters,
)


def llm_toml(model_ref: str, extra: str = "") -> str:
    return f'[llm]\nmodel = "{model_ref}"\n{extra}'


class TestWhatIsSettable:
    def test_the_names_come_from_the_signature(self):
        # Not a restatement of the implementation: this compares the
        # result against inspect directly, which is the property that
        # makes a new llama-cpp-python parameter work without an edit.
        from_signature = {
            name
            for name, parameter in inspect.signature(Llama.__init__).parameters.items()
            if name != "self"
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }

        settable = set(settable_parameters())

        assert settable < from_signature
        assert "n_threads" in settable
        assert "use_mlock" in settable
        assert "swa_full" in settable

    def test_model_path_is_not_settable(self):
        with pytest.raises(ConfigError, match="TLLAMA_RUNTIME_MODEL_PATH"):
            coerce_runtime_overrides({"model_path": "/somewhere/else.gguf"})

    @pytest.mark.parametrize(
        "name,pointer",
        [
            ("n_ctx", "TLLAMA_CONTEXT_LENGTH"),
            ("type_k", "TLLAMA_K_CACHE_TYPE"),
            ("type_v", "TLLAMA_V_CACHE_TYPE"),
        ],
    )
    def test_a_parameter_with_its_own_variable_says_which_one(self, name, pointer):
        with pytest.raises(ConfigError, match=pointer):
            coerce_runtime_overrides({name: "1"})

    def test_a_name_that_is_not_a_parameter_is_refused(self):
        with pytest.raises(ConfigError, match="TLLAMA_RUNTIME_N_THREDS"):
            coerce_runtime_overrides({"n_threds": "8"})

    def test_a_parameter_a_string_cannot_express_is_refused(self):
        # chat_handler takes an object. Nothing written in a unit file can
        # become one, and it is set from [mmproj] anyway.
        with pytest.raises(ConfigError, match="TLLAMA_RUNTIME_CHAT_HANDLER"):
            coerce_runtime_overrides({"chat_handler": "something"})


class TestTypesComeFromTheAnnotation:
    def test_an_int_parameter_arrives_as_an_int(self):
        assert coerce_runtime_overrides({"n_threads": "8"}) == {"n_threads": 8}

    def test_a_float_parameter_arrives_as_a_float(self):
        coerced = coerce_runtime_overrides({"rope_freq_base": "10000"})

        assert coerced["rope_freq_base"] == 10000.0
        assert isinstance(coerced["rope_freq_base"], float)

    def test_a_bool_parameter_arrives_as_a_bool(self):
        assert coerce_runtime_overrides({"use_mlock": "yes"})["use_mlock"] is True
        assert coerce_runtime_overrides({"use_mlock": "off"})["use_mlock"] is False

    def test_an_optional_bool_is_read_as_a_bool(self):
        # swa_full is Optional[bool]; the Optional says only that leaving
        # it out is allowed, which is what not setting the variable does.
        assert coerce_runtime_overrides({"swa_full": "true"})["swa_full"] is True

    def test_numa_takes_either_a_word_or_a_number(self):
        # Union[bool, int]: False, or one of the strategy numbers.
        assert coerce_runtime_overrides({"numa": "false"})["numa"] is False
        assert coerce_runtime_overrides({"numa": "2"})["numa"] == 2

    def test_tensor_split_takes_a_comma_separated_list(self):
        coerced = coerce_runtime_overrides({"tensor_split": "0.6, 0.4"})

        assert coerced["tensor_split"] == [0.6, 0.4]

    @pytest.mark.parametrize(
        "name,value",
        [
            ("n_threads", "many"),
            ("rope_freq_base", "high"),
            ("use_mlock", "maybe"),
            ("numa", "sometimes"),
            ("tensor_split", "0.6;0.4"),
        ],
    )
    def test_a_value_of_the_wrong_shape_names_the_variable(self, name, value):
        with pytest.raises(ConfigError, match=env_key_for(name)):
            coerce_runtime_overrides({name: value})


class TestCollectionFromTheEnvironment:
    def test_a_prefixed_variable_becomes_a_parameter(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_RUNTIME_N_THREADS", "8")

        assert load_backend_config_from_env().runtime_overrides == {"n_threads": "8"}

    def test_flash_attention_is_kept_as_an_alias(self, monkeypatch):
        monkeypatch.delenv("TLLAMA_RUNTIME_FLASH_ATTN", raising=False)
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", "1")

        config = load_backend_config_from_env()

        assert config.runtime_overrides == {"flash_attn": "1"}
        assert coerce_runtime_overrides(config.runtime_overrides) == {
            "flash_attn": True
        }

    def test_the_explicit_spelling_beats_the_alias(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", "1")
        monkeypatch.setenv("TLLAMA_RUNTIME_FLASH_ATTN", "0")

        assert load_backend_config_from_env().runtime_overrides == {"flash_attn": "0"}


class TestLayering:
    def test_the_override_reaches_the_load_kwargs(self, make_manager):
        manager = make_manager(runtime_overrides={"n_threads": "8", "use_mlock": "true"})
        spec = parse_model_toml(llm_toml("Local/m.gguf"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["n_threads"] == 8
        assert kwargs["use_mlock"] is True

    def test_a_model_s_runtime_table_wins(self, make_manager):
        manager = make_manager(runtime_overrides={"n_threads": "8"})
        spec = parse_model_toml(llm_toml("Local/m.gguf", "\n[runtime]\nn_threads = 2\n"))

        kwargs = manager._build_llama_load_kwargs("Local/m.gguf", 8192, spec)

        assert kwargs["n_threads"] == 2

    def test_a_bad_value_stops_the_manager_being_built(self, make_manager):
        # At construction, which for the module-level manager is at
        # import: an unusable value is a misconfiguration and should not
        # wait for the first request that loads a model.
        with pytest.raises(ConfigError, match="TLLAMA_RUNTIME_N_THREADS"):
            make_manager(runtime_overrides={"n_threads": "many"})
