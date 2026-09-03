"""TLLAMA_SAMPLING_*: sampling set server-wide.

Two things are being tested. That the variables are collected, named and
converted against the real completion signatures rather than a list; and
that the layer sits where it is supposed to in the priority chain, which
is only visible by comparing it against the layers on either side.
"""

import pytest

from tllama.config import ConfigError, load_backend_config_from_env
from tllama.helpers.common import build_sampling_kwargs
from tllama.helpers.sampling_params import (
    coerce_sampling_overrides,
    completion_parameters,
    env_key_for,
    settable_parameters,
)


class TestCollection:
    def test_a_variable_is_collected_lowercased(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_SAMPLING_TEMPERATURE", "0.3")

        config = load_backend_config_from_env()

        assert config.sampling_overrides == {"temperature": "0.3"}

    def test_nothing_set_collects_nothing(self, monkeypatch):
        for name in list(__import__("os").environ):
            if name.startswith("TLLAMA_SAMPLING_"):
                monkeypatch.delenv(name)

        assert load_backend_config_from_env().sampling_overrides == {}

    def test_an_empty_value_is_not_a_setting(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_SAMPLING_TOP_K", "   ")

        assert load_backend_config_from_env().sampling_overrides == {}


class TestNamesAndTypes:
    def test_the_name_set_comes_from_both_signatures(self):
        """Only parameters both completion calls take. build_sampling_kwargs
        is one function for three endpoints, so a name on just one method
        would work somewhere and be a TypeError elsewhere."""
        parameters = completion_parameters()

        assert "temperature" in parameters
        assert "suffix" not in parameters  # create_completion only
        assert "response_format" not in parameters  # create_chat_completion only

    def test_settable_is_narrowed_to_what_tllama_applies(self):
        settable = settable_parameters()

        assert "temperature" in settable
        assert "stream" not in settable  # real parameter, TLlama's to decide

    def test_an_int_becomes_an_int_and_a_float_a_float(self):
        coerced = coerce_sampling_overrides({"top_k": "20", "temperature": "1.0"})

        assert coerced == {"top_k": 20, "temperature": 1.0}
        assert isinstance(coerced["top_k"], int)
        assert isinstance(coerced["temperature"], float)

    def test_a_value_of_the_wrong_kind_stops_the_server(self):
        with pytest.raises(ConfigError) as excinfo:
            coerce_sampling_overrides({"top_k": "quite a lot"})

        assert env_key_for("top_k") in str(excinfo.value)

    def test_a_name_tllama_does_not_apply_says_so(self):
        with pytest.raises(ConfigError) as excinfo:
            coerce_sampling_overrides({"stream": "true"})

        assert "does not apply" in str(excinfo.value)

    def test_an_invented_name_says_that_instead(self):
        with pytest.raises(ConfigError) as excinfo:
            coerce_sampling_overrides({"temprature": "0.5"})

        assert "not a sampling parameter" in str(excinfo.value)

    @pytest.mark.parametrize("name", ["logit_bias", "stop"])
    def test_a_shape_a_string_cannot_express_points_at_the_toml(self, name):
        """Falls out of the coercion rather than a list of exceptions, so
        it stays right if one of these ever changes type."""
        with pytest.raises(ConfigError) as excinfo:
            coerce_sampling_overrides({name: "anything"})

        message = str(excinfo.value)
        assert ".toml" in message
        assert "[sampling]" in message


class TestPriorityChain:
    def test_a_global_value_beats_the_baseline(self):
        kwargs = build_sampling_kwargs({}, {}, {"temperature": 0.1})

        assert kwargs["temperature"] == 0.1

    def test_a_toml_beats_a_global_value(self):
        kwargs = build_sampling_kwargs(
            {},
            {"sampling_defaults": {"temperature": 0.5}},
            {"temperature": 0.1},
        )

        assert kwargs["temperature"] == 0.5

    def test_a_request_beats_both(self):
        kwargs = build_sampling_kwargs(
            {"temperature": 0.9},
            {"sampling_defaults": {"temperature": 0.5}},
            {"temperature": 0.1},
        )

        assert kwargs["temperature"] == 0.9

    def test_an_unset_global_leaves_the_baseline_alone(self):
        """The whole point of holding only what was set: a server nobody
        configured has to behave exactly as it did before."""
        with_layer = build_sampling_kwargs({}, {}, {})
        without_layer = build_sampling_kwargs({}, {})

        assert with_layer == without_layer
        assert with_layer["temperature"] == 0.8

    def test_a_global_value_for_one_key_leaves_the_others_alone(self):
        kwargs = build_sampling_kwargs({}, {}, {"top_k": 20})

        assert kwargs["top_k"] == 20
        assert kwargs["temperature"] == 0.8

    def test_max_tokens_follows_the_same_chain(self):
        assert build_sampling_kwargs({}, {}, {"max_tokens": 256})["max_tokens"] == 256
        assert build_sampling_kwargs(
            {}, {"sampling_defaults": {"max_tokens": 64}}, {"max_tokens": 256}
        )["max_tokens"] == 64
        assert build_sampling_kwargs(
            {"num_predict": 8}, {"sampling_defaults": {"max_tokens": 64}}, {"max_tokens": 256}
        )["max_tokens"] == 8

    def test_max_tokens_unset_everywhere_stays_none(self):
        assert build_sampling_kwargs({}, {}, {})["max_tokens"] is None


class TestReachesTheEndpoints:
    def test_the_manager_exposes_what_the_routers_pass(self, monkeypatch):
        """A global value is server configuration, so it travels beside the
        metadata rather than inside it."""
        monkeypatch.setenv("TLLAMA_SAMPLING_TOP_K", "20")

        from tllama.backend import ModelManager

        manager = ModelManager(load_backend_config_from_env())

        assert manager.sampling_overrides == {"top_k": 20}

    def test_an_unusable_value_stops_the_manager_being_built(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_SAMPLING_TEMPERATURE", "warm")

        from tllama.backend import ModelManager

        with pytest.raises(ConfigError):
            ModelManager(load_backend_config_from_env())
