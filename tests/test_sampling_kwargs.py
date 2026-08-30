"""build_sampling_kwargs(): one place, one priority chain, for all three
inference endpoints (/api/generate, /api/chat, /v1/chat/completions).
"""

from tllama.helpers.common import build_sampling_kwargs, _SAMPLING_DEFAULTS


class TestBaselineDefaults:
    def test_nothing_provided_uses_tllama_s_own_baseline(self):
        kwargs = build_sampling_kwargs({}, {})

        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40
        assert kwargs["min_p"] == 0.05
        assert kwargs["repeat_penalty"] == 1.0
        assert kwargs["mirostat_mode"] == 0

    def test_every_declared_default_key_is_present(self):
        kwargs = build_sampling_kwargs({}, {})

        for key in _SAMPLING_DEFAULTS:
            assert key in kwargs

    def test_none_opts_and_none_metadata_info_do_not_crash(self):
        kwargs = build_sampling_kwargs(None, None)

        assert kwargs["temperature"] == 0.8


class TestClientOverridesToml:
    def test_client_wins_over_toml(self):
        opts = {"temperature": 0.3}
        metadata_info = {"sampling_defaults": {"temperature": 0.5}}

        kwargs = build_sampling_kwargs(opts, metadata_info)

        assert kwargs["temperature"] == 0.3

    def test_toml_wins_over_baseline_when_client_silent(self):
        metadata_info = {"sampling_defaults": {"temperature": 0.5}}

        kwargs = build_sampling_kwargs({}, metadata_info)

        assert kwargs["temperature"] == 0.5

    def test_every_parameter_respects_the_same_priority(self):
        for key, baseline in _SAMPLING_DEFAULTS.items():
            client_key = {"mirostat_mode": "mirostat"}.get(key, key)
            client_value = 999 if not isinstance(baseline, float) else 0.999
            toml_value = 888 if not isinstance(baseline, float) else 0.888

            kwargs = build_sampling_kwargs({client_key: client_value}, {"sampling_defaults": {key: toml_value}})
            assert kwargs[key] == client_value, f"{key}: client should win"

            kwargs = build_sampling_kwargs({}, {"sampling_defaults": {key: toml_value}})
            assert kwargs[key] == toml_value, f"{key}: toml should win over baseline"

            kwargs = build_sampling_kwargs({}, {})
            assert kwargs[key] == baseline, f"{key}: baseline should apply"


class TestMirostatAlias:
    def test_client_uses_ollama_s_own_option_name(self):
        kwargs = build_sampling_kwargs({"mirostat": 2}, {})

        assert kwargs["mirostat_mode"] == 2

    def test_toml_uses_the_kwarg_style_name(self):
        metadata_info = {"sampling_defaults": {"mirostat_mode": 2}}

        kwargs = build_sampling_kwargs({}, metadata_info)

        assert kwargs["mirostat_mode"] == 2


class TestMaxTokens:
    def test_client_num_predict_wins(self):
        opts = {"num_predict": 256}
        metadata_info = {"sampling_defaults": {"max_tokens": 512}}

        kwargs = build_sampling_kwargs(opts, metadata_info)

        assert kwargs["max_tokens"] == 256

    def test_toml_max_tokens_used_when_client_silent(self):
        metadata_info = {"sampling_defaults": {"max_tokens": 512}}

        kwargs = build_sampling_kwargs({}, metadata_info)

        assert kwargs["max_tokens"] == 512

    def test_neither_set_is_none(self):
        kwargs = build_sampling_kwargs({}, {})

        assert kwargs["max_tokens"] is None

    def test_client_num_predict_zero_or_negative_means_unlimited(self):
        opts = {"num_predict": -1}
        metadata_info = {"sampling_defaults": {"max_tokens": 512}}

        kwargs = build_sampling_kwargs(opts, metadata_info)

        assert kwargs["max_tokens"] is None

    def test_client_num_predict_zero_also_wins_over_toml(self):
        opts = {"num_predict": 0}
        metadata_info = {"sampling_defaults": {"max_tokens": 512}}

        kwargs = build_sampling_kwargs(opts, metadata_info)

        assert kwargs["max_tokens"] is None


class TestStop:
    def test_client_stop_wins(self):
        opts = {"stop": ["CLIENT"]}
        metadata_info = {"stop_defaults": ["TOML"]}

        kwargs = build_sampling_kwargs(opts, metadata_info)

        assert kwargs["stop"] == ["CLIENT"]

    def test_toml_stop_used_when_client_silent(self):
        metadata_info = {"stop_defaults": ["TOML"]}

        kwargs = build_sampling_kwargs({}, metadata_info)

        assert kwargs["stop"] == ["TOML"]

    def test_neither_set_omits_the_key(self):
        kwargs = build_sampling_kwargs({}, {})

        assert "stop" not in kwargs

    def test_client_stop_as_bare_string_is_normalized(self):
        opts = {"stop": "CLIENT"}

        kwargs = build_sampling_kwargs(opts, {})

        assert kwargs["stop"] == ["CLIENT"]
