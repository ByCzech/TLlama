"""A number that parses is not yet a number that means something.

Every one of these used to convert and then be used, so the setting was
neither honoured nor refused: it did something else. The tests name what
that something else was, because a bound with no reason behind it is one
TLlama invented, and the sampling layer next door deliberately has none
for exactly that reason.
"""

import pytest

from tllama.config import ConfigError
from tllama.daemon_config import load_daemon_config


class TestAValueOutsideItsRangeIsRefused:
    def test_no_room_for_a_model_is_not_room_for_one(self, monkeypatch):
        # backend._ensure_capacity_for_load reads `<= 1` as "unload
        # everything else and keep one", so 0 and -5 both behave exactly
        # like 1 -- an operator asking for no resident models gets one.
        monkeypatch.setenv("TLLAMA_MAX_LOADED_MODELS", "0")

        with pytest.raises(ConfigError) as excinfo:
            load_daemon_config(False, discover=False).backend

        assert "TLLAMA_MAX_LOADED_MODELS" in str(excinfo.value)
        assert "0" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_a_collection_interval_has_to_pass_time(self, monkeypatch, value):
        # asyncio.sleep() has `if delay <= 0: await __sleep0(); return`,
        # so the janitor loop would spin without ever waiting.
        monkeypatch.setenv("TLLAMA_JANITOR_INTERVAL", value)

        with pytest.raises(ConfigError, match="TLLAMA_JANITOR_INTERVAL"):
            load_daemon_config(False, discover=False).backend

    def test_a_negative_context_length_is_not_a_context_length(self, monkeypatch):
        # _normalize_num_ctx() ends in `return value if value > 0 else
        # default`, so -1 silently became the default it was written to
        # replace.
        monkeypatch.setenv("TLLAMA_CONTEXT_LENGTH", "-1")

        with pytest.raises(ConfigError, match="TLLAMA_CONTEXT_LENGTH"):
            load_daemon_config(False, discover=False).backend

    def test_a_negative_cache_lifetime_is_not_a_lifetime(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_METADATA_CACHE_TTL", "-0.5")

        with pytest.raises(ConfigError, match="TLLAMA_METADATA_CACHE_TTL"):
            load_daemon_config(False, discover=False).backend

    def test_the_message_says_what_would_have_been_accepted(self, monkeypatch):
        # "is not valid" alone leaves an operator guessing which way to
        # move the number.
        monkeypatch.setenv("TLLAMA_JANITOR_INTERVAL", "0")

        with pytest.raises(ConfigError) as excinfo:
            load_daemon_config(False, discover=False).backend

        assert "> 0" in str(excinfo.value)


class TestTheEdgeOfEachRangeStillMeansSomething:
    """A bound one step too strict is as wrong as no bound at all."""

    def test_zero_context_length_still_means_the_model_maximum(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_CONTEXT_LENGTH", "0")

        assert load_daemon_config(False, discover=False).backend.context_length == 0

    def test_zero_cache_lifetime_still_means_do_not_cache(self, monkeypatch):
        # backend compares `age_seconds > ttl`, which is true for every
        # entry at 0, so the in-memory cache is off. That is a thing
        # somebody may want.
        monkeypatch.setenv("TLLAMA_METADATA_CACHE_TTL", "0")

        assert load_daemon_config(False, discover=False).backend.metadata_cache_ttl_seconds == 0.0

    def test_one_resident_model_is_the_default_and_still_allowed(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_MAX_LOADED_MODELS", "1")

        assert load_daemon_config(False, discover=False).backend.max_loaded_models == 1

    def test_a_small_interval_is_allowed_because_only_zero_is_the_problem(
        self, monkeypatch
    ):
        monkeypatch.setenv("TLLAMA_JANITOR_INTERVAL", "0.001")

        assert load_daemon_config(False, discover=False).backend.janitor_interval_seconds == 0.001


class TestSamplingKeepsItsRanges:
    """Shared validation preserves supported native sampler controls."""

    @pytest.mark.parametrize(
        "variable,value,expected",
        [
            ("TLLAMA_SAMPLING_TOP_K", "0", 0),
            ("TLLAMA_SAMPLING_TOP_P", "1.5", 1.5),
        ],
    )
    def test_supported_sampler_ranges_remain_available(self, monkeypatch, variable, value, expected):
        monkeypatch.setenv(variable, value)

        config = load_daemon_config(False, discover=False).backend

        assert config.sampling_overrides[variable.removeprefix("TLLAMA_SAMPLING_").lower()] == expected
