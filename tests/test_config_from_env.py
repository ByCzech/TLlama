"""A misspelt environment variable stops the server instead of vanishing.

Every parser here used to return its default when the value would not
convert, so TLLAMA_CONTEXT_LENGTH=abc came up as 0 and the only evidence
was that the setting had no effect. These pin the refusal, and equally
pin that an unset or empty variable still means "use the default" -- that
part is not an error and must keep working.
"""

import pytest

from tllama.config import (
    ConfigError,
    load_app_config_from_env,
    load_backend_config_from_env,
)


class TestUnsetAndEmptyStillMeanDefault:
    def test_nothing_set_gives_the_declared_defaults(self, monkeypatch):
        for name in (
            "TLLAMA_MODELS",
            "TLLAMA_CONTEXT_LENGTH",
            "TLLAMA_MAX_LOADED_MODELS",
            "TLLAMA_JANITOR_INTERVAL",
            "TLLAMA_FLASH_ATTENTION",
            "TLLAMA_KV_CACHE_TYPE",
        ):
            monkeypatch.delenv(name, raising=False)

        config = load_backend_config_from_env()

        assert config.context_length == 0
        assert config.max_loaded_models == 1
        assert config.flash_attention is False
        assert config.kv_cache_type is None

    def test_an_empty_value_is_not_an_error(self, monkeypatch):
        # A systemd unit with `Environment=TLLAMA_CONTEXT_LENGTH=` sets the
        # variable to the empty string. That says "I set nothing", not "I
        # set nonsense", and must not stop the server.
        monkeypatch.setenv("TLLAMA_CONTEXT_LENGTH", "")
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", "   ")

        config = load_backend_config_from_env()

        assert config.context_length == 0
        assert config.flash_attention is False


class TestAValueThatWillNotParseIsRefused:
    def test_an_integer_variable_rejects_a_non_number(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_CONTEXT_LENGTH", "8k")

        with pytest.raises(ConfigError) as excinfo:
            load_backend_config_from_env()

        # The message has to carry the variable name: an operator reading
        # it in a journal has no other way to tell which of a dozen
        # settings is the wrong one.
        assert "TLLAMA_CONTEXT_LENGTH" in str(excinfo.value)
        assert "8k" in str(excinfo.value)

    def test_a_float_variable_rejects_a_non_number(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_JANITOR_INTERVAL", "often")

        with pytest.raises(ConfigError, match="TLLAMA_JANITOR_INTERVAL"):
            load_backend_config_from_env()

    def test_a_boolean_variable_rejects_a_word_it_does_not_know(self, monkeypatch):
        # "maybe" used to be silently false, which is the same outcome as
        # "off" and reads as if the setting had been honoured.
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", "maybe")

        with pytest.raises(ConfigError, match="TLLAMA_FLASH_ATTENTION"):
            load_backend_config_from_env()

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_the_accepted_true_spellings_still_work(self, monkeypatch, value):
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", value)

        assert load_backend_config_from_env().flash_attention is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "OFF"])
    def test_the_accepted_false_spellings_still_work(self, monkeypatch, value):
        monkeypatch.setenv("TLLAMA_FLASH_ATTENTION", value)

        assert load_backend_config_from_env().flash_attention is False

    def test_a_host_with_an_unparseable_port_is_refused(self, monkeypatch):
        # This one was the most misleading of the lot: the port failed to
        # parse and the host was thrown away with it, so a server asked to
        # listen on 0.0.0.0 came up on 127.0.0.1 instead.
        monkeypatch.setenv("TLLAMA_HOST", "0.0.0.0:http")

        with pytest.raises(ConfigError, match="TLLAMA_HOST"):
            load_app_config_from_env()

    def test_a_host_without_a_port_still_takes_the_default_port(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_HOST", "0.0.0.0")

        config = load_app_config_from_env()

        assert config.host == "0.0.0.0"
        assert config.port == 54800
