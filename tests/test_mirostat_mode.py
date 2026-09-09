"""mirostat_mode is the one sampling value with a set of legal spellings.

llama-cpp-python's _init_sampler() reads it as `if == 1 ... elif == 2 ...
else`, so 3 does not fail and does not clamp: it falls through to the
ordinary sampler chain, and mirostat is off. Somebody who wrote it asked
for mirostat and got none, with nothing in the log about it.

Both places a person can configure it have to agree, which is why the
judgement sits in helpers/common.py and neither caller owns a copy.
"""

import pytest

from tllama.config import ConfigError
from tllama.errors import TomlModelError
from tllama.helpers.common import build_sampling_kwargs
from tllama.helpers.model_toml import parse_model_toml
from tllama.helpers.sampling_params import coerce_sampling_overrides


def _toml(value: str) -> str:
    return f'[llm]\nmodel = "Local/m.gguf"\n\n[sampling]\nmirostat_mode = {value}\n'


class TestAModeTheSamplerDoesNotHaveIsRefused:
    @pytest.mark.parametrize("value", ["3", "99", "-1"])
    def test_a_toml_says_which_file_and_which_value(self, value):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml(value), source="Local/m.toml")

        message = str(excinfo.value)
        assert "Local/m.toml" in message
        assert "mirostat_mode" in message
        assert value in message

    @pytest.mark.parametrize("value", ["3", "99", "-1"])
    def test_the_environment_says_which_variable(self, value):
        with pytest.raises(ConfigError) as excinfo:
            coerce_sampling_overrides({"mirostat_mode": value})

        assert "TLLAMA_SAMPLING_MIROSTAT_MODE" in str(excinfo.value)

    def test_true_is_not_mode_one(self):
        # TOML has a real boolean and bool is a subclass of int, so
        # `mirostat_mode = true` would otherwise be accepted as 1 and read
        # as if it had been meant.
        with pytest.raises(TomlModelError, match="bool"):
            parse_model_toml(_toml("true"), source="Local/m.toml")

    def test_the_message_says_what_the_value_would_have_done(self):
        # "must be one of 0, 1, 2" alone does not tell an operator that
        # the server would otherwise have run with mirostat off.
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml("3"), source="Local/m.toml")

        assert "turns mirostat off" in str(excinfo.value)


class TestTheModesThatExistStillWork:
    @pytest.mark.parametrize("value", [0, 1, 2])
    def test_a_toml_accepts_every_real_mode(self, value):
        spec = parse_model_toml(_toml(str(value)), source="Local/m.toml")

        assert spec.sampling["mirostat_mode"] == value

    @pytest.mark.parametrize("value", ["0", "1", "2"])
    def test_the_environment_accepts_every_real_mode(self, value):
        assert coerce_sampling_overrides({"mirostat_mode": value}) == {
            "mirostat_mode": int(value)
        }

    def test_an_accepted_mode_still_travels_the_whole_chain(self):
        # A check that refused nothing legal but also stopped the value
        # reaching the sampler would pass the tests above.
        kwargs = build_sampling_kwargs(
            {}, {"sampling_defaults": {"mirostat_mode": 2}}, {"mirostat_mode": 1}
        )

        assert kwargs["mirostat_mode"] == 2


class TestNothingElseGainedARange:
    """The neighbours stay unbounded, and on purpose.

    top_k at or below zero disables top-k, top_p at or above 1.0 disables
    top-p, and a negative temperature selects a sampler chain of add_dist
    alone. These are modes llama.cpp defines, not mistakes.
    """

    @pytest.mark.parametrize(
        "key,value,expected",
        [
            ("temperature", "-1", -1.0),
            ("top_k", "0", 0),
            ("top_p", "1.5", 1.5),
            ("min_p", "-0.5", -0.5),
            ("repeat_penalty", "0.0", 0.0),
        ],
    )
    def test_the_environment_still_takes_them(self, key, value, expected):
        assert coerce_sampling_overrides({key: value}) == {key: expected}

    @pytest.mark.parametrize(
        "line", ["temperature = -1.0", "top_k = 0", "top_p = 1.5"]
    )
    def test_a_toml_still_takes_them(self, line):
        text = f'[llm]\nmodel = "Local/m.gguf"\n\n[sampling]\n{line}\n'

        assert parse_model_toml(text, source="Local/m.toml").sampling
