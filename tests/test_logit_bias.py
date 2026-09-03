"""[sampling] logit_bias: from a .toml to the completion call and to show.

logit_bias differs from every other sampling parameter in two ways that
these tests are about. It has no baseline, so "the .toml said nothing" has
to stay nothing rather than becoming a value; and its TOML shape (a table
keyed by strings) is not the shape create_completion() takes (a dict keyed
by ints), so something has to convert it.
"""

import pytest

from tllama.helpers.common import (
    build_sampling_kwargs,
    format_ollama_parameters,
    sampling_parameter_names,
)
from tllama.helpers.model_toml import TomlModelError, parse_model_toml


def _toml(sampling_body: str) -> str:
    return (
        '[llm]\nmodel = "Local/model.gguf"\n\n'
        f"[sampling]\n{sampling_body}\n"
    )


class TestParsing:
    def test_table_becomes_int_keyed_floats(self):
        spec = parse_model_toml(
            _toml('logit_bias = { "128009" = -100.0, "42" = 1 }'),
            source="model.toml",
        )

        assert spec.sampling["logit_bias"] == {128009: -100.0, 42: 1.0}

    def test_key_types_are_int_not_str(self):
        """create_completion's signature says Dict[int, float]. A table's
        keys are strings, so a parse that merely passed them through would
        look right in a diff and be wrong at the call."""
        spec = parse_model_toml(_toml('logit_bias = { "7" = 2.5 }'), source="m.toml")

        (key,) = spec.sampling["logit_bias"]
        assert isinstance(key, int)
        assert isinstance(spec.sampling["logit_bias"][key], float)

    def test_absent_stays_absent(self):
        spec = parse_model_toml(_toml("temperature = 0.5"), source="m.toml")

        assert "logit_bias" not in spec.sampling

    def test_non_integer_key_is_refused_naming_the_file(self):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml('logit_bias = { "eos" = -1.0 }'), source="m.toml")

        message = str(excinfo.value)
        assert "m.toml" in message
        assert "logit_bias" in message
        assert "eos" in message

    def test_non_numeric_value_is_refused(self):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml('logit_bias = { "7" = "lots" }'), source="m.toml")

        assert "logit_bias" in str(excinfo.value)
        assert "7" in str(excinfo.value)

    def test_boolean_value_is_refused_rather_than_read_as_one(self):
        """bool is an int subclass, so a bare true would otherwise arrive
        as a bias of 1.0 and look like somebody meant it."""
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml('logit_bias = { "7" = true }'), source="m.toml")

        assert "bool" in str(excinfo.value)

    def test_wrong_shape_entirely_is_refused(self):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(_toml("logit_bias = 3"), source="m.toml")

        assert "table" in str(excinfo.value)


class TestAcceptedAsAName:
    def test_it_is_a_name_a_toml_may_use(self):
        assert "logit_bias" in sampling_parameter_names()

    def test_grammar_is_still_refused(self):
        """Same gap in llama-cpp-python, deliberately not closed: invalid
        GBNF kills the process instead of raising."""
        assert "grammar" not in sampling_parameter_names()

        with pytest.raises(TomlModelError):
            parse_model_toml(_toml('grammar = \'root ::= "a"\''), source="m.toml")


class TestReachesTheCall:
    def test_a_toml_value_is_passed_through(self):
        kwargs = build_sampling_kwargs({}, {"sampling_defaults": {"logit_bias": {7: -1.0}}})

        assert kwargs["logit_bias"] == {7: -1.0}

    def test_nothing_set_passes_no_argument_at_all(self):
        """Not None, not {} -- omitted. An empty dict is a claim that
        somebody asked for no biases, which is not the same as silence."""
        kwargs = build_sampling_kwargs({}, {})

        assert "logit_bias" not in kwargs

    def test_a_baselined_parameter_is_still_always_present(self):
        """Guards the distinction: the no-baseline handling must not have
        made the ordinary parameters conditional too."""
        kwargs = build_sampling_kwargs({}, {})

        assert kwargs["temperature"] == 0.8


class TestShow:
    def test_it_is_reported_as_json_not_python_repr(self):
        """format_ollama_parameters renders values with str() otherwise,
        which on a dict gives single quotes and nothing readable back."""
        rendered = format_ollama_parameters(
            {"sampling_defaults": {"logit_bias": {7: -1.0}}}
        )

        assert '{"7": -1.0}' in rendered
        assert "'" not in rendered
