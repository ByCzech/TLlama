"""A section or key in a .toml that nothing reads must be refused.

[mmproj], [runtime] and [sampling] each say so already, and each for its
own reason. What none of them covers is the name of the section itself,
or the contents of [llm], [template] and [system].

A misspelt key loses one setting. A misspelt section name loses every
setting in it at once, which makes it the worse of the two and the
harder to notice: the file still parses, the model still loads, and
nothing anywhere reports that a whole block was read and dropped.
"""

import pytest

from tllama.helpers.model_toml import TomlModelError, parse_model_toml


LLM = '[llm]\nmodel = "Local/model.gguf"\n'


class TestSectionNames:
    def test_a_misspelt_section_name_is_refused(self):
        """[smapling] is a typo away from [sampling] and silently empties it."""
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(
                LLM + '\n[smapling]\ntemperature = 0.7\n', source="m.toml"
            )

        assert "smapling" in str(excinfo.value)
        assert "m.toml" in str(excinfo.value)

    def test_every_known_section_together_is_accepted(self):
        """The check must not refuse the vocabulary it exists to protect."""
        spec = parse_model_toml(
            LLM
            + '\n[mmproj]\nmodel = "Local/mmproj.gguf"\n'
            + '\n[runtime]\nn_ctx = 8192\n'
            + '\n[sampling]\ntemperature = 0.7\n'
            + '\n[template]\njinja = "{{ x }}"\n'
            + '\n[system]\nprompt = "be brief"\n'
        )

        assert spec.runtime["n_ctx"] == 8192
        assert spec.sampling["temperature"] == 0.7
        assert spec.template == "{{ x }}"
        assert spec.system_prompt == "be brief"

    def test_a_section_planned_but_not_implemented_is_refused(self):
        """[lora] read and dropped means a model running without its adapter.

        Silence there is a wrong answer rather than a missing feature, so
        it is refused until something reads it.
        """
        with pytest.raises(TomlModelError):
            parse_model_toml(LLM + '\n[lora]\npath = "Local/a.gguf"\n')

    def test_a_bare_key_outside_any_section_is_refused(self):
        """'model = ...' written above [llm] belongs to no section at all."""
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml('nonsense = 1\n' + LLM)

        assert "nonsense" in str(excinfo.value)


class TestSectionShape:
    def test_a_known_section_written_as_an_array_of_tables_is_refused(self):
        """[[runtime]] is one bracket away from [runtime].

        It used to raise a bare ValueError out of dict(), which names no
        file and, worse, is caught as 'this is not a virtual model' by
        get_model_metadata -- so the whole spec vanished rather than the
        one typo being reported.
        """
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(LLM + '\n[[runtime]]\nn_ctx = 8192\n', source="m.toml")

        assert "runtime" in str(excinfo.value)
        assert "m.toml" in str(excinfo.value)

    def test_a_known_section_name_used_as_a_scalar_is_refused(self):
        with pytest.raises(TomlModelError):
            parse_model_toml('sampling = 1\n' + LLM)


class TestLlmKeys:
    def test_an_unread_llm_key_is_refused(self):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(LLM + 'nonsense = 1\n', source="m.toml")

        assert "nonsense" in str(excinfo.value)
        assert "m.toml" in str(excinfo.value)

    def test_a_nested_table_under_llm_is_refused(self):
        with pytest.raises(TomlModelError):
            parse_model_toml(LLM + '\n[llm.extra]\na = 1\n')


class TestTemplateAndSystemKeys:
    def test_a_misspelt_template_key_is_refused(self):
        """'jinj' leaves template None while the file looks like it set it."""
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(LLM + '\n[template]\njinj = "{{ x }}"\n')

        assert "jinj" in str(excinfo.value)

    def test_a_misspelt_system_key_is_refused(self):
        with pytest.raises(TomlModelError) as excinfo:
            parse_model_toml(LLM + '\n[system]\nprmopt = "be brief"\n')

        assert "prmopt" in str(excinfo.value)
