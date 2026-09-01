"""`/api/show` reports the definition a model was actually built from.

Ollama's modelfile field carries the thing a model was created from, and
the workflow that makes it worth having is fetching it, editing it and
putting it back. TLlama used to synthesise a FROM/TEMPLATE approximation
that could not be edited into anything and did not describe the model.

The parameters field used to be a hardcoded string, returned identically
for every model whether or not any of it was true.
"""

import pytest

from tllama.backend import model_manager
from tllama.helpers.common import format_ollama_parameters


DEFINITION = '''# A comment somebody wrote and expects to still be here.
[llm]
model = "Local/shown.gguf"

[sampling]
temperature = 0.7
stop = ["<|im_end|>"]
'''


@pytest.fixture
def shown_model():
    """A real GGUF, not a placeholder.

    The .toml's own overrides are layered onto the GGUF's metadata, and
    that layering is skipped entirely when the header cannot be read -- so
    a stand-in file would leave the parameters field empty for reasons
    that have nothing to do with what is being tested.
    """
    import gguf

    gguf_path = model_manager.local_models_dir / "shown.gguf"
    toml = model_manager.local_models_dir / "shown.toml"
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(path=str(gguf_path), arch="llama")
    writer.add_uint32("llama.context_length", 4096)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()

    gguf = gguf_path
    toml.write_text(DEFINITION, encoding="utf-8")
    try:
        yield "shown"
    finally:
        gguf.unlink(missing_ok=True)
        toml.unlink(missing_ok=True)
        model_manager._invalidate_metadata_cache_entry("shown")


class TestTheModelfileField:
    def test_it_is_the_definition_itself(self, client, shown_model):
        response = client.post("/api/show", json={"model": shown_model})

        assert response.json()["modelfile"] == DEFINITION

    def test_comments_survive(self, client, shown_model):
        """The round trip only works if what comes back can go back in.
        Comments are the first thing a parse-and-re-emit would lose, and
        keeping them is why this format was chosen over an INI."""
        response = client.post("/api/show", json={"model": shown_model})

        assert "somebody wrote" in response.json()["modelfile"]

    def test_it_no_longer_invents_a_from_line(self, client, shown_model):
        response = client.post("/api/show", json={"model": shown_model})

        assert not response.json()["modelfile"].startswith("FROM ")

    def test_what_comes_back_can_be_written_straight_back(self, client, shown_model):
        """The whole point, stated as the workflow it enables."""
        from tllama.helpers.model_toml import parse_model_toml

        response = client.post("/api/show", json={"model": shown_model})

        spec = parse_model_toml(response.json()["modelfile"])

        assert spec.llm_model == "Local/shown.gguf"


class TestTheParametersField:
    def test_it_reports_what_the_definition_pins(self, client, shown_model):
        response = client.post("/api/show", json={"model": shown_model})
        parameters = response.json()["parameters"]

        assert "temperature" in parameters
        assert "0.7" in parameters

    def test_a_stop_string_is_quoted(self, client, shown_model):
        response = client.post("/api/show", json={"model": shown_model})

        assert '"<|im_end|>"' in response.json()["parameters"]

    def test_the_old_hardcoded_answer_is_gone(self, client, shown_model):
        """It claimed every model stopped on <|end_of_text|>, which was
        true of almost none of them."""
        response = client.post("/api/show", json={"model": shown_model})

        assert "<|end_of_text|>" not in response.json()["parameters"]


class TestParameterFormatting:
    def test_a_definition_that_pins_nothing_reports_nothing(self):
        """Ollama's field carries the PARAMETER lines a definition set.
        Filling it with TLlama's own baseline would claim the model asked
        for values it never mentioned, and no client could tell which."""
        assert format_ollama_parameters({}) == ""

    def test_numbers_are_unquoted_and_strings_are_quoted(self):
        formatted = format_ollama_parameters({
            "sampling_defaults": {"top_k": 20},
            "stop_defaults": ["END"],
        })

        assert "top_k                          20" in formatted
        assert 'stop                           "END"' in formatted

    def test_every_stop_string_gets_its_own_line(self):
        formatted = format_ollama_parameters({"stop_defaults": ["A", "B"]})

        assert len(formatted.splitlines()) == 2

    def test_the_name_column_matches_ollamas_width(self):
        """Only matters because anything printing this verbatim, which is
        what `ollama show --parameters` does, would look subtly wrong next
        to the real thing."""
        line = format_ollama_parameters({"stop_defaults": ["X"]})

        assert line.index('"X"') == 31

    def test_none_is_tolerated(self):
        assert format_ollama_parameters(None) == ""


class TestNothingElseChanged:
    def test_the_template_field_still_carries_the_template(self, client, shown_model):
        response = client.post("/api/show", json={"model": shown_model})

        assert "template" in response.json()

    def test_an_absent_model_is_still_a_404(self, client):
        response = client.post("/api/show", json={"model": "not-here-at-all"})

        assert response.status_code == 404
