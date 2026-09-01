"""Generating a virtual-model .toml, and writing it to disk safely.

The output is meant to be read and edited by a person, so what is tested
here is not only that it parses but that it says the right things: active
where it must be, commented where the model merely has an opinion, and
silent where the model said nothing at all.

Round-tripped through parse_model_toml() throughout, because a generated
file that the reader rejects is worse than no file: it would list as a
broken model rather than as no model.
"""

import pytest

from tllama.helpers.model_toml import (
    TomlModelError,
    parse_model_toml,
    render_model_toml,
    write_model_toml,
)


MODEL = "HuggingFace/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"

FULL_METADATA = {
    "context_length": 262144,
    "recommended_sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071},
}


def active_keys(text):
    """What the file actually sets, as opposed to what it mentions."""
    return parse_model_toml(text)


class TestTheActivePart:
    def test_the_model_path_is_the_one_thing_that_is_set(self):
        spec = active_keys(render_model_toml(MODEL))

        assert spec.llm_model == MODEL
        assert spec.llm_from is None

    def test_a_file_with_no_metadata_at_all_is_still_valid(self):
        """Metadata is unavailable often enough -- an unreadable header, a
        scan that timed out -- that it cannot be a precondition for
        producing a usable file."""
        spec = active_keys(render_model_toml(MODEL))

        assert spec.llm_model == MODEL

    def test_nothing_from_the_gguf_is_active(self):
        """The whole point of commenting them: the value in effect stays
        whatever the request, the configuration and the baseline decide,
        exactly as if the lines were absent."""
        spec = active_keys(render_model_toml(MODEL, metadata=FULL_METADATA))

        assert spec.runtime == {}
        assert spec.sampling == {}
        assert spec.stop == []

    def test_an_mmproj_path_is_active_when_given(self):
        """Unlike the GGUF's opinions, a projector is a decision already
        made by whoever asked for this file."""
        text = render_model_toml(MODEL, mmproj_path="HuggingFace/ns/repo/mmproj.gguf")
        spec = active_keys(text)

        assert spec.mmproj_model == "HuggingFace/ns/repo/mmproj.gguf"

    def test_no_mmproj_section_appears_when_there_is_none(self):
        spec = active_keys(render_model_toml(MODEL))

        assert spec.mmproj_model is None


class TestWhatTheCommentsSay:
    def test_the_context_length_is_offered(self):
        text = render_model_toml(MODEL, metadata=FULL_METADATA)

        assert "# n_ctx = 262144" in text

    def test_the_authors_sampling_is_offered(self):
        text = render_model_toml(MODEL, metadata=FULL_METADATA)

        assert "# temperature = 1.0" in text
        assert "# top_k = 20" in text

    def test_a_float32_artefact_is_rounded_for_the_reader(self):
        """0.95 stored as float32 reads back as 0.949999988079071. That is
        a property of the storage, not something the author said, and this
        line exists to be uncommented by a person."""
        text = render_model_toml(MODEL, metadata=FULL_METADATA)

        assert "# top_p = 0.95" in text
        assert "0.949999988079071" not in text

    def test_uncommenting_a_line_actually_works(self):
        """The commented lines are only useful if they are valid where
        they sit -- a comment nobody can act on is decoration."""
        text = render_model_toml(MODEL, metadata=FULL_METADATA)
        uncommented = text.replace("# n_ctx", "n_ctx").replace("# temperature", "temperature")

        spec = parse_model_toml(uncommented)

        assert spec.runtime["n_ctx"] == 262144
        assert spec.sampling["temperature"] == 1.0


class TestNothingIsInvented:
    def test_a_missing_context_length_produces_no_runtime_section(self):
        """An absent line has to mean the model did not say, never that
        the generator had nothing to put there."""
        text = render_model_toml(MODEL, metadata={"context_length": 0})

        assert "[runtime]" not in text

    def test_a_model_without_recommended_sampling_gets_no_sampling_section(self):
        """Only 4 of 17 real files carry general.sampling.*; the other 13
        must not get a section full of invented numbers."""
        text = render_model_toml(MODEL, metadata={"recommended_sampling": {}})

        assert "[sampling]" not in text

    def test_a_partial_recommendation_offers_only_what_is_there(self):
        text = render_model_toml(
            MODEL, metadata={"recommended_sampling": {"top_k": 64}}
        )

        assert "# top_k = 64" in text
        assert "temperature" not in text

    def test_the_chat_template_is_never_written(self):
        """Available, but 2 to 17 kB in practice -- not something to put in
        a file a person is meant to read."""
        text = render_model_toml(
            MODEL, metadata={"template": "{% for m in messages %}{{ m }}{% endfor %}"}
        )

        assert "[template]" not in text
        assert "endfor" not in text


class TestQuoting:
    def test_a_windows_style_path_survives(self):
        """Backslashes and quotes have to be escaped, not pasted in raw,
        or the file will not parse back."""
        text = render_model_toml(r"Local/odd\name.gguf")

        assert parse_model_toml(text).llm_model == r"Local/odd\name.gguf"

    def test_a_path_containing_a_quote_survives(self):
        text = render_model_toml('Local/it"s.gguf')

        assert parse_model_toml(text).llm_model == 'Local/it"s.gguf'


class TestWriting:
    def test_the_file_lands_where_asked(self, tmp_path):
        path = write_model_toml(tmp_path / "m.toml", render_model_toml(MODEL))

        assert path.read_text(encoding="utf-8").startswith("# TLlama virtual model.")

    def test_missing_directories_are_created(self, tmp_path):
        target = tmp_path / "HuggingFace" / "ns" / "repo" / "m.toml"

        write_model_toml(target, render_model_toml(MODEL))

        assert target.is_file()

    def test_an_existing_file_is_not_clobbered(self, tmp_path):
        """A .toml is a file a person edits. Replacing one silently would
        throw their work away."""
        target = tmp_path / "m.toml"
        target.write_text("# hand written\n", encoding="utf-8")

        with pytest.raises(FileExistsError):
            write_model_toml(target, render_model_toml(MODEL))

        assert target.read_text(encoding="utf-8") == "# hand written\n"

    def test_overwriting_is_possible_when_asked_for(self, tmp_path):
        target = tmp_path / "m.toml"
        target.write_text("# hand written\n", encoding="utf-8")

        write_model_toml(target, render_model_toml(MODEL), overwrite=True)

        assert "[llm]" in target.read_text(encoding="utf-8")

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        """The write goes through a temporary in the same directory so the
        rename is atomic. The scan runs on every listing and would report a
        stray one as a model."""
        write_model_toml(tmp_path / "m.toml", render_model_toml(MODEL))

        assert [p.name for p in tmp_path.iterdir()] == ["m.toml"]

    def test_a_reader_never_sees_a_partial_file(self, tmp_path):
        """Whatever is at the path is either absent or complete, because
        the content arrives by rename rather than by being written in
        place."""
        target = tmp_path / "m.toml"
        write_model_toml(target, render_model_toml(MODEL, metadata=FULL_METADATA))

        assert parse_model_toml(target.read_text(encoding="utf-8")).llm_model == MODEL
