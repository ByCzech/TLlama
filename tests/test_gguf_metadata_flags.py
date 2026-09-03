"""Reading the GGUF keys that tell a projector, a continuation shard and
an author's recommended sampling apart from an ordinary model.

Every case here writes a real GGUF with gguf-py and reads it back through
read_gguf_metadata(), rather than handing build_model_metadata_payload() a
dictionary someone typed. The keys and their spelling are the thing being
tested; a hand-written dict would only confirm that the code agrees with
itself.

The expected values were taken from real files in a working model store:
a two-part Llama 3.3 70B, a Qwen3.6 mmproj, and 17 models of which only
Qwen3.6 and Gemma 4 carry general.sampling.*.
"""

import gguf
import pytest

from tllama.helpers.gguf_metadata import (
    build_model_metadata_payload,
    read_gguf_metadata,
)


@pytest.fixture
def write_gguf(tmp_path):
    """Write a real GGUF carrying the given keys, and return its path."""
    counter = {"n": 0}

    def factory(architecture="llama", **keys):
        counter["n"] += 1
        path = tmp_path / f"model-{counter['n']}.gguf"

        writer = gguf.GGUFWriter(path=str(path), arch=architecture)
        for key, value in keys.items():
            key = key.replace("__", ".")
            if isinstance(value, bool):
                writer.add_bool(key, value)
            elif isinstance(value, int):
                writer.add_uint32(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            else:
                writer.add_string(key, value)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        return path

    return factory


def payload_for(path):
    return build_model_metadata_payload(read_gguf_metadata(str(path)))


class TestProjectorDetection:
    def test_the_clip_architecture_marks_a_projector(self, write_gguf):
        path = write_gguf(architecture="clip")

        assert payload_for(path)["is_projector"] is True

    def test_general_type_alone_marks_a_projector(self, write_gguf):
        """An older projector may predate general.type, and a newer one may
        conceivably carry it without arch=clip. Either signal on its own is
        enough, because mistaking a projector for a model is the failure
        that matters."""
        path = write_gguf(architecture="qwen3", general__type="mmproj")

        assert payload_for(path)["is_projector"] is True

    def test_the_vision_encoder_flag_alone_marks_a_projector(self, write_gguf):
        path = write_gguf(architecture="qwen3", clip__has_vision_encoder=True)

        assert payload_for(path)["is_projector"] is True

    def test_an_ordinary_model_is_not_a_projector(self, write_gguf):
        path = write_gguf(architecture="llama", general__type="model")

        assert payload_for(path)["is_projector"] is False

    def test_a_model_with_no_general_type_at_all_is_not_a_projector(self, write_gguf):
        path = write_gguf(architecture="llama")

        assert payload_for(path)["is_projector"] is False


class TestShardPosition:
    def test_an_unsplit_file_reports_no_shard_position(self, write_gguf):
        """Absence of both keys is the signal, and it has to stay
        distinguishable from a genuine first shard: defaulting the index to
        0 would make an unsplit file and the first part of a split one look
        identical."""
        payload = payload_for(write_gguf())

        assert payload["shard_index"] is None
        assert payload["shard_count"] is None
        assert payload["is_continuation_shard"] is False

    def test_the_first_shard_is_index_zero_and_not_a_continuation(self, write_gguf):
        path = write_gguf(split__no=0, split__count=2)
        payload = payload_for(path)

        assert payload["shard_index"] == 0
        assert payload["shard_count"] == 2
        assert payload["is_continuation_shard"] is False

    def test_a_later_shard_is_a_continuation(self, write_gguf):
        path = write_gguf(split__no=1, split__count=2)
        payload = payload_for(path)

        assert payload["shard_index"] == 1
        assert payload["is_continuation_shard"] is True

    def test_a_continuation_shard_carrying_nothing_else_still_reads(self, write_gguf):
        """A real second shard carries only split.* and GGUF.* -- not even
        general.architecture. Reading it must not fail, it must simply
        report what little is there."""
        path = write_gguf(split__no=1, split__count=2)
        payload = payload_for(path)

        assert payload["is_continuation_shard"] is True
        assert payload["arch"] in ("unknown", "llama")


class TestRecommendedSampling:
    def test_the_authors_values_are_read_under_our_own_names(self, write_gguf):
        """GGUF spells it temp; build_sampling_kwargs() spells it
        temperature. The translation happens here so no caller has to know
        about the difference."""
        path = write_gguf(
            general__sampling__temp=1.0,
            general__sampling__top_k=20,
            general__sampling__top_p=0.95,
        )
        recommended = payload_for(path)["recommended_sampling"]

        assert recommended["temperature"] == pytest.approx(1.0)
        assert recommended["top_k"] == 20
        assert recommended["top_p"] == pytest.approx(0.95)

    def test_a_file_without_them_reports_an_empty_mapping(self, write_gguf):
        """Only a minority of real files carry these. Their absence is
        ordinary and must not look like an error."""
        assert payload_for(write_gguf())["recommended_sampling"] == {}

    def test_a_partial_set_reports_only_what_is_there(self, write_gguf):
        path = write_gguf(general__sampling__top_k=64)
        recommended = payload_for(path)["recommended_sampling"]

        assert recommended == {"top_k": 64}

    def test_they_now_decide_something(self, write_gguf):
        """They are applied, not only reported -- from a real header on
        disk through the metadata payload to the kwargs a completion call
        gets, which is the only path that proves the names line up.

        This assertion used to be the opposite: build_sampling_kwargs()
        gaining a tier changes what inference does, so it was held back
        for a patch of its own.
        """
        from tllama.helpers.common import build_sampling_kwargs

        path = write_gguf(general__sampling__temp=1.0)
        metadata_info = payload_for(path)

        assert build_sampling_kwargs({}, metadata_info)["temperature"] == 1.0

    def test_a_configured_value_still_wins_over_the_header(self, write_gguf):
        """Applying the header must not have put it above the layers a
        person controls."""
        from tllama.helpers.common import build_sampling_kwargs

        path = write_gguf(general__sampling__temp=1.0)
        metadata_info = payload_for(path)

        kwargs = build_sampling_kwargs({}, metadata_info, {"temperature": 0.3})

        assert kwargs["temperature"] == 0.3


class TestExistingPayloadIsUnchanged:
    def test_the_keys_that_were_there_before_are_still_there(self, write_gguf):
        path = write_gguf(architecture="llama", general__name="Something")
        payload = payload_for(path)

        for key in (
            "arch", "params", "parameter_size", "size_label", "bits",
            "template", "context_length", "display_name", "metadata_raw",
        ):
            assert key in payload

    def test_the_architecture_still_reads_correctly(self, write_gguf):
        assert payload_for(write_gguf(architecture="gemma4"))["arch"] == "gemma4"
