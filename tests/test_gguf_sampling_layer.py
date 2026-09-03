"""GGUF general.sampling.* as a layer, not just something reported.

The reading side landed long ago: metadata carries recommended_sampling
and /api/show reports it. What is tested here is that it now decides
something, and that it decides it in the right place -- which is only
visible against the layers immediately above and below.
"""

from tllama.helpers.common import build_sampling_kwargs
from tllama.helpers.gguf_metadata import GGUF_SAMPLING_KEYS, build_model_metadata_payload


def _recommending(**values) -> dict:
    return {"recommended_sampling": values}


class TestItApplies:
    def test_a_recommended_value_beats_the_baseline(self):
        kwargs = build_sampling_kwargs({}, _recommending(temperature=1.0))

        assert kwargs["temperature"] == 1.0

    def test_a_model_recommending_nothing_keeps_the_baseline(self):
        kwargs = build_sampling_kwargs({}, {"recommended_sampling": {}})

        assert kwargs["temperature"] == 0.8
        assert kwargs["top_k"] == 40

    def test_recommending_one_key_leaves_the_others_on_the_baseline(self):
        """The four real files carrying this set temp, top_k and top_p in
        varying combinations; a partial header must not drag the rest."""
        kwargs = build_sampling_kwargs({}, _recommending(top_k=20))

        assert kwargs["top_k"] == 20
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9


class TestWhereItSits:
    def test_the_environment_beats_it(self):
        kwargs = build_sampling_kwargs({}, _recommending(temperature=1.0), {"temperature": 0.3})

        assert kwargs["temperature"] == 0.3

    def test_a_toml_beats_it(self):
        metadata = _recommending(temperature=1.0)
        metadata["sampling_defaults"] = {"temperature": 0.5}

        assert build_sampling_kwargs({}, metadata)["temperature"] == 0.5

    def test_a_request_beats_it(self):
        kwargs = build_sampling_kwargs({"temperature": 0.9}, _recommending(temperature=1.0))

        assert kwargs["temperature"] == 0.9

    def test_it_beats_the_baseline_but_loses_to_everything_above(self):
        """The whole chain in one place, so the ordering cannot be right in
        pairs and wrong overall."""
        metadata = _recommending(temperature=1.0, top_k=20, top_p=0.95)
        metadata["sampling_defaults"] = {"top_k": 30}

        kwargs = build_sampling_kwargs(
            {"top_p": 0.5}, metadata, {"temperature": 0.3}
        )

        assert kwargs["top_p"] == 0.5  # request
        assert kwargs["top_k"] == 30  # .toml
        assert kwargs["temperature"] == 0.3  # environment
        assert kwargs["min_p"] == 0.05  # baseline, nothing said anything


class TestAgainstTheReadingSide:
    def test_the_names_the_reader_produces_are_names_the_chain_consumes(self):
        """The reader maps GGUF's spellings onto TLlama's. If that mapping
        produced a name the chain never looks at, every test above would
        still pass while nothing happened on a real model."""
        payload = build_model_metadata_payload(
            {gguf_key: 0.5 for gguf_key in GGUF_SAMPLING_KEYS}
        )

        recommended = payload["recommended_sampling"]
        assert recommended

        kwargs = build_sampling_kwargs({}, payload)
        for name in recommended:
            assert kwargs[name] == 0.5, name
