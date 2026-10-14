"""Native generate rendering preserves system and stop defaults."""

from types import SimpleNamespace

from tllama.helpers.prompt_render import render_generate_text


SIMPLE_TEMPLATE = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"




def make_request(**kwargs):
    defaults = {"prompt": "hi", "system": None, "options": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestStopDefaultsFallback:
    def test_client_stop_wins_over_toml(self):
        request = make_request(options={"stop": ["CLIENT"]})
        metadata_info = {"template": SIMPLE_TEMPLATE, "stop_defaults": ["TOML"]}

        _, stop = render_generate_text(metadata_info, request, bos_token="", eos_token="", mode="prompt")

        assert "CLIENT" in stop
        assert "TOML" not in stop

    def test_toml_stop_used_when_client_silent(self):
        request = make_request(options=None)
        metadata_info = {"template": SIMPLE_TEMPLATE, "stop_defaults": ["TOML"]}

        _, stop = render_generate_text(metadata_info, request, bos_token="", eos_token="", mode="prompt")

        assert "TOML" in stop

    def test_neither_set_leaves_stop_empty(self):
        request = make_request(options=None)
        metadata_info = {"template": SIMPLE_TEMPLATE}

        _, stop = render_generate_text(metadata_info, request, bos_token="", eos_token="", mode="prompt")

        assert stop == []
