"""[template] must reach /api/chat and /v1/chat/completions too, not only
/api/generate.

metadata_info["template"] only ever flowed into render_generate_prompt()
(/api/generate's own rendering). /api/chat and /v1/chat/completions resolve
a chat_handler and let it render internally, independent of metadata_info --
this was a real, confirmed gap in the original [template] patch, found while
investigating what vision support (a future step) would need to interact
with correctly.
"""

from unittest.mock import patch

from tllama.backend import _apply_template_override_to_chat_handler
from tllama.helpers.model_toml import parse_model_toml


class FakeLlm:
    def __init__(self, chat_handler=None):
        self.chat_handler = chat_handler

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def detokenize(self, token_ids):
        return {1: b"<bos>", 2: b"<eos>"}[token_ids[0]]


def spec_with_template(template: str | None):
    text = '[llm]\nmodel = "Local/m.gguf"\n'
    if template is not None:
        text += f'\n[template]\njinja = "{template}"\n'
    return parse_model_toml(text)


class TestApplyTemplateOverride:
    def test_sets_chat_handler_when_template_present(self):
        llm = FakeLlm(chat_handler=None)
        spec = spec_with_template("custom template")

        _apply_template_override_to_chat_handler(llm, spec)

        # to_chat_handler() returns an opaque closure in the real library
        # (chat_formatter_to_chat_completion_handler), not the formatter
        # object itself -- confirmed directly in its source. The only
        # honest thing to assert on the result itself is that something
        # callable got attached; what it was built *from* is checked below
        # by patching the constructor instead of inspecting the return value.
        assert llm.chat_handler is not None
        assert callable(llm.chat_handler)

    def test_constructs_jinja2chatformatter_with_the_override_template(self):
        llm = FakeLlm(chat_handler=None)
        spec = spec_with_template("custom template")

        with patch("tllama.backend.Jinja2ChatFormatter") as mock_formatter:
            _apply_template_override_to_chat_handler(llm, spec)

        mock_formatter.assert_called_once_with(
            template="custom template",
            eos_token="<eos>",
            bos_token="<bos>",
            stop_token_ids=[2],
        )
        assert llm.chat_handler is mock_formatter.return_value.to_chat_handler.return_value

    def test_no_template_leaves_chat_handler_untouched(self):
        llm = FakeLlm(chat_handler=None)
        spec = spec_with_template(None)

        _apply_template_override_to_chat_handler(llm, spec)

        assert llm.chat_handler is None

    def test_no_virtual_spec_leaves_chat_handler_untouched(self):
        llm = FakeLlm(chat_handler=None)

        _apply_template_override_to_chat_handler(llm, None)

        assert llm.chat_handler is None

    def test_existing_chat_handler_is_never_clobbered(self):
        # Stand-in for a future [mmproj] vision handler already set via
        # Llama(chat_handler=...) -- must survive this call untouched.
        sentinel = object()
        llm = FakeLlm(chat_handler=sentinel)
        spec = spec_with_template("custom template")

        _apply_template_override_to_chat_handler(llm, spec)

        assert llm.chat_handler is sentinel


def test_load_model_sync_calls_apply_template_override():
    import inspect
    from tllama.backend import ModelManager

    source = inspect.getsource(ModelManager._load_model_sync)

    assert "_apply_template_override_to_chat_handler(" in source
