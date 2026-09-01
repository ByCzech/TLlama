"""[mmproj] reaching Llama() as a chat handler, so a projector is loaded.

A definition naming a projector used to be checked and then dropped: the
resolved path went nowhere and the model loaded blind. These cover the
handler actually being built and handed to Llama(), and the [template]
rule surviving the trip -- MTMDChatHandler renders the template itself,
so a projector alongside a [template] would otherwise quietly go back to
the GGUF's own.
"""

import gguf
import pytest

from llama_cpp.llama_chat_format import MTMDChatHandler

from tllama.backend import _TemplateOverridingMTMDChatHandler
from tllama.helpers.model_toml import parse_model_toml


@pytest.fixture
def place_gguf(manager):
    def factory(relative_path, architecture="llama", **keys):
        path = manager.models_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = gguf.GGUFWriter(path=str(path), arch=architecture)
        for key, value in keys.items():
            writer.add_string(key.replace("__", "."), str(value))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        return path

    return factory


@pytest.fixture
def spec_for(manager, place_gguf):
    """A parsed spec whose files really exist in the model store."""
    def factory(body):
        place_gguf("Local/model.gguf")
        place_gguf("Local/mmproj.gguf", architecture="clip")
        return parse_model_toml(body)

    return factory


WITH_PROJECTOR = """[llm]
model = "Local/model.gguf"

[mmproj]
model = "Local/mmproj.gguf"
"""

WITHOUT_PROJECTOR = """[llm]
model = "Local/model.gguf"
"""

TEMPLATE_AND_PROJECTOR = """[llm]
model = "Local/model.gguf"

[mmproj]
model = "Local/mmproj.gguf"

[template]
jinja = "TOML TEMPLATE"
"""

TEMPLATE_ONLY = """[llm]
model = "Local/model.gguf"

[template]
jinja = "TOML TEMPLATE"
"""


class FakeModel:
    """Enough of a Llama for _get_chat_template to read a template off."""

    def __init__(self, template="GGUF TEMPLATE"):
        self.metadata = {"tokenizer.chat_template": template}


class TestTheProjectorReachesLlama:
    def test_a_named_projector_becomes_a_chat_handler(self, manager, spec_for):
        spec = spec_for(WITH_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)

        assert isinstance(kwargs["chat_handler"], MTMDChatHandler)

    def test_the_handler_points_at_the_projector_file(self, manager, spec_for):
        spec = spec_for(WITH_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)

        assert kwargs["chat_handler"].clip_model_path == str(
            manager.models_dir / "Local" / "mmproj.gguf"
        )

    def test_a_model_without_a_projector_gets_no_handler(self, manager, spec_for):
        spec = spec_for(WITHOUT_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)

        # Absent, not None: Llama() resolves its own text handler from the
        # GGUF, and a chat_handler=None passed in would say the same thing
        # but only by accident of that argument's default.
        assert "chat_handler" not in kwargs

    def test_no_virtual_spec_at_all_gets_no_handler(self, manager):
        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, None)

        assert "chat_handler" not in kwargs


class TestTheTomlTemplateOutranksTheGguf:
    def test_a_projector_alongside_a_template_renders_the_toml_one(
        self, manager, spec_for
    ):
        spec = spec_for(TEMPLATE_AND_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)
        handler = kwargs["chat_handler"]

        assert isinstance(handler, _TemplateOverridingMTMDChatHandler)
        assert handler._get_chat_template(FakeModel()) == "TOML TEMPLATE"

    def test_a_projector_without_a_template_still_reads_the_gguf(
        self, manager, spec_for
    ):
        spec = spec_for(WITH_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)
        handler = kwargs["chat_handler"]

        assert not isinstance(handler, _TemplateOverridingMTMDChatHandler)
        assert handler._get_chat_template(FakeModel()) == "GGUF TEMPLATE"

    def test_a_template_without_a_projector_is_left_to_the_old_path(
        self, manager, spec_for
    ):
        # _apply_template_override_to_chat_handler still handles this one
        # after loading; building a projector handler for a model with no
        # projector would be inventing one.
        spec = spec_for(TEMPLATE_ONLY)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)

        assert "chat_handler" not in kwargs
