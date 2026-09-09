"""[mmproj] settings reaching mtmd_context_params before the context is built.

_init_mtmd_context sets four of the structure's fields and leaves the
rest at their C defaults, so warmup, image_min_tokens, image_max_tokens
and batch_max_tokens could not be given a value at all. warmup in
particular defaults to reserving a compute graph for a full-size image,
and that graph grows with the square of the image token count.

The settings are written by standing in front of the one call
_init_mtmd_context makes for its defaults, so these cover both halves:
that the values arrive, and that they are noticed as missing if that call
ever stops happening.
"""

import gguf
import pytest

import llama_cpp.mtmd_cpp

from tllama.backend import _initialise_projector
from tllama.helpers.model_toml import TomlModelError, parse_model_toml


WITH_CONTEXT = """[llm]
model = "Local/model.gguf"

[mmproj]
model = "Local/mmproj.gguf"
warmup = false
image_min_tokens = 1024
image_max_tokens = 2048
batch_max_tokens = 4096
"""

PLAIN_PROJECTOR = """[llm]
model = "Local/model.gguf"

[mmproj]
model = "Local/mmproj.gguf"
"""


@pytest.fixture
def spec_for(manager):
    def factory(body):
        for relative, arch in (
            ("Local/model.gguf", "llama"),
            ("Local/mmproj.gguf", "clip"),
        ):
            path = manager.models_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = gguf.GGUFWriter(path=str(path), arch=arch)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.close()

        return parse_model_toml(body)

    return factory


class TestParsing:
    def test_all_four_are_read(self):
        spec = parse_model_toml(WITH_CONTEXT)

        assert spec.mmproj_context == {
            "warmup": False,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "batch_max_tokens": 4096,
        }

    def test_a_section_without_them_carries_nothing(self):
        spec = parse_model_toml(PLAIN_PROJECTOR)

        assert spec.mmproj_context == {}

    def test_the_paths_do_not_leak_into_the_context(self):
        # model and from go to mtmd_init_from_file, not into the params
        # structure, and a field named 'model' does not exist there.
        spec = parse_model_toml(WITH_CONTEXT)

        assert "model" not in spec.mmproj_context

    def test_a_number_where_a_boolean_belongs_is_refused(self):
        # TOML has a real boolean and bool is a subclass of int, so
        # warmup = 1 would otherwise pass every isinstance check and mean
        # something the file never said.
        with pytest.raises(TomlModelError, match="warmup"):
            parse_model_toml(PLAIN_PROJECTOR + "warmup = 1\n")

    def test_a_boolean_where_a_number_belongs_is_refused(self):
        with pytest.raises(TomlModelError, match="image_max_tokens"):
            parse_model_toml(PLAIN_PROJECTOR + "image_max_tokens = true\n")

    def test_a_string_is_refused(self):
        with pytest.raises(TomlModelError, match="image_min_tokens"):
            parse_model_toml(PLAIN_PROJECTOR + 'image_min_tokens = "1024"\n')

    def test_a_negative_value_is_kept(self):
        # -1 is what mtmd itself uses for 'no custom limit', so it is a
        # value with a meaning rather than a number to bound away.
        spec = parse_model_toml(PLAIN_PROJECTOR + "image_max_tokens = -1\n")

        assert spec.mmproj_context["image_max_tokens"] == -1


class TestTheSettingsReachTheStructure:
    def test_a_projector_with_no_settings_is_left_alone(self, manager, spec_for):
        # Standing in front of the library is not free: it lasts for the
        # rest of the model's life. Nothing to set means nothing to stand
        # in front of, and the handler talks to the module directly.
        spec = spec_for(PLAIN_PROJECTOR)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)

        assert kwargs["chat_handler"]._mtmd_cpp is llama_cpp.mtmd_cpp

    def test_the_defaults_come_back_carrying_the_values(self, manager, spec_for):
        spec = spec_for(WITH_CONTEXT)

        kwargs = manager._build_llama_load_kwargs("Local/model.gguf", 2048, spec)
        params = kwargs["chat_handler"]._mtmd_cpp.mtmd_context_params_default()

        assert params.warmup == 0
        assert params.image_min_tokens == 1024
        assert params.image_max_tokens == 2048
        assert params.batch_max_tokens == 4096

    def test_everything_else_still_comes_from_the_library(
        self, manager, spec_for
    ):
        spec = spec_for(WITH_CONTEXT)

        wrapper = manager._build_llama_load_kwargs(
            "Local/model.gguf", 2048, spec
        )["chat_handler"]._mtmd_cpp

        assert wrapper.mtmd_init_from_file is llama_cpp.mtmd_cpp.mtmd_init_from_file


class TestASettingThatWouldGoNowhere:
    def test_a_field_the_structure_does_not_have_is_refused(self, manager, spec_for):
        # A ctypes Structure takes an assignment to an unknown name
        # without complaint and without effect, so a field renamed
        # upstream would turn the whole section into a no-op.
        spec = spec_for(WITH_CONTEXT)
        wrapper = manager._build_llama_load_kwargs(
            "Local/model.gguf", 2048, spec
        )["chat_handler"]._mtmd_cpp
        wrapper._overrides["image_max_tokens_renamed"] = 1

        with pytest.raises(RuntimeError, match="image_max_tokens_renamed"):
            wrapper.mtmd_context_params_default()

    def test_a_handler_that_never_asked_for_defaults_is_refused(
        self, manager, spec_for
    ):
        spec = spec_for(WITH_CONTEXT)
        handler = manager._build_llama_load_kwargs(
            "Local/model.gguf", 2048, spec
        )["chat_handler"]

        class ModelWithAHandler:
            chat_handler = handler

        # Stands for a future _init_mtmd_context that builds its params
        # some other way: it initialises without error and the file's
        # settings are simply not in the context.
        handler._init_mtmd_context = lambda llama_model: None

        with pytest.raises(RuntimeError, match="without reading"):
            _initialise_projector(ModelWithAHandler())

    def test_an_initialisation_that_did_ask_is_accepted(self, manager, spec_for):
        spec = spec_for(WITH_CONTEXT)
        handler = manager._build_llama_load_kwargs(
            "Local/model.gguf", 2048, spec
        )["chat_handler"]

        class ModelWithAHandler:
            chat_handler = handler

        handler._init_mtmd_context = (
            lambda llama_model: handler._mtmd_cpp.mtmd_context_params_default()
        )

        _initialise_projector(ModelWithAHandler())
