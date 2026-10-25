"""Native generate rendering preserves system and stop defaults."""

from types import SimpleNamespace

from tllama.helpers.prompt_render import render_generate_text


SIMPLE_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
)

# Mimics a real chat template's own fallback: inject a baked-in default only
# when no system-role entry is present at all, regardless of its content.
TEMPLATE_WITH_BAKED_DEFAULT = (
    "{% if messages[0].role != 'system' %}system:BAKED DEFAULT\n{% endif %}"
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
)




def make_request(**kwargs):
    defaults = {
        "prompt": "hello",
        "system": None,
        "template": None,
        "think": None,
        "options": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestGeneratePromptMode:
    def test_request_system_wins_over_default(self):
        request = make_request(system="from request")
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "system:from request" in prompt
        assert "from toml" not in prompt

    def test_default_used_when_request_system_absent(self):
        request = make_request(system=None)
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "system:from toml" in prompt

    def test_no_system_message_when_neither_is_set(self):
        request = make_request(system=None)
        metadata_info = {"template": SIMPLE_TEMPLATE}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "system:" not in prompt

    def test_explicit_empty_request_system_is_not_the_same_as_absent(self):
        # An explicit "" is a deliberate request for no system content, not
        # "nothing provided" -- it must not fall back to the toml default.
        request = make_request(system="")
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "system:" in prompt
        assert "from toml" not in prompt

    def test_explicit_empty_request_system_suppresses_the_template_s_own_default(self):
        # A real chat template's own fallback usually checks for the
        # presence of a system-role entry, not whether its content is
        # non-empty -- an explicit "" has to count as present.
        request = make_request(system="")
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "BAKED DEFAULT" not in prompt

    def test_explicit_empty_toml_default_is_used_and_suppresses_the_template_s_own_default(self):
        request = make_request(system=None)
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT, "default_system_prompt": ""}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "BAKED DEFAULT" not in prompt

    def test_truly_nothing_provided_lets_the_template_s_own_default_through(self):
        request = make_request(system=None)
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT}

        prompt, _ = render_generate_text(metadata_info, request, bos_token="", eos_token="")

        assert "BAKED DEFAULT" in prompt
