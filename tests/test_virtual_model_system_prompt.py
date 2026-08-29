"""default_system_prompt (a virtual model's [system].prompt) is used only
when the caller supplied nothing of its own -- for both render functions,
the same rule regardless of endpoint (confirmed against real Ollama's
current GenerateHandler/ChatHandler source during the design discussion).

/api/chat's own default-system-prompt behavior is not covered by this file:
it never went through render_chat_prompt_with_explicit_think() (removed as
dead code -- superseded by create_chat_completion_ex() back in April, its
call site gone, the function itself simply never deleted afterward) and its
real fix belongs to a separate, dedicated patch.
"""

from types import SimpleNamespace

from tllama.helpers.prompt_render import render_generate_prompt


SIMPLE_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
)

# Mimics a real chat template's own fallback: inject a baked-in default only
# when no system-role entry is present at all, regardless of its content.
TEMPLATE_WITH_BAKED_DEFAULT = (
    "{% if messages[0].role != 'system' %}system:BAKED DEFAULT\n{% endif %}"
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
)


class FakeLlm:
    def token_bos(self):
        return -1

    def token_eos(self):
        return -1


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

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "system:from request" in prompt
        assert "from toml" not in prompt

    def test_default_used_when_request_system_absent(self):
        request = make_request(system=None)
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "system:from toml" in prompt

    def test_no_system_message_when_neither_is_set(self):
        request = make_request(system=None)
        metadata_info = {"template": SIMPLE_TEMPLATE}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "system:" not in prompt

    def test_explicit_empty_request_system_is_not_the_same_as_absent(self):
        # An explicit "" is a deliberate request for no system content, not
        # "nothing provided" -- it must not fall back to the toml default.
        request = make_request(system="")
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "system:" in prompt
        assert "from toml" not in prompt

    def test_explicit_empty_request_system_suppresses_the_template_s_own_default(self):
        # A real chat template's own fallback usually checks for the
        # presence of a system-role entry, not whether its content is
        # non-empty -- an explicit "" has to count as present.
        request = make_request(system="")
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "BAKED DEFAULT" not in prompt

    def test_explicit_empty_toml_default_is_used_and_suppresses_the_template_s_own_default(self):
        request = make_request(system=None)
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT, "default_system_prompt": ""}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "BAKED DEFAULT" not in prompt

    def test_truly_nothing_provided_lets_the_template_s_own_default_through(self):
        request = make_request(system=None)
        metadata_info = {"template": TEMPLATE_WITH_BAKED_DEFAULT}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "BAKED DEFAULT" in prompt


class TestGeneratePromptMessagesMode:
    """mode="messages" is currently unreachable from any router, but the
    contract should still hold if/when it is."""

    def test_existing_system_message_wins_over_default(self):
        request = make_request(messages=[
            {"role": "system", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="messages")

        assert "system:from messages" in prompt
        assert "from toml" not in prompt

    def test_default_prepended_when_no_system_message(self):
        request = make_request(messages=[{"role": "user", "content": "hi"}])
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="messages")

        assert "system:from toml" in prompt
