"""default_system_prompt (a virtual model's [system].prompt) is used only
when the caller supplied nothing of its own -- for both render functions,
the same rule regardless of endpoint (confirmed against real Ollama's
current GenerateHandler/ChatHandler source during the design discussion).
"""

from types import SimpleNamespace

from tllama.helpers.prompt_render import (
    render_generate_prompt,
    render_chat_prompt_with_explicit_think,
)


SIMPLE_TEMPLATE = (
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

    def test_empty_string_request_system_falls_back_to_default(self):
        # Ollama's own GenerateHandler treats an empty system string as "not
        # provided" (req.System == "" check) -- same rule here.
        request = make_request(system="")
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_generate_prompt(FakeLlm(), metadata_info, request, mode="prompt")

        assert "system:from toml" in prompt


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


class TestChatPromptWithExplicitThink:
    def test_existing_system_message_wins_over_default(self):
        messages = [
            {"role": "system", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ]
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_chat_prompt_with_explicit_think(
            FakeLlm(), metadata_info, messages, think_enabled=False, user_stop=[]
        )

        assert "system:from messages" in prompt
        assert "from toml" not in prompt

    def test_default_prepended_when_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_chat_prompt_with_explicit_think(
            FakeLlm(), metadata_info, messages, think_enabled=False, user_stop=[]
        )

        assert "system:from toml" in prompt

    def test_developer_role_also_counts_as_system(self):
        messages = [
            {"role": "developer", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ]
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        prompt, _ = render_chat_prompt_with_explicit_think(
            FakeLlm(), metadata_info, messages, think_enabled=False, user_stop=[]
        )

        assert "from toml" not in prompt

    def test_no_default_leaves_messages_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"template": SIMPLE_TEMPLATE}

        prompt, _ = render_chat_prompt_with_explicit_think(
            FakeLlm(), metadata_info, messages, think_enabled=False, user_stop=[]
        )

        assert "system:" not in prompt

    def test_original_messages_list_is_not_mutated(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"template": SIMPLE_TEMPLATE, "default_system_prompt": "from toml"}

        render_chat_prompt_with_explicit_think(
            FakeLlm(), metadata_info, messages, think_enabled=False, user_stop=[]
        )

        assert messages == [{"role": "user", "content": "hi"}]
