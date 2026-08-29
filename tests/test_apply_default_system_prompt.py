"""apply_default_system_prompt(): the real fix for /api/chat's [system]
default, wired directly into routers/ollama.py's actual chat handler.
"""

from tllama.helpers.chat import apply_default_system_prompt


class TestApplyDefaultSystemPrompt:
    def test_existing_system_message_wins_over_default(self):
        messages = [
            {"role": "system", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ]
        metadata_info = {"default_system_prompt": "from toml"}

        result = apply_default_system_prompt(messages, metadata_info)

        assert result[0] == {"role": "system", "content": "from messages"}
        assert not any(m["content"] == "from toml" for m in result)

    def test_default_prepended_when_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"default_system_prompt": "from toml"}

        result = apply_default_system_prompt(messages, metadata_info)

        assert result[0] == {"role": "system", "content": "from toml"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_developer_role_also_counts_as_system(self):
        messages = [
            {"role": "developer", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ]
        metadata_info = {"default_system_prompt": "from toml"}

        result = apply_default_system_prompt(messages, metadata_info)

        assert not any(m["content"] == "from toml" for m in result)

    def test_no_default_leaves_messages_untouched(self):
        messages = [{"role": "user", "content": "hi"}]

        result = apply_default_system_prompt(messages, {})

        assert result == [{"role": "user", "content": "hi"}]

    def test_original_messages_list_is_not_mutated(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"default_system_prompt": "from toml"}

        apply_default_system_prompt(messages, metadata_info)

        assert messages == [{"role": "user", "content": "hi"}]

    def test_explicit_empty_system_message_is_not_replaced(self):
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]
        metadata_info = {"default_system_prompt": "from toml"}

        result = apply_default_system_prompt(messages, metadata_info)

        assert result[0] == {"role": "system", "content": ""}
        assert not any(m["content"] == "from toml" for m in result)

    def test_explicit_empty_toml_default_is_still_prepended(self):
        messages = [{"role": "user", "content": "hi"}]
        metadata_info = {"default_system_prompt": ""}

        result = apply_default_system_prompt(messages, metadata_info)

        assert result[0] == {"role": "system", "content": ""}
