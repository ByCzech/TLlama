"""Same default-system-prompt rule for /v1/chat/completions.

This is a separate call site from prompt_render.py's two functions --
build_openai_chat_messages() feeds llama-cpp-python's own chat completion
directly, bypassing our Jinja rendering entirely -- so the rule has to be
implemented here too, not inherited for free.
"""

from types import SimpleNamespace

from tllama.helpers.openai_compat import build_openai_chat_messages


def make_request(messages):
    return SimpleNamespace(messages=[SimpleNamespace(**m) for m in messages])


class TestBuildOpenaiChatMessages:
    def test_existing_system_message_wins_over_default(self):
        request = make_request([
            {"role": "system", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_openai_chat_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": "from messages"}
        assert not any(m["content"] == "from toml" for m in messages)

    def test_default_prepended_when_no_system_message(self):
        request = make_request([{"role": "user", "content": "hi"}])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_openai_chat_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": "from toml"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_developer_role_also_counts_as_system(self):
        request = make_request([
            {"role": "developer", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_openai_chat_messages(request, metadata_info)

        assert not any(m["content"] == "from toml" for m in messages)

    def test_no_metadata_info_leaves_messages_untouched(self):
        request = make_request([{"role": "user", "content": "hi"}])

        messages = build_openai_chat_messages(request)

        assert messages == [{"role": "user", "content": "hi"}]

    def test_no_default_in_metadata_info_leaves_messages_untouched(self):
        request = make_request([{"role": "user", "content": "hi"}])

        messages = build_openai_chat_messages(request, {})

        assert messages == [{"role": "user", "content": "hi"}]

    def test_explicit_empty_toml_default_is_still_prepended(self):
        # An explicit "" from [system].prompt is a deliberate override, not
        # "nothing configured" -- it still needs to show up as a real
        # (empty-content) system message so the downstream chat handler
        # sees a system entry is present, same reasoning as the other call
        # sites.
        request = make_request([{"role": "user", "content": "hi"}])
        metadata_info = {"default_system_prompt": ""}

        messages = build_openai_chat_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": ""}

    def test_explicit_empty_system_message_is_not_replaced(self):
        request = make_request([
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_openai_chat_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": ""}
        assert not any(m["content"] == "from toml" for m in messages)
