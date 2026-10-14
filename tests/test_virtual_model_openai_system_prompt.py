"""OpenAI native requests preserve virtual model system defaults."""

from dataclasses import replace
from pathlib import Path

from tllama.config import BackendConfig
from tllama.native_config import resolve_worker_config
from tllama.native_requests import prepare_openai_chat
from tllama.schemas.openai import ChatCompletionRequest


def build_messages(request, metadata_info=None):
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={"device": "none"}), Path("/fake.gguf"))
    resolved = replace(resolved, system_prompt=(metadata_info or {}).get("default_system_prompt"))
    return prepare_openai_chat(request, resolved=resolved).body()["messages"]


def make_request(messages):
    return ChatCompletionRequest(model="m", messages=messages)


class TestBuildOpenaiChatMessages:
    def test_existing_system_message_wins_over_default(self):
        request = make_request([
            {"role": "system", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": "from messages"}
        assert not any(m["content"] == "from toml" for m in messages)

    def test_default_prepended_when_no_system_message(self):
        request = make_request([{"role": "user", "content": "hi"}])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": "from toml"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_developer_role_also_counts_as_system(self):
        request = make_request([
            {"role": "developer", "content": "from messages"},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_messages(request, metadata_info)

        assert not any(m["content"] == "from toml" for m in messages)

    def test_no_metadata_info_leaves_messages_untouched(self):
        request = make_request([{"role": "user", "content": "hi"}])

        messages = build_messages(request)

        assert messages == [{"role": "user", "content": "hi"}]

    def test_no_default_in_metadata_info_leaves_messages_untouched(self):
        request = make_request([{"role": "user", "content": "hi"}])

        messages = build_messages(request, {})

        assert messages == [{"role": "user", "content": "hi"}]

    def test_explicit_empty_toml_default_is_still_prepended(self):
        # An explicit "" from [system].prompt is a deliberate override, not
        # "nothing configured" -- it still needs to show up as a real
        # (empty-content) system message so the downstream chat handler
        # sees a system entry is present, same reasoning as the other call
        # sites.
        request = make_request([{"role": "user", "content": "hi"}])
        metadata_info = {"default_system_prompt": ""}

        messages = build_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": ""}

    def test_explicit_empty_system_message_is_not_replaced(self):
        request = make_request([
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ])
        metadata_info = {"default_system_prompt": "from toml"}

        messages = build_messages(request, metadata_info)

        assert messages[0] == {"role": "system", "content": ""}
        assert not any(m["content"] == "from toml" for m in messages)
