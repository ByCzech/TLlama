"""Truthful offline show provenance, profile limits and no worker operations."""

from dataclasses import replace
import json
from types import SimpleNamespace

import gguf
import pytest

from tllama.helpers.metadata_cache import load_metadata_cache
from tllama.helpers.show_diagnostics import chat_diagnostics
from tllama.native_requests import prepare_openai_chat
from tllama.native_routes import lease_metadata
from tllama.schemas.openai import ChatCompletionRequest

from test_native_requests import resolved
from test_thinking_controls import GPT, QWEN, QWEN35


@pytest.fixture
def show_repo(manager, monkeypatch):
    from tllama.routers import ollama
    monkeypatch.setattr(ollama, 'model_manager', manager)

    async def forbidden(*args, **kwargs):
        pytest.fail('show must not operate on model workers')
    for name in ('acquire_native_model', 'preload_native_model', 'unload_native_model'):
        monkeypatch.setattr(manager, name, forbidden)

    def define(name='m', *, metadata=GPT, override=None, tool=None, extra=''):
        weights = manager.local_models_dir / (name + '.gguf')
        writer = gguf.GGUFWriter(str(weights), metadata['arch'])
        if metadata.get('template'):
            writer.add_string('tokenizer.chat_template', metadata['template'])
        if tool is not None:
            writer.add_string('tokenizer.chat_template.tool_use', tool)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.close()
        toml = manager.local_models_dir / (name + '.toml')
        text = '[llm]\nmodel="Local/' + name + '.gguf"\n' + extra
        if override is not None:
            text += '\n[template]\njinja=' + json.dumps(override) + '\n'
        toml.write_text(text)
        return weights, toml
    return manager, define


def shown(client, name='m'):
    response = client.post('/api/show', json={'model': name})
    assert response.status_code == 200, response.text
    return response.json()


def test_gguf_source_and_qualified_thinking_are_reported_without_a_worker(client, show_repo):
    manager, define = show_repo
    define()
    data = shown(client)
    assert (data['renderer'], data['parser'], data['template_source']) == ('llama.cpp', 'native', 'gguf_chat_template')
    assert data['template'] == GPT['template']
    assert data['thinking'] == {'values': ['low', 'medium', 'high'], 'default': 'medium'}
    assert not manager.native_snapshot()


@pytest.mark.parametrize('metadata', [QWEN, QWEN35])
def test_boolean_thinking_preserves_json_boolean_types(client, show_repo, metadata):
    _, define = show_repo
    define(metadata=metadata)
    assert shown(client)['thinking'] == {'values': [False, True], 'default': True}


def test_identical_toml_text_is_still_a_toml_source(client, show_repo):
    _, define = show_repo
    define(override=GPT['template'])
    data = shown(client)
    assert data['template_source'] == 'tllama_toml' and 'thinking' in data


def test_empty_toml_override_uses_gguf_fallback_and_keeps_definition_verbatim(client, show_repo):
    _, define = show_repo
    _, toml = define(override='')
    data = shown(client)
    assert data['template_source'] == 'gguf_chat_template' and data['template'] == GPT['template']
    assert data['modelfile'] == toml.read_text() and 'jinja=""' in data['modelfile']
    assert data['thinking']['default'] == 'medium'


@pytest.mark.parametrize('override', [None, ''])
def test_missing_template_is_empty_unknown_and_not_synthetic(client, show_repo, override):
    _, define = show_repo
    define(metadata={'arch': 'llama'}, override=override)
    data = shown(client)
    assert data['template'] == '' and data['template_source'] == 'unknown'
    assert data['renderer'] == 'llama.cpp' and data['parser'] == 'native'
    assert 'thinking' not in data


def test_custom_override_is_not_given_inherited_thinking_levels(client, show_repo):
    _, define = show_repo
    define(override='{{ messages }}')
    data = shown(client)
    assert data['template_source'] == 'tllama_toml' and 'thinking' not in data


@pytest.mark.parametrize('tool', [QWEN['template'], '{{ messages }}'])
def test_different_or_unknown_tool_contract_suppresses_shared_thinking(client, show_repo, tool):
    _, define = show_repo
    define(tool=tool)
    data = shown(client)
    assert data['template_source'] == 'gguf_chat_template' and data['template'] == GPT['template']
    assert 'thinking' not in data


def test_toml_override_replaces_both_gguf_variants(client, show_repo):
    _, define = show_repo
    define(tool='{{ messages }}', override=GPT['template'])
    assert shown(client)['thinking']['default'] == 'medium'


@pytest.mark.parametrize('tool', [QWEN['template'], QWEN35['template']])
def test_equivalent_qualified_boolean_tool_contract_can_be_reported(client, show_repo, tool):
    _, define = show_repo
    define(metadata=QWEN, tool=tool)
    assert shown(client)['thinking'] == {'values': [False, True], 'default': True}


def test_edits_and_alias_provenance_do_not_mutate_shared_gguf_cache(client, show_repo):
    manager, define = show_repo
    weights, toml = define()
    before = shown(client)
    cached = load_metadata_cache(manager.metadata_cache_dir, weights)
    assert '_template_source' not in cached and cached['template'] == GPT['template']
    alias = manager.local_models_dir / 'alias.toml'
    alias.write_text(toml.read_text() + '\n[template]\njinja=' + json.dumps(GPT['template']) + '\n')
    assert shown(client, 'alias')['template_source'] == 'tllama_toml'
    assert shown(client)['template_source'] == before['template_source'] == 'gguf_chat_template'
    alias.write_text(toml.read_text() + '\n[template]\njinja="{{ messages }}"\n')
    assert 'thinking' not in shown(client, 'alias')
    alias.write_text(toml.read_text())
    assert shown(client, 'alias')['template_source'] == 'gguf_chat_template'
    assert load_metadata_cache(manager.metadata_cache_dir, weights) == cached


def test_embedding_definition_omits_chat_diagnostics_even_with_a_thinking_template(client, show_repo):
    _, define = show_repo
    define(extra='\n[runtime]\nembedding=true\npooling_type=1\n')
    assert not ({'renderer', 'parser', 'template_source', 'thinking'} & shown(client).keys())


def test_unknown_model_is_still_404(client, show_repo):
    assert client.post('/api/show', json={'model': 'missing'}).status_code == 404


def test_header_failure_keeps_explicit_embedding_mode_without_chat_fields(client, show_repo, monkeypatch):
    manager, define = show_repo
    define(extra='\n[runtime]\nembedding=true\npooling_type=1\n')
    async def unavailable(_):
        return None
    monkeypatch.setattr(manager, '_get_raw_model_metadata', unavailable)
    assert not ({'renderer', 'parser', 'template_source', 'thinking'} & shown(client).keys())


def test_header_failure_retains_nonempty_toml_source(client, show_repo, monkeypatch):
    manager, define = show_repo
    define(override='{{ messages }}')
    async def unavailable(_):
        return None
    monkeypatch.setattr(manager, '_get_raw_model_metadata', unavailable)
    data = shown(client)
    assert data['template'] == '{{ messages }}' and data['template_source'] == 'tllama_toml'
    assert 'thinking' not in data


def test_unintegrated_decision_head_omits_chat_diagnostics(client, show_repo, monkeypatch):
    manager, define = show_repo
    define()
    async def metadata(_):
        return {**GPT, 'tool_template': GPT['template'], 'metadata_raw': {
            'general.architecture': 'gpt-oss', 'gpt-oss.decision.type': 'future'}}
    monkeypatch.setattr(manager, '_get_raw_model_metadata', metadata)
    assert not ({'renderer', 'parser', 'template_source', 'thinking'} & shown(client).keys())


def test_incomplete_old_metadata_does_not_claim_a_shared_tool_contract():
    assert 'thinking' not in chat_diagnostics(GPT)
    assert chat_diagnostics({**GPT, 'tool_template': GPT['template']})['thinking']['default'] == 'medium'


def test_empty_runtime_override_uses_the_same_native_fallback_metadata(resolved):
    resolved = replace(resolved, launch=replace(resolved.launch, template=''))
    lease = SimpleNamespace(resolved=resolved, definition=SimpleNamespace(metadata=GPT),
        worker=SimpleNamespace(props={'chat_template': GPT['template']}))
    metadata = lease_metadata(lease)
    assert metadata['template'] == GPT['template']
    request = ChatCompletionRequest(model='m', messages=[], reasoning_effort='minimal')
    assert prepare_openai_chat(request, resolved=resolved, metadata_info=metadata).body()['reasoning_effort'] == 'low'
