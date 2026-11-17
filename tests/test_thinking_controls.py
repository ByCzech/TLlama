"""Request precedence and conservative, qualified-template thinking controls."""

from dataclasses import replace
import logging
from pathlib import Path

import pytest

from tllama.helpers.chat import build_think_kwargs_ex
from tllama.helpers.prompt_render import render_generate_text
from tllama.helpers.thinking import resolve_think, thinking_profile
from tllama.native_requests import prepare_openai_chat
from tllama.schemas.ollama import OllamaGenerateRequest
from tllama.schemas.openai import ChatCompletionRequest

from test_native_requests import resolved


FIXTURES = Path(__file__).with_name('fixtures') / 'thinking'
GPT = {'arch': 'gpt-oss', 'template': (FIXTURES / 'openai-gpt-oss-120b.jinja').read_text()}
QWEN = {'arch': 'qwen3', 'template': (FIXTURES / 'Qwen-Qwen3-0.6B.jinja').read_text()}
QWEN35 = {'arch': 'qwen35', 'template': (FIXTURES / 'Qwen3.5-4B.jinja').read_text()}


def warnings(caplog):
    return [r for r in caplog.records if r.name == 'tllama.helpers.thinking' and r.levelno >= logging.WARNING]


@pytest.mark.parametrize('fields,expected,warning', [
    ({}, None, False), ({'reasoning_effort': ' high '}, 'high', False),
    ({'reasoning': {'effort': 'high'}}, 'high', False),
    ({'reasoning_effort': 'low', 'reasoning': {'effort': 'high'}}, 'high', True),
    ({'reasoning_effort': 'none', 'reasoning': {'effort': 'high'}}, 'high', True),
    ({'reasoning_effort': 'high', 'reasoning': {'effort': 'none'}}, 'none', True),
    ({'reasoning_effort': 'high', 'reasoning': {'effort': 'HIGH'}}, 'high', False),
    ({'reasoning_effort': 'high', 'reasoning': None}, 'high', True),
    ({'reasoning_effort': 'high', 'reasoning': {}}, 'high', True),
    ({'reasoning_effort': 'high', 'reasoning': {'effort': None}}, 'high', True),
    ({'reasoning_effort': 'high', 'reasoning': {'effort': ''}}, 'high', True),
    ({'reasoning_effort': 'high', 'reasoning': {'effort': ' \t '}}, 'high', True),
    ({'reasoning_effort': '', 'reasoning': {'effort': 'high'}}, 'high', True),
    ({'reasoning_effort': None, 'reasoning': {'effort': 'high'}}, 'high', True),
    ({'reasoning_effort': '', 'reasoning': {'effort': None}}, None, False),
    ({'reasoning_effort': None, 'reasoning': {}}, None, False),
    ({'reasoning_effort': 'custom-level', 'reasoning': {'effort': 'other-custom'}}, 'other-custom', True),
])
def test_public_precedence_null_empty_and_warning_once(client, native_reply, caplog, fields, expected, warning):
    state = native_reply()
    response = client.post('/v1/chat/completions', json={
        'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], **fields})
    assert response.status_code == 200, response.text
    body = state.worker.calls[-1][1]
    assert body.get('reasoning_effort') == expected
    if expected is None:
        assert 'chat_template_kwargs' not in body
    assert len(warnings(caplog)) == int(warning)
    if warning:
        assert 'using reasoning' in warnings(caplog)[0].getMessage()
    assert state.refs == 0


@pytest.mark.parametrize('bad', [False, 0, [], {}, 1.5])
@pytest.mark.parametrize('field', ['flat', 'nested'])
def test_invalid_effort_type_even_on_losing_side_is_rejected_before_admission(client, native_reply, bad, field):
    state = native_reply()
    body = {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], 'keep_alive': 0,
            'reasoning_effort': 'low', 'reasoning': {'effort': 'high'}}
    if field == 'flat':
        body['reasoning_effort'] = bad
    else:
        body['reasoning']['effort'] = bad
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 400
    assert not state.worker.calls and not state.actions and state.refs == 0


@pytest.mark.parametrize('value,expected,warning', [
    (None, None, False), (' ', None, False), (True, 'medium', False),
    ('low', 'low', False), ('medium', 'medium', False), ('high', 'high', False),
    ('minimal', 'low', False), ('xhigh', 'high', False), ('ultra', 'high', False),
    (False, 'low', True), ('none', 'low', True), ('false', 'low', True),
    ('max', 'medium', True), ('future-level', 'medium', True),
])
def test_qualified_gpt_oss_uses_supported_level_alias_lowest_or_default(caplog, value, expected, warning):
    assert resolve_think(value, GPT) == expected
    assert len(warnings(caplog)) == int(warning)


@pytest.mark.parametrize('metadata', [QWEN, QWEN35])
@pytest.mark.parametrize('value,expected,warning', [
    (None, None, False), (True, True, False), (False, False, False),
    ('none', False, False), ('minimal', True, False), ('low', True, False),
    ('medium', True, False), ('high', True, False), ('xhigh', True, False),
    ('ultra', True, False), ('max', True, False), ('future-level', True, True),
])
def test_qualified_boolean_templates_do_not_invent_effort_levels(caplog, metadata, value, expected, warning):
    kwargs = build_think_kwargs_ex(value, metadata)
    assert kwargs.get('enable_thinking') is expected
    assert 'ThinkLevel' not in kwargs if expected is not False else kwargs['ThinkLevel'] == 'none'
    assert len(warnings(caplog)) == int(warning)


@pytest.mark.parametrize('value', ['minimal', 'xhigh', 'ultra', 'max', 'future-level', False])
@pytest.mark.parametrize('metadata', [{}, {'arch': 'gpt-oss', 'template': '{{ content }}'},
    {**GPT, 'template': GPT['template'] + '\n'}, {**GPT, 'arch': 'other'}])
def test_unqualified_contracts_pass_values_through_without_inventing_restrictions(caplog, metadata, value):
    assert thinking_profile(metadata) is None
    assert resolve_think(value, metadata) == value
    assert not warnings(caplog)


@pytest.mark.parametrize('flat,nested', [('low', 'minimal'), ('high', 'xhigh'), ('high', 'ultra')])
def test_equal_qualified_aliases_do_not_warn(resolved, caplog, flat, nested):
    request = ChatCompletionRequest(model='arbitrary-alias', messages=[],
        reasoning_effort=flat, reasoning={'effort': nested})
    body = prepare_openai_chat(request, resolved=resolved, metadata_info=GPT).body()
    assert body['reasoning_effort'] == flat
    assert not warnings(caplog)


def test_fallback_does_not_switch_back_to_the_other_request_field(resolved, caplog):
    request = ChatCompletionRequest(model='m', messages=[], reasoning_effort='low', reasoning={'effort': 'max'})
    body = prepare_openai_chat(request, resolved=resolved, metadata_info=GPT).body()
    assert body['reasoning_effort'] == 'medium'
    assert len(warnings(caplog)) == 2  # Conflict, then selected value's default substitution.


def test_tool_and_explicit_templates_select_their_own_contract(resolved, caplog):
    request = ChatCompletionRequest(model='m', messages=[], reasoning_effort='xhigh',
                                    tools=[{'type': 'function', 'function': {'name': 'f'}}])
    metadata = {**GPT, 'tool_template': '{{ messages }}'}
    assert prepare_openai_chat(request, resolved=resolved, metadata_info=metadata).body()['reasoning_effort'] == 'xhigh'
    explicit = replace(resolved, launch=replace(resolved.launch, template=QWEN['template']))
    body = prepare_openai_chat(request, resolved=explicit, metadata_info=metadata).body()
    assert body['chat_template_kwargs']['enable_thinking'] is True and 'reasoning_effort' not in body
    assert not warnings(caplog)


@pytest.mark.parametrize('value,expected', [('minimal', 'low'), ('xhigh', 'high'), ('max', 'medium'), (False, 'low')])
def test_real_gpt_template_rendering_matches_native_chat_control(resolved, value, expected):
    request = OllamaGenerateRequest(model='m', prompt='hi', think=value)
    prompt, _ = render_generate_text(GPT, request, bos_token='', eos_token='')
    assert 'Reasoning: ' + expected + '\n' in prompt
    kwargs = build_think_kwargs_ex(value, GPT)
    assert kwargs['reasoning_effort'] == expected and kwargs['enable_thinking'] is True


@pytest.mark.parametrize('metadata', [QWEN, QWEN35])
def test_real_boolean_templates_retain_disabled_prefill_and_no_level_metadata(metadata):
    off, _ = render_generate_text(metadata, OllamaGenerateRequest(model='m', prompt='hi', think=False),
                                  bos_token='', eos_token='')
    on, _ = render_generate_text(metadata, OllamaGenerateRequest(model='m', prompt='hi', think='max'),
                                 bos_token='', eos_token='')
    assert off.endswith('<think>\n\n</think>\n\n')
    assert off != on and '<think>\n\n</think>\n\n' not in on


def test_generate_request_override_invalidates_qualified_profile():
    request = OllamaGenerateRequest(model='gpt-oss', prompt='hi', think='max', template='{{ reasoning_effort }}')
    assert render_generate_text(GPT, request, bos_token='', eos_token='')[0] == 'max'


def test_generate_optional_prefill_comparison_does_not_duplicate_warning(client, native_reply, caplog):
    state = native_reply(metadata=QWEN35)
    response = client.post('/api/generate', json={'model': 'm', 'prompt': 'hi', 'think': 'future-level', 'stream': False})
    assert response.status_code == 200, response.text
    assert state.worker.prompts[-1].endswith('<think>\n')
    assert len(warnings(caplog)) == 1


def test_omitted_value_preserves_actual_template_defaults():
    request = OllamaGenerateRequest(model='m', prompt='hi')
    prompt, _ = render_generate_text(GPT, request, bos_token='', eos_token='')
    assert 'Reasoning: medium\n' in prompt
    assert build_think_kwargs_ex(None, GPT) == {}
