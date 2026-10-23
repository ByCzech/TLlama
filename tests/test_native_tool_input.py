"""Tools admission rejects broken links without loading or dispatching models."""

from copy import deepcopy
import json

import pytest

from tllama.native_requests import normalize_chat_input
from tllama.native_tool_input import tool_options
from test_native_public_api import public, native_manager


TOOL = {'type': 'function', 'function': {'name': 'lookup_record', 'parameters': {'type': 'object'}}}


def history(*, ids=True, objects=False):
    calls = [{'type': 'function', 'function': {'name': 'lookup_record',
              'arguments': {'key': key} if objects else json.dumps({'key': key})},
              **({'id': f'call-{i}'} if ids else {})} for i, key in enumerate(['alpha', 'beta'])]
    return [{'role': 'assistant', 'tool_calls': calls},
            *[{'role': 'tool', 'content': f'marker-{i}', **({'tool_call_id': f'call-{i}'} if ids else
               {'tool_name': 'lookup_record'})} for i in range(2)]]


@pytest.mark.parametrize('tools', [
    [{}], [{'type': 'custom', 'function': {'name': 'f'}}], [{'function': []}],
    [{'function': {'name': ''}}], [{'function': {'name': '  '}}],
    [TOOL, TOOL], [{'function': {'name': 'f', 'parameters': []}}],
    [{'function': {'name': 'f', 'description': 1}}], [{'function': {'name': 'f', 'strict': 'true'}}],
])
@pytest.mark.parametrize('path', ['/api/chat', '/v1/chat/completions'])
async def test_invalid_declarations_fail_before_worker_admission(public, tools, path):
    manager, client, name, _ = public
    response = await client.post(path, json={'model': name, 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': tools})
    assert response.status_code == 400 and 'error' in response.json()
    assert not manager.test.ledger and not manager.native_snapshot()


@pytest.mark.parametrize('choice', ['required', {'type': 'function', 'function': {'name': 'missing'}},
    {'type': 'function', 'function': {'name': ''}}, 'invalid'])
async def test_invalid_choice_is_checked_before_even_preload_admission(public, choice):
    manager, client, name, _ = public
    response = await client.post('/v1/chat/completions', json={
        'model': name, 'messages': [], 'tools': [], 'tool_choice': choice})
    assert response.status_code == 400 and not manager.test.ledger


@pytest.mark.parametrize('value', ['false', 1])
async def test_parallel_flag_is_not_coerced_from_non_boolean_json(public, value):
    manager, client, name, _ = public
    response = await client.post('/v1/chat/completions', json={
        'model': name, 'messages': [], 'tools': [TOOL], 'parallel_tool_calls': value})
    assert response.status_code == 400 and not manager.test.ledger


def bad_history(case):
    messages = history()
    if case == 'duplicate-id':
        messages[0]['tool_calls'][1]['id'] = 'call-0'
    elif case == 'wrong-id':
        messages[1]['tool_call_id'] = 'unknown'
    elif case == 'wrong-name':
        messages[1]['name'] = 'other'
    elif case == 'duplicate-result':
        messages[2] = deepcopy(messages[1])
    elif case == 'missing-result':
        messages.pop()
    elif case == 'interrupted':
        messages.insert(1, {'role': 'user', 'content': 'new question'})
    elif case == 'orphan':
        messages.pop(0)
    elif case == 'bad-arguments':
        messages[0]['tool_calls'][0]['function']['arguments'] = '{"key":'
    elif case == 'duplicate-keys':
        messages[0]['tool_calls'][0]['function']['arguments'] = '{"key":1,"key":2}'
    elif case == 'non-assistant':
        messages[0]['role'] = 'user'
    elif case == 'unknown-type':
        messages[0]['tool_calls'][0]['type'] = 'custom'
    elif case == 'conflicting-alias':
        messages[1].update(name='lookup_record', tool_name='other')
    return messages


@pytest.mark.parametrize('case,path', [(case, path) for case in [
    'duplicate-id', 'wrong-id', 'wrong-name', 'duplicate-result', 'missing-result', 'interrupted', 'orphan',
    'bad-arguments', 'duplicate-keys', 'non-assistant', 'unknown-type', 'conflicting-alias']
    for path in ['/api/chat', '/v1/chat/completions'] if case != 'conflicting-alias' or path == '/api/chat'])
async def test_broken_history_is_a_public_400_without_loading(public, case, path):
    manager, client, name, _ = public
    response = await client.post(path, json={'model': name, 'messages': bad_history(case), 'stream': False})
    assert response.status_code == 400, response.text
    assert not manager.test.ledger and not manager.native_snapshot()


def test_ollama_name_only_parallel_history_and_id_enrichment_preserve_call_order():
    messages = history(objects=True)
    for message in messages[1:]:
        del message['tool_call_id']
        message['tool_name'] = 'lookup_record'
    original = deepcopy(messages)
    result = normalize_chat_input(messages)
    assert [m['tool_call_id'] for m in result[1:]] == ['call-0', 'call-1']
    assert all(m['name'] == 'lookup_record' for m in result[1:])
    assert messages == original
    no_ids = normalize_chat_input(history(ids=False, objects=True))
    assert all('tool_call_id' not in m for m in no_ids)
    assert all('id' not in c for c in no_ids[0]['tool_calls'])
    messages = history(ids=False, objects=True)
    for call in messages[0]['tool_calls']:
        call['id'] = None
    for message in messages[1:]:
        message['tool_call_id'] = None
    normalized = normalize_chat_input(messages)
    assert all('id' not in c for c in normalized[0]['tool_calls'])
    assert all('tool_call_id' not in m for m in normalized[1:])


def test_ollama_unnamed_results_and_explicit_reverse_ids_are_unambiguous():
    messages = history()
    for message in messages[1:]:
        del message['tool_call_id']
    assert [m['tool_call_id'] for m in normalize_chat_input(messages)[1:]] == ['call-0', 'call-1']
    messages = history()
    messages[1:] = reversed(messages[1:])
    assert [m['tool_call_id'] for m in normalize_chat_input(messages, tool_api='openai')[1:]] == ['call-1', 'call-0']


def test_historical_tools_need_no_current_declaration_and_ids_can_repeat_in_later_turns():
    messages = history() + [{'role': 'user', 'content': 'again'}] + history()
    assert len(normalize_chat_input(messages, tool_api='openai')) == len(messages)
    assert tool_options([], 'none') == ([], 'none')
    assert tool_options(None) == (None, None)


@pytest.mark.parametrize('missing', ['call-id', 'result-id', 'string-arguments'])
def test_openai_history_uses_its_explicit_id_and_string_contract(missing):
    messages = history()
    if missing == 'call-id':
        del messages[0]['tool_calls'][0]['id']
    elif missing == 'result-id':
        del messages[1]['tool_call_id']
    else:
        messages[0]['tool_calls'][0]['function']['arguments'] = {}
    with pytest.raises(ValueError):
        normalize_chat_input(messages, tool_api='openai')


def test_options_keep_schema_extensions_and_named_choice_without_input_mutation():
    declaration = deepcopy(TOOL)
    declaration['function']['parameters'] = {'$ref': '#/$defs/args', '$defs': {'args': {'type': 'object'}}, 'x-vendor': True}
    original = deepcopy(declaration)
    selected, choice = tool_options([declaration], {'type': 'function', 'function': {'name': 'lookup_record'}})
    assert selected == [declaration] and choice == 'required' and declaration == original
    for bad in ([], 1, True):
        with pytest.raises(ValueError, match='tool_choice'):
            tool_options([declaration], bad)
    with pytest.raises(ValueError, match='parallel_tool_calls'):
        tool_options([declaration], parallel=1)
