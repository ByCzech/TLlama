"""Ollama-compatible reasoning names at the public/native protocol boundary."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tllama.schemas.openai import ChatCompletionRequest, ChatMessage


THOUGHT = 'Úvaha: <think>literal</think>\n'
PATH = '/v1/chat/completions'


@pytest.mark.parametrize('fields,expected', [
    ({}, None), ({'reasoning': None}, None), ({'reasoning_content': None}, None),
    ({'reasoning': None, 'reasoning_content': None}, None),
    ({'reasoning': THOUGHT}, THOUGHT), ({'reasoning_content': THOUGHT}, THOUGHT),
    ({'reasoning': THOUGHT, 'reasoning_content': THOUGHT}, THOUGHT),
    ({'reasoning': None, 'reasoning_content': THOUGHT}, THOUGHT),
    ({'reasoning': THOUGHT, 'reasoning_content': None}, THOUGHT),
    ({'reasoning': ''}, ''), ({'reasoning_content': ''}, ''),
    ({'reasoning': '', 'reasoning_content': ''}, ''),
])
def test_alias_becomes_one_canonical_field_without_mutating_input(fields, expected):
    source = {'role': 'assistant', 'content': None, **fields}
    before = deepcopy(source)
    message = ChatMessage.model_validate(source)
    assert source == before
    assert message.reasoning == expected
    assert message.model_dump(exclude_none=True) == {
        'role': 'assistant', **({'reasoning': expected} if expected is not None else {})}


@pytest.mark.parametrize('fields', [
    {'reasoning': 'a', 'reasoning_content': 'b'},
    {'reasoning': '', 'reasoning_content': 'b'},
    {'reasoning': 'a', 'reasoning_content': ''},
])
def test_conflicting_non_null_aliases_are_rejected(fields):
    with pytest.raises(ValidationError, match='Conflicting message reasoning and reasoning_content'):
        ChatMessage(role='assistant', **fields)


@pytest.mark.parametrize('field', ['reasoning', 'reasoning_content'])
@pytest.mark.parametrize('value', [False, 0, 1.5, [], {}, {'effort': 'high'}])
def test_message_reasoning_is_text_not_a_thinking_control(field, value):
    with pytest.raises(ValidationError):
        ChatMessage(role='assistant', **{field: value})


def output_message(response, stream):
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()['choices'][0]['message']
    frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    assert frames[-1] == '[DONE]'
    bodies = [json.loads(frame) for frame in frames[:-1]]
    deltas = [choice['delta'] for body in bodies for choice in body['choices']]
    assert all('reasoning_content' not in delta for delta in deltas)
    return {'role': 'assistant', 'content': ''.join(delta.get('content', '') for delta in deltas),
            **({'reasoning': ''.join(delta.get('reasoning', '') for delta in deltas)}
               if any('reasoning' in delta for delta in deltas) else {})}


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('field', ['reasoning', 'reasoning_content', 'both'])
def test_public_alias_replay_and_canonical_response_preserve_native_payload(client, native_reply, stream, field):
    state = native_reply(content='answer', reasoning=THOUGHT,
                         metadata={'sampling_defaults': {'temperature': 0.23}})
    fields = {key: THOUGHT for key in (['reasoning', 'reasoning_content'] if field == 'both' else [field])}
    payload = {'model': 'm', 'stream': stream, 'reasoning': {'effort': 'low'},
               'messages': [{'role': 'assistant', 'content': 'previous', **fields},
                            {'role': 'user', 'content': 'next'}]}
    message = output_message(client.post(PATH, json=payload), stream)
    assert message == {'role': 'assistant', 'content': 'answer', 'reasoning': THOUGHT}
    _, body = state.worker.calls[-1]
    assert body['messages'][0] == {'role': 'assistant', 'content': 'previous', 'reasoning_content': THOUGHT}
    assert body['temperature'] == 0.23 and body['reasoning_effort'] == 'low'
    assert state.refs == 0
    # A response can be appended verbatim and replayed on the next turn.
    payload['messages'] += [message, {'role': 'user', 'content': 'again'}]
    assert client.post(PATH, json=payload).status_code == 200
    assert state.worker.calls[-1][1]['messages'][-2] == {
        'role': 'assistant', 'content': 'answer', 'reasoning_content': THOUGHT}


@pytest.mark.parametrize('field', ['reasoning', 'reasoning_content'])
@pytest.mark.parametrize('value', [None, ''])
def test_null_is_absent_but_explicit_empty_history_is_preserved(client, native_reply, field, value):
    state = native_reply()
    response = client.post(PATH, json={'model': 'm', 'messages': [
        {'role': 'assistant', 'content': '', field: value}, {'role': 'user', 'content': 'next'}]})
    assert response.status_code == 200
    assert state.worker.calls[-1][1]['messages'][0] == {
        'role': 'assistant', 'content': '', **({'reasoning_content': ''} if value == '' else {})}


@pytest.mark.parametrize('stream', [False, True])
def test_conflict_is_rejected_before_admission_even_with_zero_keep_alive(client, native_reply, stream):
    state = native_reply()
    response = client.post(PATH, json={'model': 'm', 'stream': stream, 'keep_alive': 0,
        'messages': [{'role': 'assistant', 'reasoning': '', 'reasoning_content': 'different'}]})
    assert response.status_code == 400
    assert 'Conflicting message reasoning and reasoning_content' in response.json()['error']['message']
    assert not state.worker.calls and not state.actions and state.refs == 0


@pytest.mark.parametrize('field', ['reasoning', 'reasoning_content'])
def test_tools_only_assistant_history_preserves_reasoning_and_call_identity(client, native_reply, field):
    state = native_reply()
    call = {'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
    response = client.post(PATH, json={'model': 'm', 'messages': [
        {'role': 'assistant', 'content': None, field: THOUGHT, 'tool_calls': [call]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'}]})
    assert response.status_code == 200
    assert state.worker.calls[-1][1]['messages'][0] == {
        'role': 'assistant', 'content': '', 'reasoning_content': THOUGHT, 'tool_calls': [call]}


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('answer,reasoning', [('answer', ''), ('', THOUGHT)])
def test_public_output_keeps_empty_and_reasoning_only_results_separate(client, native_reply, stream, answer, reasoning):
    native_reply(content=answer, reasoning=reasoning)
    response = client.post(PATH, json={'model': 'm', 'stream': stream,
        'messages': [{'role': 'user', 'content': 'next'}]})
    assert output_message(response, stream) == {
        'role': 'assistant', 'content': answer, **({'reasoning': reasoning} if reasoning else {})}


@pytest.mark.parametrize('stream', [False, True])
def test_public_diagnostic_consumer_collects_canonical_reasoning_for_replay(stream):
    script = Path(__file__).resolve().parents[1] / 'scripts/tools-api-smoke.py'
    spec = importlib.util.spec_from_file_location('reasoning_public_consumer', script)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    consumer = smoke.Consumer('openai')
    if stream:
        for text in ['Úvaha', ': next']:
            consumer.push({'choices': [{'delta': {'reasoning': text}, 'finish_reason': None}]}, True)
        consumer.push({'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}]}, True)
    else:
        consumer.push({'choices': [{'message': {'reasoning': 'Úvaha: next', 'content': 'answer'},
                                    'finish_reason': 'stop'}]}, False)
    assert consumer.result()['message'] == {'role': 'assistant', 'content': 'answer', 'reasoning': 'Úvaha: next'}


def test_request_thinking_object_and_message_reasoning_are_independent():
    request = ChatCompletionRequest(model='m', reasoning={'effort': 'low'}, reasoning_effort='none',
        messages=[{'role': 'assistant', 'reasoning': THOUGHT}])
    assert request.reasoning == {'effort': 'low'} and request.reasoning_effort == 'none'
    assert request.messages[0].reasoning == THOUGHT
