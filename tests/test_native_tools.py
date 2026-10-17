"""Production projections checked against captured native and hostile streams."""

from contextlib import aclosing
import json
from pathlib import Path

import pytest

from tllama.native_api import (_read_events, read_result, ollama_response, openai_response,
                               ollama_stream, openai_stream, ReplyContext)
from tllama.native_config import _plain
from tllama.native_requests import NativeRequest
from tllama.native_tools import ToolCalls, full_calls, argument_object
from tllama.worker import WorkerProtocolError


CHAT = NativeRequest('chat', {'messages': []})
CONTEXT = ReplyContext('public-model')
CAPTURE = json.loads((Path(__file__).parent / 'fixtures/native_tools_3144.json').read_text())


def chunk(calls=None, finish=None, **delta):
    if calls is not None:
        delta['tool_calls'] = calls
    return {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


def call(index=0, *, identity='call-0', name='lookup_record', arguments='{}'):
    return {'index': index, 'id': identity, 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}


async def sequence(values):
    for value in values:
        if isinstance(value, BaseException):
            raise value
        yield value


async def projected(values, api):
    async with aclosing(sequence(values)) as upstream:
        async with aclosing(_read_events(upstream, CHAT)) as events:
            if api == 'ollama':
                return [json.loads(line) async for line in ollama_stream(events, CONTEXT)]
            return [frame[6:].strip() async for frame in openai_stream(events, CONTEXT)]


@pytest.mark.parametrize('probe', CAPTURE['probes'])
async def test_hardware_capture_preserves_native_ids_and_arguments_on_both_surfaces(probe):
    captured = CAPTURE['probes'][probe]
    full = read_result(captured['response'], CHAT)
    original = captured['response']['choices'][0]['message']['tool_calls']
    assert openai_response(full, CONTEXT)['choices'][0]['message']['tool_calls'] == original
    expected = [json.loads(c['function']['arguments']) for c in original]
    assert [c['function']['arguments'] for c in ollama_response(full, CONTEXT)['message']['tool_calls']] == expected
    outputs = await projected(captured['events'], 'ollama')
    calls = [c for output in outputs for c in output['message'].get('tool_calls', [])]
    native_ids = [c['id'] for payload in captured['events'] for choice in payload['choices']
                  for c in choice['delta'].get('tool_calls', []) if 'id' in c]
    assert [c['id'] for c in calls] == list(dict.fromkeys(native_ids))
    assert [c['id'] for c in ollama_response(full, CONTEXT)['message']['tool_calls']] == [c['id'] for c in original]
    assert [c['function']['arguments'] for c in calls] == expected
    assert [c['function']['index'] for c in calls] == list(range(len(calls)))
    assert outputs[-2]['done'] is False and outputs[-2]['message']['tool_calls'] == calls
    assert outputs[-1]['done'] and 'tool_calls' not in outputs[-1]['message']
    assert outputs[-1]['done_reason'] == 'stop'
    assert outputs[-1]['eval_count'] == captured['response']['usage']['completion_tokens']
    if probe == 'escaped-required':
        # The native engine already returned a literal backslash+n. No repair.
        assert expected[0]['key'].endswith('folder' + chr(92) + 'nline')
        assert chr(10) not in expected[0]['key']
    if probe == 'thinking-auto':
        assert ''.join(o['message'].get('thinking', '') for o in outputs)
        assert any(o['message'].get('thinking') for o in outputs[:-2])
    frames = await projected(captured['events'], 'openai')
    assert frames[-1] == '[DONE]'
    assembled = {}
    for frame in map(json.loads, frames[:-1]):
        for choice in frame['choices']:
            for delta in choice['delta'].get('tool_calls', []):
                item = assembled.setdefault(delta['index'], {'name': '', 'arguments': ''})
                if 'id' in delta:
                    item['id'] = delta['id']
                for key, value in delta.get('function', {}).items():
                    item[key] += value
    native = []
    for payload in captured['events']:
        for choice in payload['choices']:
            for delta in choice['delta'].get('tool_calls', []):
                if 'id' in delta:
                    native.append(delta['id'])
    assert [item['id'] for item in assembled.values()] == list(dict.fromkeys(native))
    assert [json.loads(item['arguments']) for item in assembled.values()] == expected
    assert all(item['name'] == 'lookup_record' for item in assembled.values())


async def test_ollama_waits_for_verified_transport_end_but_text_and_thinking_flow():
    progressed = []
    async def source():
        yield chunk([call(arguments='{}')], content='text', reasoning_content='thought')
        progressed.append('parseable')
        yield chunk(finish='tool_calls')
        progressed.append('finish')
        yield {'choices': [], 'usage': {'completion_tokens': 4}}
        progressed.append('transport-verified')
    async with aclosing(source()) as upstream:
        async with aclosing(_read_events(upstream, CHAT)) as events:
            async with aclosing(ollama_stream(events, CONTEXT)) as output:
                first = json.loads(await anext(output))
                assert first['message'] == {'role': 'assistant', 'content': 'text', 'thinking': 'thought'}
                assert progressed == []
                second = json.loads(await anext(output))
                assert second['message']['tool_calls'][0]['function']['arguments'] == {}
                assert progressed == ['parseable', 'finish', 'transport-verified']
                assert json.loads(await anext(output))['done']


async def test_interleaved_growing_names_become_appendable_openai_suffixes():
    values = [chunk([call(1, identity='b', name='look', arguments='{"b":'),
                     call(name='look', arguments='{"a":')]),
              chunk([{'index': 0, 'id': 'call-0', 'function': {'name': 'lookup', 'arguments': '1}'}},
                     {'index': 1, 'id': 'b', 'function': {'name': 'lookup_record', 'arguments': '2}'}}]),
              chunk([{'index': 0, 'function': {'name': 'lookup_record'}}], finish='tool_calls')]
    frames = await projected(values, 'openai')
    names = {0: '', 1: ''}
    for frame in map(json.loads, frames[:-1]):
        for choice in frame['choices']:
            for delta in choice['delta'].get('tool_calls', []):
                names[delta['index']] += delta.get('function', {}).get('name', '')
    assert names == {0: 'lookup_record', 1: 'lookup_record'}
    calls = (await projected(values, 'ollama'))[-2]['message']['tool_calls']
    assert [c['id'] for c in calls] == ['call-0', 'b']
    assert [c['function']['arguments'] for c in calls] == [{'a': 1}, {'b': 2}]


@pytest.mark.parametrize('tail', [
    [chunk(finish='length')],
    [chunk(finish='tool_calls'), WorkerProtocolError('missing [DONE]')],
    [WorkerProtocolError('native task failed')],
    [chunk([{'index': 0, 'function': {'arguments': 'trailing'}}], finish='tool_calls')],
])
async def test_parseable_prefix_cannot_leak_an_ollama_call_after_length_or_failure(tail):
    outputs = await projected([chunk([call()]), *tail], 'ollama')
    assert not any(o.get('message', {}).get('tool_calls') for o in outputs)
    if tail[0] == chunk(finish='length'):
        assert outputs[-1]['done_reason'] == 'length'
    else:
        assert 'error' in outputs[-1] and not any(o.get('done') for o in outputs)


async def test_openai_keeps_partial_arguments_and_length_without_claiming_a_completed_call():
    frames = await projected([chunk([call(arguments='{"key":')]), chunk(finish='length')], 'openai')
    assert frames[-1] == '[DONE]'
    assert json.loads(frames[-2])['choices'][0]['finish_reason'] == 'length'
    assert json.loads(frames[1])['choices'][0]['delta']['tool_calls'][0]['function']['arguments'] == '{"key":'
    result = read_result({'choices': [{'message': {'tool_calls': [{k: v for k, v in call(arguments='{').items() if k != 'index'}]},
                                      'finish_reason': 'length'}]}, CHAT)
    assert 'tool_calls' not in ollama_response(result, CONTEXT)['message']
    assert openai_response(result, CONTEXT)['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == '{'


@pytest.mark.parametrize('value', ['{"x":', '[]', 'null', '{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}', '{"x":{"a":1,"a":2}}'])
def test_unusable_arguments_are_typed_errors_without_json_repair(value):
    with pytest.raises(WorkerProtocolError):
        full_calls([{k: v for k, v in call(arguments=value).items() if k != 'index'}], 'tool_calls')


@pytest.mark.parametrize('value', [{'index': True}, {'index': -1}, {'index': 128}, {'index': 0, 'type': 'custom'},
    {'index': 0, 'function': {'arguments': {}}}, {'index': 0, 'function': []}, {'index': 0, 'id': None}])
def test_malformed_native_calls_fail_explicitly(value):
    with pytest.raises(WorkerProtocolError):
        ToolCalls().push([value])


def test_call_identity_name_indices_and_byte_budget_are_request_local(monkeypatch):
    from tllama import native_tools
    state = ToolCalls()
    state.push([call()])
    for value in [call(identity='changed'), call(1), call(name='other')]:
        with pytest.raises(WorkerProtocolError):
            state.push([value])
    for values in [[], [call(1)], [{'index': 0, 'function': {'arguments': '{}'}}]]:
        with pytest.raises(WorkerProtocolError):
            full = ToolCalls()
            full.push(values)
            full.finish('tool_calls')
    other = ToolCalls()
    other.push([call()])
    assert _plain(other.finish('tool_calls'))[0]['id'] == 'call-0'
    monkeypatch.setattr(native_tools, 'MAX_CALL_BYTES', 24)
    with pytest.raises(WorkerProtocolError, match='byte limit'):
        ToolCalls().push([call(arguments='x' * 25)])
    assert argument_object('{"key":"žluťoučký"}') == {'key': 'žluťoučký'}
