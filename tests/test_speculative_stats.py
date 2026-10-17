"""Request-scoped speculative counters across native and public transports."""

import asyncio
import json

import pytest

from tllama.native_api import (RequestStats, NativeResult, ollama_response, openai_response,
                               ollama_stream, openai_stream)
from tllama.worker import WorkerProtocolError

from test_native_api import CHAT, RAW, USAGE, TIMINGS, collected, context, read_result, chat_result, sequence
from test_native_public_api import public, request_body
from test_native_manager import native_manager


COUNTERS = {'draft_n': 12, 'draft_n_accepted': 6}
EXTENSION = {'tllama': {'speculative': {'draft_tokens': 12, 'accepted_tokens': 6, 'acceptance': 0.5}}}


@pytest.mark.parametrize('native,expected', [({}, {}), ({'draft_n': None, 'draft_n_accepted': None}, {}),
    ({'draft_n': 0, 'draft_n_accepted': 0}, {'draft_tokens': 0, 'accepted_tokens': 0}),
    ({'draft_n': 12, 'draft_n_accepted': 0}, {'draft_tokens': 12, 'accepted_tokens': 0, 'acceptance': 0.0}),
    ({'draft_n': 12}, {'draft_tokens': 12}),
    ({'draft_n_accepted': 6}, {'accepted_tokens': 6})])
def test_unknown_partial_and_measured_zero_counters_are_distinct(native, expected):
    stats = RequestStats.read({'timings': native})
    assert stats.extensions() == ({'tllama': {'speculative': expected}} if expected else {})
    assert stats.ollama() == stats.openai() == {}


@pytest.mark.parametrize('native', [{'draft_n': True}, {'draft_n': -1}, {'draft_n': 1.5},
    {'draft_n_accepted': '2'}, {'draft_n_accepted': False}, {'draft_n_accepted': -1},
    {'draft_n': 0, 'draft_n_accepted': 1}, {'draft_n': 2, 'draft_n_accepted': 3}])
def test_invalid_native_counters_do_not_become_public_measurements(native):
    with pytest.raises(WorkerProtocolError):
        RequestStats.read({'timings': native})


def test_partial_stream_merge_validates_combined_counters_and_preserves_zero():
    stats = RequestStats.read({'timings': {'draft_n': 12}})
    assert stats.merge(RequestStats.read({'timings': {'draft_n_accepted': 0}})).extensions() == {
        'tllama': {'speculative': {'draft_tokens': 12, 'accepted_tokens': 0, 'acceptance': 0.0}}}
    with pytest.raises(WorkerProtocolError):
        stats.merge(RequestStats.read({'timings': {'draft_n_accepted': 13}}))


@pytest.mark.parametrize('raw', [False, True])
def test_export_preserves_native_and_public_token_count_meanings(raw):
    payload = ({'content': 'answer', 'stop': True, 'stop_type': 'limit',
                'tokens_evaluated': 25, 'tokens_predicted': 10} if raw else chat_result())
    result = read_result({**payload, 'timings': {**TIMINGS, **COUNTERS}}, RAW if raw else CHAT)
    ollama = ollama_response(result, context(), kind='completion' if raw else 'chat')
    assert ollama['eval_count'] == 10 and ollama['prompt_eval_count'] == 4
    assert ollama['tllama'] == EXTENSION['tllama']
    assert openai_response(result, context())['tllama'] == EXTENSION['tllama']
    assert openai_response(result, context())['usage'] == USAGE
    # No speculative count is inferred from accepted_prediction_tokens: that
    # standard field describes a different OpenAI prediction feature.
    assert RequestStats.read({'usage': {'completion_tokens_details': {
        'accepted_prediction_tokens': 6, 'rejected_prediction_tokens': 6}}}).extensions() == {}


@pytest.mark.parametrize('include_usage', [False, True])
async def test_late_native_counters_export_once_independently_of_openai_usage(include_usage):
    values = [{'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}]},
              {'choices': [], 'usage': USAGE, 'timings': {**TIMINGS, **COUNTERS}}]
    events = await collected(values)
    frames = [f async for f in openai_stream(sequence(*events), context(), include_usage=include_usage)]
    assert frames[-1] == 'data: [DONE]\n\n'
    bodies = [json.loads(frame[6:]) for frame in frames[:-1]]
    exported = [b for b in bodies if 'tllama' in b]
    assert len(exported) == 1 and exported[0]['tllama'] == EXTENSION['tllama']
    assert exported[0]['choices'][0]['finish_reason'] == 'stop'
    if include_usage:
        assert bodies[-1]['usage'] == USAGE and 'tllama' not in bodies[-1]


async def test_simultaneous_requests_do_not_share_native_counters():
    async def run(counters):
        return await collected([{'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'stop'}]},
                                {'choices': [], 'timings': counters}])
    first, second = await asyncio.gather(run(COUNTERS), run({}))
    assert first[-1].stats.extensions() == EXTENSION
    assert second[-1].stats.extensions() == {}


async def test_ollama_tool_calls_do_not_duplicate_terminal_statistics():
    result = NativeResult(tool_calls=({'id': 'call-a', 'type': 'function',
        'function': {'name': 'lookup', 'arguments': '{}'}},), finish_reason='tool_calls',
        terminal=True, stats=RequestStats.read({'timings': COUNTERS}))
    bodies = [json.loads(line) async for line in ollama_stream(sequence(result), context())]
    assert len(bodies) == 2 and bodies[0]['message']['tool_calls']
    assert not bodies[0]['done'] and 'tllama' not in bodies[0]
    assert bodies[1]['done'] and bodies[1]['tllama'] == EXTENSION['tllama']


@pytest.mark.parametrize('presenter', [ollama_stream, openai_stream])
async def test_incomplete_stream_cannot_publish_successful_speculative_statistics(presenter):
    values = [{'choices': [{'delta': {'content': 'partial'}}], 'timings': COUNTERS}]
    from tllama.native_api import _read_events
    frames = [f async for f in presenter(_read_events(sequence(*values), CHAT), context())]
    assert 'partial' in ''.join(frames) and 'error' in frames[-1]
    assert not any('tllama' in f or '[DONE]' in f or '"done": true' in f for f in frames)


@pytest.mark.parametrize('path,raw', [('/api/chat', False), ('/api/generate', False),
                                    ('/api/generate', True), ('/v1/chat/completions', False)])
@pytest.mark.parametrize('streaming', [False, True])
async def test_real_native_transport_through_public_routes(public, monkeypatch, path, raw, streaming):
    manager, client, name, toml = public
    monkeypatch.setenv('STUB_PUBLIC_MTP', '1')
    toml.write_text(toml.read_text() + '\n[runtime]\nspec_type="draft-mtp"\n')
    body = request_body(name, path, streaming)
    if raw:
        body['raw'] = True
    response = await client.post(path, json=body)
    assert response.status_code == 200
    if streaming:
        bodies = [json.loads(line[6:] if path.startswith('/v1/') else line)
                  for line in response.text.splitlines() if line and line != 'data: [DONE]']
        exported = [b for b in bodies if 'tllama' in b]
        assert len(exported) == 1
        reply = exported[0]
    else:
        reply = response.json()
    assert reply['tllama'] == EXTENSION['tllama']
    if not path.startswith('/v1/'):
        assert reply['eval_count'] == 10 and reply['done']
    assert manager.native_snapshot()[0]['leases'] == 0 and len(manager.test.ledger) == 1
