"""Verbose statistics keep units, missing measurements and JSON boundaries."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tllama.cli import build_parser
from tllama.cli_metrics import statistics
from tllama.cli_repl import command
from tllama.cli_run import RunSession
from tllama.client import ClientError


def test_statistics_use_evaluated_not_cached_tokens_for_rates():
    text = statistics({'total_duration': 3_000_000_000, 'load_duration': 1_000_000_000,
                       'prompt_eval_count': 20, 'prompt_eval_cached_count': 100,
                       'prompt_eval_duration': 200_000_000, 'eval_count': 8,
                       'eval_duration': 2_000_000_000, 'done_reason': 'length'}, elapsed=3.5, first_text=1.5)
    assert '3.500 s' in text and '1.500 s' in text
    assert 'Server total:' in text and '3.000 s' in text
    assert 'Server preparation:' in text and '1.000 s' in text
    assert 'Prompt cached:' in text and '100 tokens' in text
    assert '0.200 s (100.00 tok/s)' in text and '2.000 s (4.00 tok/s)' in text
    assert 'Stop reason:' in text and 'length' in text


def test_missing_measurements_are_omitted_and_zero_is_not_missing():
    assert statistics({}, elapsed=1) == 'Statistics:\n  Client total:          1.000 s'
    text = statistics({'prompt_eval_count': 0, 'eval_count': 0, 'eval_duration': 0,
                       'prompt_eval_cached_count': None}, elapsed=0)
    assert '0 tokens' in text and '0.000 s' in text and 'tok/s' not in text
    assert 'Prompt cached' not in text and 'First text' not in text and 'Server preparation' not in text


@pytest.mark.parametrize('bad', [True, -1, '10', float('nan'), float('inf'), 10**1000])
def test_bad_or_unrepresentable_durations_cannot_break_output(bad):
    text = statistics({'eval_duration': bad, 'eval_count': 3}, elapsed=1)
    assert 'Generation:' not in text and 'tok/s' not in text


@pytest.mark.parametrize('json_mode,hidden', [(False, False), (True, False), (False, True)])
async def test_session_verbose_uses_terminal_stats_and_first_real_text(monkeypatch, capsys, json_mode, hidden):
    clock = iter([10.0, 10.75, 12.0])
    monkeypatch.setattr('tllama.cli_run.time', SimpleNamespace(perf_counter=lambda: next(clock)))
    events = [
        {'message': {'role': 'assistant', 'content': ''}, 'done': False},
        {'message': {'role': 'assistant', 'thinking': 'reason'}, 'done': False},
        {'message': {'role': 'assistant', 'content': 'answer'}, 'done': False},
        {'message': {'role': 'assistant', 'content': ''}, 'done': True, 'eval_count': 4,
         'eval_duration': 1_000_000_000, 'prompt_eval_count': 5},
    ]
    async def chat(*args, **kwargs):
        for event in events:
            yield event
    session = RunSession('model', 'completion', verbose=True, as_json=json_mode, hide_thinking=hidden)
    await session.submit(SimpleNamespace(chat=chat), 'prompt')
    output = capsys.readouterr()
    assert 'First text:' in output.err and '0.750 s' in output.err
    assert 'Client total:' in output.err and '2.000 s' in output.err
    assert '1.000 s (4.00 tok/s)' in output.err
    if json_mode:
        assert [json.loads(line) for line in output.out.splitlines()] == events
    else:
        assert output.out == 'answer\n'


@pytest.mark.parametrize('outcome', ['error', 'cancel'])
async def test_partial_request_does_not_claim_completed_statistics(capsys, outcome):
    async def chat(*args, **kwargs):
        yield {'message': {'role': 'assistant', 'content': 'partial'}, 'done': False}
        raise (ClientError('failed') if outcome == 'error' else asyncio.CancelledError())
    session = RunSession('model', 'completion', verbose=True)
    with pytest.raises(ClientError if outcome == 'error' else asyncio.CancelledError):
        await session.submit(SimpleNamespace(chat=chat), 'prompt')
    assert 'Statistics:' not in capsys.readouterr().err and session.messages == []


@pytest.mark.parametrize('mode', ['embedding', 'decision'])
async def test_verbose_switch_is_local_for_non_generation_modes(capsys, mode):
    session = RunSession('model', mode)
    await command(None, session, '/set verbose')
    assert session.verbose and session.controls == {}
    await command(None, session, '/set quiet')
    assert not session.verbose and session.controls == {}


async def test_initial_preload_time_is_separate(monkeypatch, capsys):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr('tllama.cli_run.time', SimpleNamespace(perf_counter=lambda: next(clock)))
    async def preload(*args, **kwargs):
        return True
    await RunSession('model', 'completion', verbose=True).preload(SimpleNamespace(preload=preload))
    text = capsys.readouterr().err
    assert text == 'Preload request: 2.500 s\n' and 'First text' not in text


@pytest.mark.parametrize('flag', ['--verbose', '-v'])
def test_run_verbose_flags(flag):
    assert build_parser().parse_args(['run', 'model', flag]).verbose


@pytest.mark.parametrize('mode,payload,text', [
    ('embedding', {'embeddings': [[1.0]], 'prompt_eval_count': 3, 'total_duration': 1_000_000}, 'text'),
    ('decision', {'answers': {}, 'usage': {'input_tokens': 7, 'output_tokens': 0}},
     '{"state":{},"questions":{}}'),
])
async def test_mode_specific_verbose_does_not_invent_generation_metrics(capsys, mode, payload, text):
    async def request(*args, **kwargs):
        return payload
    client = SimpleNamespace(embed=request, decision=request)
    await RunSession('model', mode, verbose=True).submit(client, text)
    output = capsys.readouterr()
    assert json.loads(output.out) == payload
    assert 'Statistics:' in output.err and 'tok/s' not in output.err and 'First text' not in output.err
    if mode == 'decision':
        assert 'Input:' in output.err and '7 tokens' in output.err
        assert 'Output:' in output.err and '0 tokens' in output.err
