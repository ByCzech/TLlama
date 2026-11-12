"""Measured drafting reaches verbose stderr without changing replies or usage."""

import asyncio
from copy import deepcopy
import json
import locale
from types import SimpleNamespace

import httpx
import pytest

from tllama.cli_metrics import statistics
from tllama.cli_run import RunSession
from tllama.client import ClientError, TLlamaClient
from tllama.main import app
from test_helper_definitions import publish, reference
from test_native_manager import native_manager
from test_native_public_api import public


def measured(**counts):
    return {'tllama': {'speculative': counts}}


def rows(data):
    return {line.strip().split(':', 1)[0]: line.strip().split(':', 1)[1].strip()
            for line in statistics(data, elapsed=1).splitlines()[1:]}


@pytest.mark.parametrize('counts,expected', [
    ({}, {}), ({'draft_tokens': None, 'accepted_tokens': None}, {}),
    ({'draft_tokens': 0, 'accepted_tokens': 0}, {'Drafted':'0 tokens', 'Draft accepted':'0 tokens'}),
    ({'draft_tokens': 12}, {'Drafted':'12 tokens'}),
    ({'accepted_tokens': 6}, {'Draft accepted':'6 tokens'}),
    ({'draft_tokens':12, 'accepted_tokens':0},
        {'Drafted':'12 tokens', 'Draft accepted':'0 tokens', 'Draft acceptance':'0.00%'}),
    ({'draft_tokens':12, 'accepted_tokens':6},
        {'Drafted':'12 tokens', 'Draft accepted':'6 tokens', 'Draft acceptance':'50.00%'}),
    ({'draft_tokens':3, 'accepted_tokens':1},
        {'Drafted':'3 tokens', 'Draft accepted':'1 tokens', 'Draft acceptance':'33.33%'}),
])
def test_only_measured_counts_and_defined_ratio_are_displayed(counts, expected):
    actual = rows(measured(**counts))
    actual.pop('Client total')
    assert actual == expected


@pytest.mark.parametrize('extension', [None, True, [], 'bad', {'speculative': None},
                                    {'speculative': []}, {'speculative': 'bad'}])
def test_absent_or_malformed_optional_blocks_do_not_break_output(extension):
    assert rows({'tllama':extension}) == {'Client total':'1.000 s'}


@pytest.mark.parametrize('bad', [True, -1, '6', 1.5, float('nan'), float('inf'), {}])
def test_invalid_optional_counter_is_omitted_without_inventing_acceptance(bad):
    assert rows(measured(draft_tokens=12, accepted_tokens=bad)) == {
        'Client total':'1.000 s', 'Drafted':'12 tokens'}
    assert rows(measured(draft_tokens=bad, accepted_tokens=6)) == {
        'Client total':'1.000 s', 'Draft accepted':'6 tokens'}


@pytest.mark.parametrize('drafted,accepted', [(0,1), (2,3)])
def test_contradictory_counters_do_not_become_valid_statistics(drafted, accepted):
    assert rows(measured(draft_tokens=drafted, accepted_tokens=accepted)) == {'Client total':'1.000 s'}


def test_percentage_uses_counters_and_never_openai_prediction_fields_or_stale_ratio():
    data = measured(draft_tokens=12, accepted_tokens=6, acceptance=0.99)
    data.update(eval_count=10, eval_duration=2_000_000_000, usage={'completion_tokens_details':{
        'accepted_prediction_tokens':100, 'rejected_prediction_tokens':20}})
    before = deepcopy(data)
    result = rows(data)
    assert result['Draft acceptance'] == '50.00%'
    assert result['Generated'] == '10 tokens' and result['Generation'] == '2.000 s (5.00 tok/s)'
    assert data == before
    for other in ({'acceptance':0.5}, {}):
        assert 'Draft acceptance' not in rows(measured(**other))
    assert rows({'usage':data['usage']}) == {'Client total':'1.000 s'}


def test_speculative_counts_and_percentage_follow_client_locale(monkeypatch):
    convention = dict(locale.localeconv(), decimal_point=',', thousands_sep='\u202f', grouping=[3,0])
    monkeypatch.setattr(locale, 'localeconv', lambda: convention)
    result = rows(measured(draft_tokens=24690, accepted_tokens=12345))
    assert result['Drafted'] == '24\u202f690 tokens'
    assert result['Draft accepted'] == '12\u202f345 tokens'
    assert result['Draft acceptance'] == '50,00%'


@pytest.mark.parametrize('json_mode', [False, True])
@pytest.mark.parametrize('verbose', [False, True])
async def test_only_terminal_counters_are_printed_once_and_json_is_preserved(capsys, json_mode, verbose):
    events = [
        {'message':{'role':'assistant', 'content':'answer'}, 'done':False,
         **measured(draft_tokens=99, accepted_tokens=99)},
        {'message':{'role':'assistant', 'content':''}, 'done':True,
         **measured(draft_tokens=12, accepted_tokens=6, acceptance=0.5)},
    ]
    async def chat(*args, **kwargs):
        for event in events:
            yield event
    await RunSession('model', 'completion', verbose=verbose, as_json=json_mode).submit(
        SimpleNamespace(chat=chat), 'prompt')
    output = capsys.readouterr()
    assert output.err.count('Drafted:') == int(verbose)
    assert output.err.count('Draft accepted:') == int(verbose)
    if verbose:
        assert '12 tokens' in output.err and '6 tokens' in output.err and '50.00%' in output.err
        assert '99 tokens' not in output.err
    if json_mode:
        assert [json.loads(line) for line in output.out.splitlines()] == events
    else:
        assert output.out == 'answer\n'


@pytest.mark.parametrize('cancel', [False, True])
async def test_partial_failed_request_does_not_print_successful_draft_metrics(capsys, cancel):
    async def chat(*args, **kwargs):
        yield {'message':{'role':'assistant', 'content':'partial'}, 'done':False,
               **measured(draft_tokens=12, accepted_tokens=6)}
        raise asyncio.CancelledError() if cancel else ClientError('failed')
    with pytest.raises(asyncio.CancelledError if cancel else ClientError):
        await RunSession('model', 'completion', verbose=True).submit(SimpleNamespace(chat=chat), 'prompt')
    assert 'Statistics:' not in capsys.readouterr().err


@pytest.mark.parametrize('json_mode', [False, True])
async def test_real_dflash_worker_through_client_prints_measured_stats_and_off_does_not_reuse_them(
        public, monkeypatch, tmp_path, capsys, json_mode):
    manager, _, name, toml = public
    _, draft = publish(manager, tmp_path, 'dflash', count=2)
    original = toml.read_text() + '\n' + reference('draft', draft) + '[runtime]\nspec_type="draft-dflash"\n'
    toml.write_text(original)
    monkeypatch.setenv('STUB_PUBLIC_DFLASH', '1')
    session = RunSession(name, 'completion', verbose=True, as_json=json_mode)
    async with TLlamaClient('http://public', transport=httpx.ASGITransport(app=app)) as client:
        await session.submit(client, 'prompt')
        enabled = capsys.readouterr()
        assert enabled.err.count('Drafted:') == enabled.err.count('Draft accepted:') == 1
        assert '12 tokens' in enabled.err and '6 tokens' in enabled.err and '50.00%' in enabled.err
        if json_mode:
            terminal = json.loads(enabled.out.splitlines()[-1])
            assert terminal['done'] and terminal['tllama']['speculative']['acceptance'] == 0.5
        else:
            assert 'Ahoj světe' in enabled.out and 'Statistics:' not in enabled.out
        session.controls['options'] = {'draft_num_predict':0}
        await session.submit(client, 'off')
        disabled = capsys.readouterr()
        assert 'Statistics:' in disabled.err and 'Draft' not in disabled.err
        assert 'Generated:' in disabled.err and len(manager.test.ledger) == 2
        assert '--spec-draft-model' not in manager.test.ledger[-1].command
    assert toml.read_text() == original
