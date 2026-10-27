"""Trace the same inference through raw native XML and both public adapters."""

from copy import deepcopy
import json
import runpy
import subprocess
import sys

import pytest

from test_functional_qualification_smoke import SCRIPT, arguments, setup_files
from test_native_memory_api import payload
from test_native_memory_locality import extended
from test_native_weight_storage import weights
from test_native_tensor_diagnostics import diagnosed
from tllama.worker import WorkerError


api = runpy.run_path(str(SCRIPT.with_name('tools-api-smoke.py')))


@pytest.mark.parametrize('fault,ok,protocol,diagnosis', [
    ('normal', True, True, 'exact'),
    ('generated', False, True, 'generated-value-mismatch'),
    ('parser', False, True, 'native-parser-difference'),
    ('bad-prompt', False, True, 'input-needs-review'),
    ('unknown-xml', False, True, 'raw-format-needs-review'),
    ('missing-debug', False, False, None),
    ('no-raw', False, False, None),
    ('no-grammar', False, False, None),
])
def test_focused_capture_uses_same_native_response_and_preserves_failures(tmp_path, diagnosed, fault, ok, protocol, diagnosis):
    root, models, executable, env, definition = setup_files(tmp_path, diagnosed, 'tools')
    env['STUB_TOOLS_DEBUG'] = fault
    command = [sys.executable, str(SCRIPT), *arguments(tmp_path, 'tools-escape',
        '--executable', str(executable), '--transport', 'tcp', '--load-timeout', '5', '--request-timeout', '5')]
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
    result = json.loads((tmp_path / 'capture/result.json').read_text())
    assert (result['ok'], result['protocol_ok'], run.returncode) == (ok, protocol, 0 if ok else 1), (result, run.stderr)
    rows = [json.loads(line) for line in (tmp_path / 'capture/raw.jsonl').read_text().splitlines()]
    debug = [row for row in rows if row['stage'] == 'native-debug-response']
    assert len(debug) == (4 if protocol else 1)
    assert all(row['request']['verbose'] is True and row['request']['stream'] is False for row in debug)
    assert all('parse_tool_calls' not in row['request'] and 'grammar' not in row['request'] for row in debug)
    assert not any(row['stage'] in {'followup', 'stream-cancel', 'after-cancel'} for row in rows)
    if protocol:
        comparisons = [row for row in rows if row['stage'] == 'escape-comparison']
        assert len(comparisons) == 4
        assert {row['api'] for row in comparisons} == {'ollama', 'openai'}
        assert all(row['native_public_match'] for row in comparisons)
        assert comparisons[0]['diagnosis'] == diagnosis
        if fault in {'generated', 'parser'}:
            controls = [row for row in comparisons if row['probe'] == 'literal-backslash-n']
            assert all(row['diagnosis'] == 'exact' for row in controls)
            assert all(row['expected_key'].endswith('folder' + chr(92) + 'nline') for row in controls)
            assert len(result['observations']) == 2
    assert all(row['ok'] for row in rows if row['stage'] in {'worker-cleanup', 'worker-exit'})
    assert len([row for row in rows if row['stage'] == 'qualification-memory']) == 2
    assert (models / 'a.toml').read_text() == definition
    requests = [json.loads(line)['body'] for line in (root / 'requests').read_text().splitlines()
                if json.loads(line)['path'] == '/v1/chat/completions']
    assert len(requests) == len(debug)
    assert all(not body['stream'] and body['verbose'] for body in requests)
    assert all(body.get('tool_choice', 'auto') in {'auto', 'required'} for body in requests)
    summary = (tmp_path / 'capture/summary.jsonl').read_text()
    assert 'native-debug-response' not in summary and '"__verbose":' not in summary
    before = (root / 'launch').read_bytes()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert repeat.returncode != 0 and (root / 'launch').read_bytes() == before


@pytest.mark.parametrize('key', ['a\nb', 'a' + chr(92) + 'nb', '"quote"' + chr(92) + 'folder\nline', '\nvalue\n'])
def test_raw_xml_and_json_keep_distinct_newline_and_backslash_values(key):
    ref = api['reference']
    body = {'messages': [{'role': 'user', 'content': 'keys: ' + json.dumps([key])}]}
    payload = {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'native-call', 'type': 'function', 'function': {
            'name': 'lookup_record', 'arguments': json.dumps({'key': key})}}]}, 'finish_reason': 'tool_calls'}],
        '__verbose': {'content': '<tool_call>\n<function=lookup_record>\n<parameter=key>\n' + key
                       + '\n</parameter>\n</function>\n</tool_call>',
                      'prompt': body['messages'][0]['content'], 'generation_settings': {'grammar': 'test'}}}
    result = ref.full_result(payload)
    inspected = api['inspect_escape'](body, payload, result, key)
    assert inspected['diagnosis'] == 'exact'
    assert inspected['raw_xml_key'] == inspected['native_key'] == key
    changed = deepcopy(result)
    changed['arguments'][0]['arguments']['key'] += 'changed'
    with pytest.raises(WorkerError, match='Native/public'):
        api['inspect_escape'](body, payload, changed, key)


def test_original_prompt_is_unchanged_and_control_uses_literal_backslash_n():
    ref = api['reference']
    original, control = api['escape_probes'](256)
    assert original == next(case for case in ref.probes(256) if case[0] == 'escaped-required')
    assert original[2] != control[2]
    assert json.loads(control[1]['messages'][-1]['content'].split(': ', 1)[1]) == control[2]
    assert '\n' in original[2][0] and '\n' not in control[2][0]
