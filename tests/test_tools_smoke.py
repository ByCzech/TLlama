"""Independent diagnostic assembly and real owned-subprocess Tools transport."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.worker import WorkerError


SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'tools-smoke.py'
spec = importlib.util.spec_from_file_location('tools_smoke',SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def chunk(calls=None, finish=None, **delta):
    if calls is not None:
        delta['tool_calls'] = calls
    return {'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}


def test_interleaved_calls_keep_indices_ids_unicode_and_json_escapes_separate():
    a = smoke.Assembly()
    for index in (1,0):
        a.push(chunk([{'index':index,'id':f'call-{index}','type':'function','function':{'name':'lookup_record'}}]))
    texts = [json.dumps({'key':smoke.ESCAPED},ensure_ascii=False),'{}']
    for offset in range(max(map(len,texts))):
        for index,text in enumerate(texts):
            if offset < len(text):
                a.push(chunk([{'index':index,'function':{'arguments':text[offset]}}]))
    a.push(chunk(finish='tool_calls',reasoning_content='reasoning remains separate'))
    a.push({'choices':[],'usage':{'completion_tokens':100}})
    result = a.result()
    assert [c['id'] for c in result['message']['tool_calls']] == ['call-0','call-1']
    assert [r['arguments'] for r in result['arguments']] == [{'key':smoke.ESCAPED},{}]
    assert result['message']['reasoning_content'] == 'reasoning remains separate'
    assert result['argument_fragments'] == sum(map(len,texts))


@pytest.mark.parametrize('value',['{"key":','{"key":NaN}','{"key":1,"key":2}','[]','null',''])
def test_incomplete_or_ambiguous_arguments_are_never_repaired(value):
    with pytest.raises(ValueError):
        smoke.json_arguments(value)


@pytest.mark.parametrize('bad',[
    {'index':True}, {'index':-1}, {'index':64}, {'index':0,'function':{'arguments':{}}},
    {'index':0,'function':[]}, {'index':0,'type':'custom'},
])
def test_invalid_native_delta_has_explicit_failure(bad):
    with pytest.raises(WorkerError):
        smoke.Assembly().push(chunk([bad]))


def test_length_terminal_retains_unfinished_arguments_without_execution():
    a = smoke.Assembly()
    a.push(chunk([{'index':0,'id':'call-a','function':{'name':'lookup_record','arguments':'{"key":'}}]))
    a.push(chunk(finish='length'))
    result = a.result()
    assert result['arguments'][0]['valid'] is False
    assert result['message']['tool_calls'][0]['function']['arguments'] == '{"key":'
    assert smoke.semantics(result,['alpha']) is False


def test_same_index_cannot_silently_switch_identity_or_accept_post_terminal_data():
    a = smoke.Assembly()
    a.push(chunk([{'index':0,'id':'call-a'}]))
    with pytest.raises(WorkerError,match='identity changed'):
        a.push(chunk([{'index':0,'id':'call-b'}]))
    a.push(chunk(finish='stop'))
    with pytest.raises(WorkerError,match='after native finish'):
        a.push(chunk(content='late'))


def test_native_name_revisions_replace_cumulative_names_instead_of_concatenating():
    a = smoke.Assembly()
    for name in ('look','lookup','lookup_record'):
        a.push(chunk([{'index':0,'id':'call-a','function':{'name':name}}]))
    a.push(chunk([{'index':0,'function':{'arguments':'{}'}}],finish='tool_calls'))
    result = a.result()
    assert result['message']['tool_calls'][0]['function']['name'] == 'lookup_record'
    assert result['arguments'][0]['name_revisions'] == 3
    b = smoke.Assembly()
    b.push(chunk([{'index':0,'function':{'name':'lookup'}}]))
    with pytest.raises(WorkerError,match='growing prefix'):
        b.push(chunk([{'index':0,'function':{'name':'other'}}]))


@pytest.mark.parametrize('mode,ok,review',[
    ('normal',True,False), ('semantic-bad',True,True), ('malformed-json',True,True),
    ('truncated',False,True), ('http400',True,True), ('growing-name',True,False),
    ('no-done',False,True), ('http500',False,True),
])
def test_cli_captures_raw_evidence_semantic_review_cancellation_and_cleanup(tmp_path, mode, ok, review):
    root = tmp_path/'stub';root.mkdir()
    models = tmp_path/'models'/'Local';models.mkdir(parents=True)
    (models/'a.gguf').write_bytes(b'fake weights')
    (models/'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_ctx=8192\n')
    executable = tmp_path/'native-stub'
    executable.write_text(f'#!{sys.executable}\n'+Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path/'results'
    command = [sys.executable,str(SCRIPT),'--model','a','--models-dir',str(models.parent),
               '--executable',str(executable),'--output',str(output)]
    environment = {**os.environ,'PYTHONPATH':str(SCRIPT.parents[1]/'src'),
                   'STUB_TOOLS':mode,'STUB_ROOT':str(root),'TLLAMA_MEMORY_SOURCE':'log','STUB_PUBLIC_CTX':'4096'}
    run = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=30)
    assert run.returncode == (0 if ok else 1),run.stdout+run.stderr
    raw = [json.loads(l) for l in (output/'raw.jsonl').read_text().splitlines()]
    concise = [json.loads(l) for l in (output/'summary.jsonl').read_text().splitlines()]
    assert run.stdout == (output/'summary.jsonl').read_text()
    assert not any(r['stage'] in {'request','native-event','native-response'} for r in concise)
    assert raw[-1]['protocol_ok'] is ok and raw[-1]['review_required'] is review
    assert raw[-1]['public_api'] == 'not_exercised'
    assert next(r for r in raw if r['stage']=='ready')['context'] == 4096
    closed = next(r for r in raw if r['stage']=='worker-closed')
    assert closed['clean'] and closed['directory_removed']
    assert raw[-2]['residency'] == []
    if mode in {'normal','growing-name'}:
        comparisons = [r for r in raw if r['stage']=='stream-comparison']
        assert len(comparisons)==6 and all(r['tool_calls_match'] for r in comparisons)
        answers = [r for r in raw if r['stage']=='assembled' and r['phase']=='followup']
        assert len(answers)==10 and all(r['semantic_match'] for r in answers)
        cancellation = next(r for r in raw if r['stage']=='stream-cancel')
        assert cancellation['partial_arguments_seen'] and not cancellation['tool_dispatched']
        assert (root/'cancelled').read_text().splitlines() == ['"tools"']
        assert next(r for r in raw if r['stage']=='after-cancel')['pid'] == closed['pid']
        launch = json.loads((root/'launch').read_text().splitlines()[0])
        assert launch['argv'][launch['argv'].index('--ctx-size')+1] == '4096'
        requests = [json.loads(l) for l in (root/'requests').read_text().splitlines()]
        assert all(r['body']['model']=='tllama-worker' and 'authorization' in r['header_names']
                   for r in requests if r['path']=='/v1/chat/completions')
        assert any(r['body'].get('tool_choice')=='none' and r['body']['messages'][-1]['role']=='tool'
                   for r in requests if r['path']=='/v1/chat/completions')
    elif mode=='semantic-bad':
        assert any('semantic_mismatch' in r for r in raw[-1]['observations'])
    elif mode=='malformed-json':
        assert any(not a['valid'] for r in raw if r['stage']=='assembled' for a in r['arguments'])
    before = (root/'launch').read_text()
    repeat = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=10)
    assert repeat.returncode==2 and (root/'launch').read_text()==before
