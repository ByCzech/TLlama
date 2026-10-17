"""Public smoke CLI owns real TCP workers; no model-quality simulation claim."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/tools-api-smoke.py'


@pytest.mark.parametrize('mode,ok,review,clients_only', [('normal', True, False, False), ('growing-name', True, False, False),
    ('semantic-bad', True, True, False), ('no-done', False, True, False), ('malformed-json', False, True, False),
    ('normal', True, False, True), ('growing-name', True, False, True), ('semantic-bad', True, True, True)])
def test_public_cli_full_cycle_raw_frames_cancellation_and_cleanup(tmp_path, monkeypatch, mode, ok, review, clients_only):
    if clients_only:
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            monkeypatch.delenv(key, raising=False)
        pytest.importorskip('openai')
        pytest.importorskip('ollama')
    root = tmp_path / 'stub'; root.mkdir()
    models = tmp_path / 'models/Local'; models.mkdir(parents=True)
    (models / 'a.gguf').write_bytes(b'fake weights')
    (models / 'a.toml').write_text('[llm]\nmodel="Local/a.gguf"\n[runtime]\ndevice="none"\nn_ctx=8192\n')
    executable = tmp_path / 'native-stub'
    executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name('native_worker_stub.py').read_text())
    executable.chmod(0o700)
    output = tmp_path / 'results'
    command = [sys.executable, str(SCRIPT), '--model', 'a', '--models-dir', str(models.parent),
               '--executable', str(executable), '--output', str(output), '--request-timeout', '5']
    if clients_only:
        command.append('--clients-only')
    env = {**os.environ, 'PYTHONPATH': str(SCRIPT.parents[1] / 'src'), 'STUB_TOOLS': mode,
           'STUB_ROOT': str(root), 'TLLAMA_MEMORY_SOURCE': 'log', 'STUB_PUBLIC_CTX': '4096'}
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=40)
    assert run.returncode == (0 if ok else 1), run.stdout + run.stderr
    raw = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert raw[-1]['protocol_ok'] is ok and raw[-1]['review_required'] is review
    assert raw[-1]['public_api'] == 'exercised' and raw[-1]['residency'] == []
    assert run.stdout == (output / 'summary.jsonl').read_text()
    assert not any('public-event' in line for line in (output / 'summary.jsonl').read_text().splitlines())
    assert raw[-2]['stage'] == 'worker-closed' and raw[-2]['clean']
    ready = next(r for r in raw if r['stage'] == 'ready')
    assert ready['argv'][ready['argv'].index('--ctx-size') + 1] == '4096'
    if ok and not review and clients_only:
        assert len([r for r in raw if r['stage'] == 'sdk-assembled']) == 4
        assert len([r for r in raw if r['stage'] == 'sdk-followup']) == 4
        assert all(r['semantic_match'] for r in raw if r['stage'] in {'sdk-assembled', 'sdk-followup'})
        assert not any(r['stage'] == 'assembled' for r in raw)
    elif ok and not review:
        assert len([r for r in raw if r['stage'] == 'assembled']) == 22
        assert len([r for r in raw if r['stage'] == 'followup']) == 20
        assert all(r['semantic_match'] for r in raw if r['stage'] in {'assembled', 'followup'})
        assert any(r['stage'] == 'stream-cancel' and r['partial_arguments_seen'] for r in raw)
        assert (root / 'cancelled').read_text().splitlines() == ['"tools"']
        assert next(r for r in raw if r['stage'] == 'after-cancel')['pid'] == ready['pid']
        request_records = [json.loads(line) for line in (root / 'requests').read_text().splitlines()]
        replays = [m for r in request_records for m in r.get('body', {}).get('messages', []) if m.get('tool_calls')]
        assert any(isinstance(m['tool_calls'][0]['function']['arguments'], dict) for m in replays)
        assert any(isinstance(m['tool_calls'][0]['function']['arguments'], str) for m in replays)
    before = (root / 'launch').read_text()
    repeat = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert repeat.returncode != 0 and (root / 'launch').read_text() == before
