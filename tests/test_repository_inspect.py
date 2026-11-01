"""Sharing/counts stay read-only, including protected or damaged repositories."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tllama.daemon import main
from tllama.model_definitions import create_definition
from tllama.repository_inspect import inspect_repository
from tllama.repository_pins import Reservation
from tllama.repository_store import RepositoryStore
from test_repository_cleanup import repo, text_for
from test_repository_check import before


def test_stats_count_shared_shards_once_and_never_touch_repository(repo):
    shared, orphan = repo.test_publish(count=2, upload=True), repo.test_publish('qwen3')
    create_definition(repo, 'a', text_for(shared.relative), {})
    create_definition(repo, 'b', text_for(shared.relative), {})
    old = before(repo.models_dir)
    report = inspect_repository(repo.repository)
    counts = report['summary']
    assert report['ok'] and report['operation'] == 'stats' and counts['shared_groups'] == 1
    assert counts['valid_group_weight_bytes'] == sum(path.stat().st_size for path in [*shared.paths, *orphan.paths])
    # Upload aliases do not add another copy to managed group byte counts.
    assert counts['valid_group_unique_inode_bytes'] == counts['valid_group_weight_bytes']
    assert counts['prune_candidate_weight_bytes'] == orphan.paths[0].stat().st_size
    assert counts['groups_by_usage'] == {'referenced':1,'reserved':0,'orphan':1,'unknown':0}
    assert all(row['weight_bytes'] == sum(path.stat().st_size for path in shared.paths) for row in report['definitions'])
    assert before(repo.models_dir) == old


def test_inspect_model_explains_shared_projector_and_lora_groups(repo):
    llm, projector, lora = [repo.test_publish(arch) for arch in ('llama','clip','adapter')]
    create_definition(repo, 'a', text_for(llm.relative, projector.relative, lora.relative), {})
    create_definition(repo, 'b', text_for(llm.relative, projector.relative), {})
    report = inspect_repository(repo.repository, model='a')
    assert report['ok'] and set(report['selected_groups']) == {item.group for item in (llm,projector,lora)}
    projector_row = next(row for row in report['groups'] if row['id'] == projector.group)
    assert projector_row['references'] == ['a','b'] and not projector_row['prune_candidate']


@pytest.mark.parametrize('scope', ['group','upload','all-data'])
def test_stats_do_not_offer_protected_bytes_as_prune_candidates(repo, scope):
    data = repo.test_publish(upload=True)
    options = {'groups':[data.group]} if scope == 'group' else (
        {'paths':['.tllama/uploads']} if scope == 'upload' else {'all_data':True})
    with Reservation(repo.repository, 'reader', **options):
        old = before(repo.models_dir)
        report = inspect_repository(repo.repository, group=data.group)
        assert report['ok'] and report['summary']['prune_candidate_weight_bytes'] == 0
        assert not report['groups'][0]['prune_candidate']
        assert before(repo.models_dir) == old


def test_hardlinked_group_files_are_counted_by_inode_for_unique_bytes(repo):
    # Structure-only inspection validates headers, not the content address.
    # Check --hashes can separately detect the deliberately mismatched ID.
    first, second = repo.test_publish('clip'), repo.test_publish('adapter')
    second.paths[0].unlink()
    os.link(first.paths[0], second.paths[0])
    report = inspect_repository(repo.repository)
    assert report['ok']
    assert report['summary']['valid_group_weight_bytes'] - report['summary']['valid_group_unique_inode_bytes'] == first.paths[0].stat().st_size


@pytest.mark.parametrize('fault', ['definition','pin','group','target'])
def test_invalid_state_does_not_claim_complete_accounting_or_reclaimable_bytes(repo, fault):
    data = repo.test_publish()
    if fault == 'definition':
        (repo.local_models_dir / 'bad.toml').write_text('bad toml')
    elif fault == 'pin':
        (repo.models_dir / '.tllama/pins/unknown').write_text('unknown')
    elif fault == 'group':
        data.paths[0].write_bytes(b'broken')
    old = before(repo.models_dir)
    report = inspect_repository(repo.repository, model='absent' if fault == 'target' else None)
    assert not report['ok'] and not report['summary']['byte_counts_complete']
    assert report['summary']['prune_candidate_weight_bytes'] is None
    assert before(repo.models_dir) == old


@pytest.mark.parametrize('args', [['stats'], ['inspect','a'], ['inspect','--group','invalid']])
def test_missing_repository_is_reported_without_initialization(tmp_path, capsys, args):
    path = tmp_path / 'missing'
    assert main(['repo', *args, '--no-config','--models-dir',str(path),'--json']) == 1
    assert not json.loads(capsys.readouterr().out)['ok'] and not path.exists()


@pytest.mark.parametrize('as_json', [False,True])
@pytest.mark.parametrize('command', ['stats','model','group'])
def test_local_cli_inspection_and_optional_storage_configuration(repo, capsys, monkeypatch, as_json, command):
    data = repo.test_publish()
    create_definition(repo, 'a', text_for(data.relative), {})
    monkeypatch.setenv('TLLAMA_MODELS', str(repo.models_dir))
    args = ['stats'] if command == 'stats' else ['inspect','a'] if command == 'model' else ['inspect','--group',data.group]
    old = before(repo.models_dir)
    assert main(['repo', *args, '--no-config'] + (['--json'] if as_json else [])) == 0
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)['ok']
    elif command != 'stats':
        assert data.relative in output and 'definitions: a' in output
    assert before(repo.models_dir) == old


def test_inspection_selection_is_unambiguous(repo):
    for args in (['inspect'], ['inspect','a','--group','0'*64]):
        with pytest.raises(SystemExit) as exc:
            main(['repo', *args, '--no-config','--models-dir',str(repo.models_dir)])
        assert exc.value.code == 2


def test_local_stats_do_not_import_http_or_backend(repo):
    code = '''
import importlib.abc
import sys
class LocalOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'tllama.backend','tllama.main','tllama.client','httpx','fastapi','uvicorn'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, LocalOnly())
from tllama.daemon import main
raise SystemExit(main(['repo','stats','--no-config','--models-dir',sys.argv[1],'--json']))
'''
    result = subprocess.run([sys.executable,'-c',code,str(repo.models_dir)], capture_output=True, text=True,
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1] / 'src')}, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ok']
