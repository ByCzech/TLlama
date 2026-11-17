"""External MTP role/admission tests; small GGUF fixtures are not HW evidence."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.helpers.draft_metadata import group_draft_kind, tensor_summary
from tllama.helpers.gguf_metadata import read_gguf_metadata, build_model_metadata_payload
from tllama.helpers.metadata_cache import load_metadata_cache, get_metadata_cache_path
from tllama.helpers.model_toml import parse_model_toml
from tllama.model_identity import parameter_changes
from tllama.native_config import resolve_worker_config
from tllama.repository_names import definition_path
from tllama.repository_store import prepare_group
from tllama.repository_prune import prune_repository
from tllama.worker import WorkerError
from test_repository_runtime import hf
from test_native_manager import native_manager
from test_repository_worker_pins import protected_groups
from test_model_create import create_client
from tllama.client_import import create_from_file


def evidence(arch='qwen4exp', *, names=('blk.48.nextn.eh_proj.weight',), blocks=49, nextn=1, shared=False):
    return {'arch': arch, '_draft': tensor_summary(arch, blocks, nextn, shared, names)}


def fixture_file(path, arch='qwen4exp', *, names=('blk.48.nextn.eh_proj.weight',),
                 blocks=49, nextn=1, shared=False, shard=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(path), arch=arch)
    w.add_uint32(arch + '.block_count', blocks)
    w.add_uint32(arch + '.nextn_predict_layers', nextn)
    if shared:
        w.add_bool(arch + '.nextn_shared_target_tensors', True)
    if shard is not None:
        w.add_uint16('split.no', shard)
        w.add_uint16('split.count', 2)
    for name in names:
        w.add_tensor(name, np.zeros((1,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.mark.parametrize('arch', ['qwen3next', 'qwen35', 'qwen35moe', 'qwen4exp', 'nemotron_h_moe', 'deepseek2', 'deepseek4'])
@pytest.mark.parametrize('shared', [False, True])
def test_nextn_only_detection_uses_existing_reader_and_keeps_summary_private(tmp_path, arch, shared):
    path = fixture_file(tmp_path / 'anonymous.gguf', arch, shared=shared)
    raw = read_gguf_metadata(str(path))
    payload = build_model_metadata_payload(raw)
    assert group_draft_kind([payload]) == 'mtp'
    assert bool(payload['_draft']['shared_target']) == shared
    assert not any('nextn' in k or k == '_draft' for k in payload['metadata_raw'])


@pytest.mark.parametrize('changes', [dict(blocks=None), dict(nextn=0), dict(nextn=49), dict(nextn=True),
    dict(names=()), dict(names=('blk.0.attn_norm.weight', 'blk.48.nextn.eh_proj.weight')),
    dict(names=('blk.49.nextn.eh_proj.weight',)), dict(arch='future')])
def test_missing_contradictory_integrated_or_unknown_evidence_does_not_guess_helper(changes):
    assert group_draft_kind([evidence(**changes)]) is None


def test_split_classification_unions_all_shards_not_only_head_or_tail(tmp_path):
    paths = [fixture_file(tmp_path / f'm-0000{i+1}-of-00002.gguf', shard=i,
                          names=() if i == 0 else ('blk.48.nextn.eh_proj.weight',)) for i in range(2)]
    group = prepare_group(paths)
    assert group_draft_kind(group.metadata) == 'mtp'
    trunk = evidence(names=('blk.0.attn_norm.weight',))
    assert group_draft_kind([trunk, group.metadata[1]]) is None
    assert group_draft_kind([evidence(arch='future'), group.metadata[1]]) is None
    assert group_draft_kind([evidence(blocks=50), group.metadata[1]]) is None


def test_gemma_assistant_and_dspark_have_distinct_role_evidence():
    assert group_draft_kind([evidence('gemma4-assistant', names=('blk.0.attn_norm.weight',), blocks=4, nextn=4)]) == 'mtp'
    assert group_draft_kind([evidence('dflash', names=('markov_w1.weight',))]) == 'dspark'


def resolve(arch, strategy='draft-mtp', overrides=None, **tuning):
    body = '[llm]\nmodel="Local/main.gguf"\n[draft]\nmodel="Local/draft.gguf"\n[runtime]\n'
    body += f'spec_type="{strategy}"\n' + ''.join(f'{k}={json.dumps(v)}\n' for k, v in tuning.items())
    return resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}), Path('/models/main.gguf'),
        parse_model_toml(body), draft=Path('/models/draft.gguf'), draft_metadata=arch,
        runtime_overrides=overrides)


@pytest.mark.parametrize('arch', [{}, {'arch': 'unknown'}, evidence(), {'arch': 'gemma4-assistant'}])
@pytest.mark.parametrize('count', [None, 0, 2])
def test_external_mtp_activation_off_and_explicit_none(arch, count):
    overrides = {} if count is None else {'draft_num_predict': count}
    r = resolve(arch, overrides=overrides, draft_type_k='q8_0', draft_n_gpu_layers=0)
    command = r.launch.command(Path('/private/key'))
    assert ('--spec-draft-model' in command) == (count != 0)
    assert ('--spec-draft-type-k' in command) == (count != 0)
    assert r.launch.model == Path('/models/main.gguf')
    off = resolve(arch, overrides={'spec_type': 'none', 'draft_num_predict': 4})
    assert off.launch.draft is None
    assert '--spec-draft-model' not in off.launch.command(Path('/private/key'))


def test_dflash_binding_keeps_integrated_mtp_and_tuning_dormant():
    r = resolve({'arch': 'dflash'}, draft_device='none', draft_n_gpu_layers='all')
    assert r.launch.draft is None
    assert '--spec-draft-ngl' not in r.launch.command(Path('/private/key'))


def test_known_unsupported_shared_and_wrong_strategy_fail_without_fallback():
    with pytest.raises(ConfigError, match='Shared-target'):
        resolve(evidence(shared=True))
    assert resolve(evidence(shared=True), overrides={'draft_num_predict': 0}).launch.draft is None
    with pytest.raises(ConfigError, match='requires dflash weights'):
        resolve(evidence(), strategy='draft-dflash')
    with pytest.raises(ConfigError, match='DSpark'):
        resolve(evidence('dflash', names=('markov_w1.weight',)), strategy='draft-dflash')
    with pytest.raises(ConfigError, match='conflicts'):
        resolve(evidence(), draft_device='none', draft_n_gpu_layers='all')


def test_external_tuning_identity_is_active_only_for_the_selected_helper():
    def spec(value):
        return parse_model_toml(f'[llm]\nmodel="m"\n[draft]\nmodel="d"\n[runtime]\nspec_type="draft-mtp"\ndraft_n_ubatch={value}\n')
    assert parameter_changes(spec(64), {}, draft_metadata=evidence()) != parameter_changes(spec(128), {}, draft_metadata=evidence())
    assert parameter_changes(spec(64), {}, draft_metadata={'arch': 'dflash'}) == parameter_changes(spec(128), {}, draft_metadata={'arch': 'dflash'})


@pytest.mark.parametrize('arch', ['qwen4exp', 'gemma4-assistant', 'nemotron_h_moe', 'deepseek4'])
def test_pull_helper_reuse_user_role_and_old_cache_enrichment(manager, hf, tmp_path, arch):
    hf.files['ns/repo', 'anonymous.gguf'] = fixture_file(tmp_path / f'{arch}.gguf', arch).read_bytes()
    path = Path(manager._pull_hf_file_sync('ns/repo', 'anonymous.gguf'))
    target = manager.models_dir / definition_path('ns/repo/anonymous')
    assert parse_model_toml(target.read_text()).model_type == 'draft'
    assert not manager._list_local_models_sync()
    # Old metadata v9 remains usable. Only the selected import header is enriched.
    cache = get_metadata_cache_path(manager.metadata_cache_dir, path)
    document = json.loads(cache.read_text())
    document['metadata'].pop('_draft')
    stamp = cache.stat().st_mtime_ns
    cache.write_text(json.dumps(document))
    import os
    os.utime(cache, ns=(stamp, stamp))
    original = '# User-selected main role\n[llm]\nmodel=' + json.dumps(path.relative_to(manager.models_dir).as_posix()) + '\n'
    target.write_text(original)
    assert manager._pull_hf_file_sync('ns/repo', 'anonymous.gguf') == str(path)
    assert target.read_text() == original
    assert not (target.parent / 'anonymous_01.toml').exists()
    assert load_metadata_cache(manager.metadata_cache_dir, path)['_draft']
    assert len(hf.calls) == 1
    assert prune_repository(manager.repository, apply=True)['removed_groups'] == []


async def test_create_standalone_helper_and_direct_attachment_share_weights(create_client, manager, tmp_path):
    helper = fixture_file(tmp_path / 'unknown-name.gguf')
    definition = tmp_path / 'draft.toml'
    definition.write_text('[llm]\nfrom=' + json.dumps(str(helper)) + '\n')
    await create_from_file(create_client, 'helper', definition, as_json=True)
    spec = parse_model_toml(manager.read_model_definition('helper'))
    assert spec.model_type == 'draft'
    assert spec.draft_model == f'data/{sha256(helper.read_bytes()).hexdigest()}/weights.gguf'
    from test_repository_runtime import publish
    from tllama.model_definitions import copy_definition, create_definition
    main = publish(manager, tmp_path, arch='llama')
    create_definition(manager, 'main', '[llm]\nmodel=' + json.dumps(main) + '\n', {})
    copy_definition(manager, 'main', 'composite', draft='helper')
    composite = parse_model_toml(manager.read_model_definition('composite'))
    assert composite.draft_model == spec.draft_model and not composite.runtime
    manager.delete_model_definition('helper')
    assert (manager.models_dir / composite.draft_model).is_file()
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('fatal', [False, True])
async def test_manager_selects_external_before_embedded_and_preserves_failure(native_manager, tmp_path, monkeypatch, fatal):
    manager = native_manager(mode='fatal' if fatal else 'normal')
    factory = manager._native._factory
    monkeypatch.setattr(manager._native, '_factory', lambda spec: factory(replace(spec, transport='tcp')))
    source = fixture_file(tmp_path / 'external.gguf')
    draft, = manager.repository.publish(prepare_group([source]))
    relative = draft.relative_to(manager.models_dir).as_posix()
    name, toml = manager.test.define(extra=f'[draft]\nmodel="{relative}"\n[runtime]\nspec_type="draft-mtp"\ndraft_n_ubatch=64\n')
    if fatal:
        with pytest.raises(WorkerError, match='kernel image is invalid'):
            await manager.preload_native_model(name)
        assert len(manager.test.ledger) == 1 and not protected_groups(manager)
        return
    generation = await manager.preload_native_model(name)
    [worker] = manager.test.ledger
    assert worker.command[worker.command.index('--spec-draft-model') + 1] == str(draft)
    assert worker.command[worker.command.index('--spec-draft-n-ubatch') + 1] == '64'
    assert draft.parent.name in protected_groups(manager)
    toml.write_text(toml.read_text().replace('draft_n_ubatch=64', 'draft_n_ubatch=128'))
    changed = await manager.preload_native_model(name)
    assert changed != generation
    assert manager.test.ledger[-1].command[manager.test.ledger[-1].command.index('--spec-draft-n-ubatch') + 1] == '128'
    off = await manager.preload_native_model(name, runtime_overrides={'draft_num_predict': 0})
    assert off != changed and '--spec-draft-model' not in manager.test.ledger[-1].command
    assert draft.parent.name not in protected_groups(manager)
    await manager.unload_native_model(name)
    await manager.wait_native_closed(off)


@pytest.mark.parametrize('integrated', [False, True])
def test_pull_split_head_with_later_trunk_never_becomes_helper(manager, hf, tmp_path, integrated):
    for i in range(2):
        name = f'anonymous-0000{i+1}-of-00002.gguf'
        names = ('blk.48.nextn.eh_proj.weight',) if i else (('blk.0.attn_norm.weight',) if integrated else ())
        hf.files['ns/repo', name] = fixture_file(tmp_path / name, shard=i, names=names).read_bytes()
    manager._pull_hf_file_sync('ns/repo', 'anonymous-00001-of-00002.gguf')
    spec = parse_model_toml(manager.read_model_definition('ns/repo/anonymous-00001-of-00002'))
    assert spec.model_type == ('main' if integrated else 'draft')


def test_manual_unknown_helper_role_survives_repeat_pull(manager, hf):
    hf.add('mtp-name-is-not-evidence.gguf', arch='llama')
    path = Path(manager._pull_hf_file_sync('ns/repo', 'mtp-name-is-not-evidence.gguf'))
    target = manager.models_dir / definition_path('ns/repo/mtp-name-is-not-evidence')
    assert parse_model_toml(target.read_text()).model_type == 'main'
    text = '# User declaration\n[draft]\nmodel=' + json.dumps(path.relative_to(manager.models_dir).as_posix()) + '\n'
    target.write_text(text)
    manager._pull_hf_file_sync('ns/repo', 'mtp-name-is-not-evidence.gguf')
    assert target.read_text() == text and len(hf.calls) == 1
    assert len(list(target.parent.glob('*.toml'))) == 1
