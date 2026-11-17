"""RAM metadata identity follows the actual file, including paths and ns time."""

from hashlib import sha256
import json
import os

import gguf
import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig
from tllama.model_definitions import create_definition
from tllama.repository_store import prepare_group


def publish(owner, path, context):
    writer = gguf.GGUFWriter(str(path), arch='llama')
    writer.add_uint32('llama.context_length', context)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    stamp = 1700000000000000000
    os.utime(path, ns=(stamp, stamp))
    return owner.repository.publish(prepare_group([path]))[0]


def definition(owner, path):
    return '[llm]\nmodel=' + json.dumps(owner.repository.relative_path(path).as_posix()) + '\n'


@pytest.mark.parametrize('alias', [False, True])
async def test_manual_rebind_between_equal_size_and_time_groups_uses_new_metadata(manager, tmp_path, alias):
    first = publish(manager, tmp_path / 'first.gguf', 4096)
    second = publish(manager, tmp_path / 'second.gguf', 8192)
    if alias:
        root = tmp_path / 'alias'
        root.symlink_to(manager.models_dir, target_is_directory=True)
        manager = ModelManager(BackendConfig(models_dir=str(root)))
    assert first.stat().st_size == second.stat().st_size
    assert first.stat().st_mtime_ns == second.stat().st_mtime_ns
    create_definition(manager, 'model', definition(manager, first), {})
    assert (await manager.get_model_metadata('model'))['context_length'] == 4096
    key = manager._build_model_file_info('model')['sha256']
    (manager.local_models_dir / 'model.toml').write_text(definition(manager, second))
    metadata = await manager.get_model_metadata('model')
    assert metadata['context_length'] == 8192
    assert metadata['model_definition']['artifacts']['llm'] == manager.repository.relative_path(second).as_posix()
    assert manager._build_model_file_info('model')['sha256'] != key


def test_one_nanosecond_change_is_not_lost_through_float_time(manager, tmp_path):
    path = publish(manager, tmp_path / 'model.gguf', 4096)
    before = path.stat()
    key = manager._build_model_file_info_from_path(path)['sha256']
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    after = path.stat()
    if after.st_mtime_ns == before.st_mtime_ns:
        pytest.skip('Filesystem does not retain nanosecond timestamps')
    assert before.st_mtime == after.st_mtime
    assert manager._build_model_file_info_from_path(path)['sha256'] != key


async def test_warm_metadata_and_public_content_identity_remain_stable(manager, tmp_path, monkeypatch):
    path = publish(manager, tmp_path / 'model.gguf', 4096)
    create_definition(manager, 'model', definition(manager, path), {})
    before = manager._listing_identity('model')
    assert before[0] == sha256(path.read_bytes()).hexdigest()
    assert (await manager.get_model_metadata('model'))['context_length'] == 4096
    def forbidden(*args):
        raise AssertionError('Warm metadata must not rescan headers or hash weights')
    monkeypatch.setattr(manager, '_get_model_metadata_sync', forbidden)
    monkeypatch.setattr('tllama.repository_store.hash_file', forbidden)
    assert manager._listing_identity('model') == before
    (manager.local_models_dir / 'model.toml').write_text(
        definition(manager, path) + '[system]\nprompt="fresh override"\n')
    result = await manager.get_model_metadata('model')
    assert result['context_length'] == 4096 and result['default_system_prompt'] == 'fresh override'
    assert manager._get_content_digest_sync(path)['content_sha256'] == before[0]
