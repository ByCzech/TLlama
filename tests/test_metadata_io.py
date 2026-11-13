"""Metadata file work stays off-loop and remains owned until it finishes."""

import asyncio
import json
from pathlib import Path
import threading

import pytest

from tllama.helpers.metadata_cache import get_metadata_cache_path
from tllama.helpers.model_toml import TomlModelError
from tllama.model_definitions import create_definition
from tllama.repository_store import prepare_group
from test_model_create import gguf_file


@pytest.fixture
def vision(manager, tmp_path):
    llm, projector = [manager.repository.publish(prepare_group([
        gguf_file(tmp_path / (arch + '.gguf'), arch=arch)]))[0] for arch in ('llama', 'clip')]
    text = '[llm]\nmodel=' + json.dumps(llm.relative_to(manager.repository.root).as_posix())
    text += '\n[mmproj]\nmodel=' + json.dumps(projector.relative_to(manager.repository.root).as_posix()) + '\n'
    create_definition(manager, 'vision', text, {})
    return manager, llm, projector


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_cold_projector_scan_and_toml_reads_run_outside_event_loop(vision, monkeypatch):
    manager, llm, projector = vision
    get_metadata_cache_path(manager.metadata_cache_dir, projector).unlink()
    loop_thread = threading.get_ident()
    reads, definitions = [], []
    scan, resolve = manager._get_model_metadata_sync, manager._resolve_virtual_model_spec
    def read(path):
        reads.append((Path(path), threading.get_ident()))
        return scan(path)
    def definition(name):
        definitions.append(threading.get_ident())
        return resolve(name)
    monkeypatch.setattr(manager, '_get_model_metadata_sync', read)
    monkeypatch.setattr(manager, '_resolve_virtual_model_spec', definition)
    result = await manager.get_model_metadata('vision')
    assert result['has_projector'] and result['vision_configured']
    assert reads and all(thread != loop_thread for path, thread in reads)
    assert reads[0][0] == projector
    assert definitions and all(thread != loop_thread for thread in definitions)
    reads.clear()
    assert (await manager.get_model_metadata('vision'))['projector_info'] == result['projector_info']
    assert reads == []


@pytest.mark.parametrize('phase', ['file-info', 'overrides', 'gguf'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_slow_metadata_does_not_block_loop_and_cancel_joins_owned_work(vision, monkeypatch, phase, cancel):
    manager, llm, projector = vision
    if phase == 'gguf':
        get_metadata_cache_path(manager.metadata_cache_dir, llm).unlink()
    name = {'file-info': '_build_model_file_info', 'overrides': '_model_metadata_overrides',
            'gguf': '_get_model_metadata_sync'}[phase]
    work = getattr(manager, name)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    def blocked(*args):
        assert threading.get_ident() != loop_thread
        entered.set()
        if not release.wait(5):
            raise TimeoutError('test did not release metadata work')
        try:
            return work(*args)
        finally:
            finished.set()
    monkeypatch.setattr(manager, name, blocked)
    task = asyncio.create_task(manager.get_model_metadata('vision'))
    try:
        await until(entered.is_set)
        # An independent coroutine actually makes progress while file work waits.
        assert await asyncio.wait_for(asyncio.sleep(0, result='responsive'), 1) == 'responsive'
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not finished.is_set()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task)['has_projector']
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('phase', ['file-info', 'overrides'])
async def test_structural_metadata_error_is_preserved(vision, monkeypatch, phase):
    manager, *_ = vision
    def bad(*args):
        raise TomlModelError('broken model definition')
    name = '_build_model_file_info' if phase == 'file-info' else '_model_metadata_overrides'
    monkeypatch.setattr(manager, name, bad)
    with pytest.raises(TomlModelError, match='broken model definition'):
        await manager.get_model_metadata('vision')


async def test_optional_main_header_scan_failure_still_returns_none(manager, tmp_path, monkeypatch):
    path, = manager.repository.publish(prepare_group([gguf_file(tmp_path / 'llm.gguf')]))
    create_definition(manager, 'model', '[llm]\nmodel=' + json.dumps(path.relative_to(manager.repository.root).as_posix()), {})
    get_metadata_cache_path(manager.metadata_cache_dir, path).unlink()
    def bad(*args):
        raise OSError('optional reader unavailable')
    monkeypatch.setattr(manager, '_get_model_metadata_sync', bad)
    assert await manager.get_model_metadata('model') is None
