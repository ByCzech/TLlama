"""Server logging keeps routine worker HTTP traffic out of the normal log."""

import logging
from types import SimpleNamespace

import pytest

from test_native_worker import factory, records
from test_native_memory_api import payload
from test_worker_telemetry import prepare
from test_native_manager import native_manager, wait_until


HTTP_LOGGERS = ('httpx', 'httpx2', 'httpcore', 'httpcore2')


@pytest.fixture
def server_logging(monkeypatch):
    from tllama import main
    root = logging.getLogger()
    loggers = [root, *(logging.getLogger(name) for name in HTTP_LOGGERS)]
    levels = [logger.level for logger in loggers]
    launches = []
    monkeypatch.setattr(main.uvicorn, 'run', lambda *args, **kwargs: launches.append(kwargs))

    def start(debug=False):
        # Reproduce CLI logging setup before start_server reads the configured log level.
        root.setLevel(logging.INFO)
        monkeypatch.setattr(main, 'current_daemon_config', lambda: SimpleNamespace(app=SimpleNamespace(
            log_level='debug' if debug else 'info', host='127.0.0.1', port=11434, reload=False)))
        main.start_server()
        return launches[-1]

    yield start
    for logger, level in zip(loggers, levels):
        logger.setLevel(level)


@pytest.mark.parametrize('debug', [False, True])
def test_startup_log_level_controls_http_diagnostics_without_a_debug_field(server_logging, debug):
    kwargs = server_logging(debug)
    assert kwargs['log_level'] == ('debug' if debug else 'info')
    assert logging.getLogger('httpx').isEnabledFor(logging.DEBUG) is debug


@pytest.mark.parametrize('debug', [False, True])
async def test_real_polling_is_quiet_unless_debug_and_warnings_remain_visible(
        server_logging, factory, payload, caplog, monkeypatch, debug):
    monkeypatch.setattr('tllama.worker._MEMORY_INTERVAL', 0.02)
    kwargs = server_logging(debug)
    worker, root = factory()
    prepare(root, payload)
    async with worker:
        await wait_until(lambda: len([r for r in records(root, 'requests') if r['path'] == '/tllama/memory']) >= 3)
        assert worker.memory_snapshot()['source'] == 'native_memory_api'
        http_records = [record for record in caplog.records if record.name == 'httpx'
                        and '/tllama/memory' in record.getMessage()]
        assert bool(http_records) is debug
        logging.getLogger('httpx').warning('transport diagnostic remains visible')
        logging.getLogger('tllama.worker').info('worker lifecycle remains visible')
        assert 'transport diagnostic remains visible' in caplog.text
        assert 'worker lifecycle remains visible' in caplog.text
        (root / 'memory-status').write_text('404')
        await wait_until(lambda: worker.memory_snapshot()['native']['reason'] == 'endpoint_not_supported')
        assert 'memory telemetry: endpoint_not_supported' in caplog.text
    assert worker._memory_task.done() and worker._memory_client.is_closed
    if debug:
        assert kwargs['log_level'] == 'debug'
        assert logging.getLogger('tllama.worker').isEnabledFor(logging.DEBUG)
    else:
        assert not logging.getLogger('tllama.worker').isEnabledFor(logging.DEBUG)


async def test_reloads_stop_old_pollers_instead_of_accumulating_them(native_manager, payload, monkeypatch):
    monkeypatch.setattr('tllama.worker._MEMORY_INTERVAL', 0.02)
    manager = native_manager(memory_report=payload)
    name, _ = manager.test.define()
    previous = None
    for threads in (1, 2, 3):
        generation = await manager.preload_native_model(name, runtime_overrides={'n_threads': threads})
        worker, root = manager.test.ledger[-1], manager.test.roots[-1]
        def count(path):
            return sum(r['path'] == '/tllama/memory' for r in records(path, 'requests'))
        await wait_until(lambda: count(root) >= 3)
        if previous is not None:
            old_generation, old, old_root = previous
            await manager.wait_native_closed(old_generation)
            assert old._memory_task.done() and old._memory_client.is_closed
            assert old.process.returncode is not None
            stopped, current = count(old_root), count(root)
            await wait_until(lambda: count(root) >= current + 2)
            assert count(old_root) == stopped
        previous = generation, worker, root
    assert len(manager.test.ledger) == 3
