import locale
import os
import tempfile

# Set before anything imports tllama. The package builds a ModelManager at
# module scope from the environment, so a stray TLLAMA_MODELS on the machine
# running the tests would otherwise point the suite at a real model store.
os.environ["TLLAMA_MODELS"] = tempfile.mkdtemp(prefix="tllama-tests-")

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig


@pytest.fixture(autouse=True)
def isolated_formatting_locale(monkeypatch):
    """Use stable human-output expectations and restore CLI locale changes.

    main() reads the environment even when called in-process. Locale-specific
    tests can override these defaults explicitly, without requiring OS locales.
    """
    categories = {locale.LC_NUMERIC: 'LC_NUMERIC', locale.LC_TIME: 'LC_TIME'}
    previous = {category: locale.setlocale(category) for category in categories}
    # LC_ALL would otherwise override both per-category environment settings.
    monkeypatch.delenv('LC_ALL', raising=False)
    for category, variable in categories.items():
        monkeypatch.setenv(variable, 'C')
        locale.setlocale(category, 'C')
    try:
        yield
    finally:
        for category, value in previous.items():
            locale.setlocale(category, value)


@pytest.fixture
def clean_tllama_config_environment(monkeypatch):
    """Give opt-in config tests only the settings they explicitly supply."""
    for name in tuple(os.environ):
        # Keep the test-owned store in place for application imports.
        if name.startswith("TLLAMA_") and name != "TLLAMA_MODELS":
            monkeypatch.delenv(name)


@pytest.fixture
def models_root(tmp_path):
    """An empty model store that never touches the real one."""
    return tmp_path / "models"


@pytest.fixture
def manager(models_root):
    """A ModelManager with its storage laid out, and nothing loaded."""
    manager = ModelManager(BackendConfig(models_dir=str(models_root)))
    manager.ensure_storage()
    return manager


@pytest.fixture
def make_manager(models_root):
    """Build a manager with specific configuration overrides."""
    def factory(**overrides):
        manager = ModelManager(
            BackendConfig(models_dir=str(models_root), **overrides)
        )
        manager.ensure_storage()
        return manager

    return factory


@pytest.fixture
def native_reply(manager, monkeypatch):
    """Install explicit native lease/worker fakes for a public API test."""
    from native_route_fakes import install_reply

    def factory(**values):
        return install_reply(monkeypatch, manager, **values)
    return factory


@pytest.fixture
def gguf_file(models_root):
    """Place a file in the model store and return its path.

    The bytes are arbitrary: nothing here parses a GGUF header.
    """
    def factory(relative_path, payload=b"gguf-bytes"):
        path = models_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    return factory


@pytest.fixture
def toml_file(models_root):
    """Place a virtual-model .toml file in the model store and return its path."""
    def factory(relative_path, text):
        path = models_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return factory


@pytest.fixture
def client():
    """A test client over the whole application."""
    from fastapi.testclient import TestClient

    from tllama.main import app

    with TestClient(app) as test_client:
        yield test_client
