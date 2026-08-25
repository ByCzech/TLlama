import os
import tempfile

# Set before anything imports tllama. The package builds a ModelManager at
# module scope from the environment, so a stray TLLAMA_MODELS on the machine
# running the tests would otherwise point the suite at a real model store.
os.environ["TLLAMA_MODELS"] = tempfile.mkdtemp(prefix="tllama-tests-")

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig


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
def client():
    """A test client over the whole application."""
    from fastapi.testclient import TestClient

    from tllama.main import app

    with TestClient(app) as test_client:
        yield test_client
