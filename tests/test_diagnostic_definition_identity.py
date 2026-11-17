"""Owned diagnostic TOMLs are explicit exceptions to repository-relative paths."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

from tllama.backend import ModelManager
from tllama.config import BackendConfig


@pytest.fixture(params=[('decision-api-smoke.py', 'DiagnosticManager'),
                       ('embeddings-api-smoke.py', 'DiagnosticManager'),
                       ('embeddings-multimodal-smoke.py', 'ManagedProbeManager')])
def diagnostic_type(request):
    filename, class_name = request.param
    name = 'definition_identity_' + filename.replace('-', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield getattr(module, class_name)
    finally:
        sys.modules.pop(name, None)


def test_owned_capture_definition_tracks_revision_and_deletion(tmp_path, diagnostic_type):
    root = tmp_path / 'models'
    capture = tmp_path / 'capture'
    capture.mkdir()
    definition = capture / 'probe.toml'
    definition.write_text('[llm]\nmodel="Local/a.gguf"\n')
    manager = diagnostic_type(BackendConfig(models_dir=str(root)), 'a', definition)
    before = manager._toml_fingerprint_for_reference('a')
    assert before['path'] == str(definition.resolve())
    assert before['size'] == definition.stat().st_size
    assert manager._toml_fingerprint_for_reference('unregistered') is None
    assert manager._toml_fingerprint_for_reference('../escape') is None
    stamp = definition.stat()
    os.utime(definition, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
    touched = manager._toml_fingerprint_for_reference('a')
    assert touched != before and touched['size'] == before['size']
    definition.write_text(definition.read_text() + '[system]\nprompt="changed"\n')
    assert manager._toml_fingerprint_for_reference('a')['size'] > touched['size']
    definition.unlink()
    assert manager._toml_fingerprint_for_reference('a') is None
    assert not root.exists()  # Capturing identity is read-only.


def test_production_definition_identity_stays_relative_and_rejects_external_target(tmp_path):
    root = tmp_path / 'models'
    local = root / 'Local'
    local.mkdir(parents=True)
    definition = local / 'a.toml'
    definition.write_text('[llm]\nmodel="Local/a.gguf"\n')
    manager = ModelManager(BackendConfig(models_dir=str(root)))
    assert manager._toml_fingerprint_for_reference('a')['path'] == 'Local/a.toml'
    external = tmp_path / 'external.toml'
    definition.rename(external)
    definition.symlink_to(external)
    with pytest.raises(ValueError):
        manager._toml_fingerprint_for_reference('a')
