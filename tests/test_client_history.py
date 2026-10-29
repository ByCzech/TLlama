import os

from prompt_toolkit.history import InMemoryHistory
import pytest

from tllama.cli_history import input_history
from tllama.client_config import load_client_config
from tllama.client_state import state_directory


def test_history_persists_multiline_and_is_scoped_by_normalized_server_and_model():
    history = input_history('localhost:54800', 'a/model')
    history.append_string('first\nsecond')
    assert list(input_history('http://localhost:54800/', 'a/model').load_history_strings()) == ['first\nsecond']
    assert list(input_history('other:54800', 'a/model').load_history_strings()) == []
    assert list(input_history('localhost:54800', '../other').load_history_strings()) == []
    path = state_directory('localhost:54800', 'a/model', 'history') / 'input.history'
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_disabled_history_does_not_create_state_and_clear_removes_saved_inputs(tmp_path):
    memory = input_history(None, 'model', enabled=False)
    assert isinstance(memory, InMemoryHistory)
    assert not (tmp_path / 'client-state').exists()
    input_history(None, 'model').append_string('private')
    input_history(None, 'model', enabled=False, clear=True)
    path = state_directory(None, 'model', 'history') / 'input.history'
    assert not path.exists()
    assert list(input_history(None, 'model').load_history_strings()) == []


def test_unwritable_state_falls_back_without_affecting_inference(monkeypatch, capsys):
    def denied(path):
        raise PermissionError('denied')
    monkeypatch.setattr('tllama.cli_history.private_directory', denied)
    assert isinstance(input_history(None, 'm'), InMemoryHistory)
    assert 'using memory only' in capsys.readouterr().err


def test_history_rejects_symlink_without_overwriting_target(tmp_path, capsys):
    path = state_directory(None, 'm', 'history') / 'input.history'
    path.parent.mkdir(parents=True)
    target = tmp_path / 'other'
    target.write_text('unchanged')
    path.symlink_to(target)
    input_history(None, 'm').append_string('input')
    assert target.read_text() == 'unchanged'
    assert 'using memory only' in capsys.readouterr().err


def test_history_config_default_override_and_type(tmp_path):
    file = tmp_path / 'client.toml'
    assert load_client_config(environ={}).history_enabled is True
    file.write_text('[history]\nenabled=false\n')
    assert load_client_config(file).history_enabled is False
    assert load_client_config(file, {'history_enabled': True}).history_enabled is True
    file.write_text('[history]\nenabled=1\n')
    with pytest.raises(ValueError, match='true or false'):
        load_client_config(file)
