"""The ordinary show command explains helpers while retaining Ollama fallback."""

import json

import pytest

from tllama.cli import main
from tllama.cli_output import show_text
from tllama.client import ClientError
from test_client_cli import cli_server


@pytest.mark.parametrize('role', ['mmproj', 'draft'])
def test_show_displays_helper_type_path_and_attachment_example(cli_server, capsys, role):
    path = 'data/' + 'a' * 64 + '/weights-00001-of-00002.gguf'
    cli_server['responses']['/api/show'] = (200, {
        'details': {'family': 'clip' if role == 'mmproj' else 'qwen3'},
        'capabilities': [], 'template': '', 'modelfile': f'[{role}]\nmodel="{path}"\n',
        'model_definition': {'type': role, 'artifacts': {role: path}}})
    assert main(['--host', cli_server['url'], 'show', 'helper name']) == 0
    out = capsys.readouterr()
    assert f'Type: {role}' in out.out
    assert path in out.out and 'repository-relative' in out.out
    assert f'Usage: attach to a main model using its [{role}].model reference.' in out.out
    assert 'Capabilities: none (helper model)' in out.out
    assert cli_server['calls'] == [('POST', '/api/show', {'model': 'helper name'})]


@pytest.mark.parametrize('selector', ['--json', '--modelfile'])
def test_show_keeps_json_and_modelfile_output_unchanged(cli_server, capsys, selector):
    definition = '# Keep exact text\n[mmproj]\nmodel="data/group/weights.gguf"\n'
    data = {'modelfile': definition, 'capabilities': [],
            'model_definition': {'type': 'mmproj', 'artifacts': {'mmproj': 'data/group/weights.gguf'}}}
    cli_server['responses']['/api/show'] = (200, data)
    assert main(['--host', cli_server['url'], 'show', 'helper', selector]) == 0
    output = capsys.readouterr().out
    assert (json.loads(output) == data) if selector == '--json' else (output == definition)


def test_show_without_extension_retains_ollama_behavior():
    text = show_text('model', {'details': {'family': 'llama'}, 'capabilities': ['completion']})
    assert 'Architecture: llama' in text and 'Capabilities: completion' in text
    assert 'Type:' not in text and 'Usage:' not in text


@pytest.mark.parametrize('definition', [[], {'type': [], 'artifacts': {}}, {'type': 'unknown', 'artifacts': {}},
    {'type': 'mmproj', 'artifacts': []}, {'type': 'mmproj', 'artifacts': {'mmproj': 1}},
    {'type': 'main', 'artifacts': {'other': 'path'}}])
def test_invalid_definition_metadata_is_reported(definition):
    with pytest.raises(ClientError, match='invalid model definition'):
        show_text('model', {'model_definition': definition})
