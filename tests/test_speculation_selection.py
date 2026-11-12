"""Resolve public draft intent only after the configured strategy is known."""

from pathlib import Path

import pytest

from tllama.config import BackendConfig, ConfigError
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import resolve_worker_config
from tllama.ollama_options import ollama_runtime_overrides, resolve_ollama_speculation
from test_native_manager import native_manager
from test_native_public_api import public, request_body


@pytest.mark.parametrize('value', [0, 1, 7])
@pytest.mark.parametrize('strategy', [None, 'none', 'draft-mtp', 'draft-dflash'])
def test_deferred_request_resolution_preserves_selected_method_without_mutation(value, strategy):
    public_options = {'num_batch': 128, 'draft_num_predict': value}
    overrides = ollama_runtime_overrides(public_options)
    assert overrides == {'n_batch': 128, 'n_ubatch': 128, 'draft_num_predict': value}
    result = resolve_ollama_speculation(overrides, strategy)
    expected = 'none' if value == 0 else strategy if strategy not in (None, 'none') else 'draft-mtp'
    assert result == {'n_batch': 128, 'n_ubatch': 128, 'spec_type': expected,
                      **({'draft_n_max': value} if value else {})}
    assert overrides['draft_num_predict'] == value and 'spec_type' not in overrides
    assert public_options == {'num_batch': 128, 'draft_num_predict': value}


@pytest.mark.parametrize('value', [-1, True, None, '3', 1.5, 2 ** 31])
def test_invalid_deferred_value_is_rejected_at_translation_and_resolution(value):
    with pytest.raises(ValueError, match='draft_num_predict'):
        ollama_runtime_overrides({'draft_num_predict': value})
    with pytest.raises(ConfigError, match='draft_num_predict'):
        resolve_worker_config(BackendConfig(), Path('/models/main.gguf'),
                              runtime_overrides={'draft_num_predict': value})


@pytest.mark.parametrize('global_type, toml_type', [(None, None), ('draft-mtp', None),
    ('draft-mtp', 'none'), ('none', 'draft-mtp')])
@pytest.mark.parametrize('value', [0, 1, 7])
def test_effective_worker_selection_uses_configuration_layers_before_request(global_type, toml_type, value):
    global_values = {'device': 'none', 'draft_n_max': '5'}
    if global_type is not None:
        global_values['spec_type'] = global_type
    config = BackendConfig(runtime_overrides=global_values)
    text = '[llm]\nmodel="Local/model.gguf"\n'
    if toml_type is not None:
        text += f'[runtime]\nspec_type="{toml_type}"\n'
    spec = parse_model_toml(text)
    resolved = resolve_worker_config(config, Path('/models/main.gguf'), spec,
                                    runtime_overrides=ollama_runtime_overrides({'draft_num_predict': value}))
    args = resolved.launch.command(Path('/private/key'))
    assert args[args.index('--spec-type') + 1] == ('draft-mtp' if value else 'none')
    assert resolved.sources['spec_type'] == 'request'
    assert 'draft_num_predict' not in resolved.runtime
    assert 'draft_num_predict' not in resolved.launch.runtime_options
    if value:
        assert args[args.index('--spec-draft-n-max') + 1] == str(value)
    else:
        assert '--spec-draft-n-max' not in args and resolved.runtime['draft_n_max'] == 5
    assert spec.document.as_string() == text and config.runtime_overrides == global_values


@pytest.mark.parametrize('value', [0, 3])
def test_direct_native_draft_limit_does_not_implicitly_activate_a_strategy(value):
    resolved = resolve_worker_config(BackendConfig(runtime_overrides={'device': 'none'}),
                                    Path('/models/main.gguf'), runtime_overrides={'draft_n_max': value})
    args = resolved.launch.command(Path('/private/key'))
    assert '--spec-type' not in args and '--spec-draft-n-max' not in args


def test_external_dflash_requires_a_resolved_draft():
    spec = parse_model_toml('[llm]\nmodel="Local/model.gguf"\n[runtime]\nspec_type="draft-dflash"\n')
    with pytest.raises(ConfigError, match='requires a resolved'):
        resolve_worker_config(BackendConfig(), Path('/models/main.gguf'), spec)


@pytest.mark.parametrize('path', ['/api/chat', '/api/generate'])
@pytest.mark.parametrize('stream', [False, True])
async def test_request_off_on_omitted_uses_expected_generations_and_keeps_toml(public, path, stream):
    manager, client, name, toml = public
    toml.write_text(toml.read_text() + '\n[runtime]\nspec_type="draft-mtp"\ndraft_n_max=5\n')
    original = toml.read_bytes()
    for value in (0, 0, 2, 2, None):
        options = {} if value is None else {'draft_num_predict': value}
        response = await client.post(path, json={**request_body(name, path, stream), 'options': options})
        assert response.status_code == 200, response.text
        args = manager.test.ledger[-1].command
        assert args[args.index('--spec-type') + 1] == ('none' if value == 0 else 'draft-mtp')
        if value == 0:
            assert '--spec-draft-n-max' not in args
        else:
            assert args[args.index('--spec-draft-n-max') + 1] == str(5 if value is None else value)
    assert len(manager.test.ledger) == 3
    assert toml.read_bytes() == original
