"""Native BOS/EOS strings and completion input preserve generate rendering."""

import pytest

from tllama.helpers.prompt_render import render_generate_text
from tllama.schemas.ollama import OllamaGenerateRequest
from tllama.worker import WorkerProtocolError

from test_native_worker import factory


def test_render_uses_native_tokens_toml_defaults_and_stop_precedence():
    metadata = {'template': '{{ bos_token }}{{ System }}|{{ Prompt }}|{{ ThinkLevel }}',
                'default_system_prompt': 'TOML', 'stop_defaults': ['TOML STOP']}
    request = OllamaGenerateRequest(model='m', prompt='hi', think='high')
    text, stop = render_generate_text(metadata, request, bos_token='<B>', eos_token='<E>')
    assert text == '<B>TOML|hi|high' and stop == ['TOML STOP', '<E>']
    request.options['stop'] = ['CLIENT', '<E>']
    assert render_generate_text(metadata, request, bos_token='<B>', eos_token='<E>')[1] == ['CLIENT', '<E>']


def test_request_template_empty_system_and_custom_aliases_win():
    request = OllamaGenerateRequest(model='m', prompt='hi', system='', think=False,
        template='{{ System }}|{{ Prompt }}|{{ Think }}|{{ IsThinkSet }}|{{ developer_instructions }}|{{ model_identity }}',
        options={'tllama': {'developer_instructions': 'dev', 'model_identity': 'id'}})
    text, _ = render_generate_text({'template':'TOML','default_system_prompt':'TOML'}, request, bos_token='', eos_token='')
    assert text == '|hi|False|True|dev|id'


async def test_native_tokenizer_preserves_special_token_control(factory):
    worker, _ = factory()
    await worker.start()
    with_special = await worker.tokenize('<bos>hello')
    without = await worker.tokenize('<bos>hello', add_special=False)
    assert with_special[:2] == [1, 1] and without[0] == 1
    assert with_special[1:] == without


async def test_malformed_native_token_ids_are_protocol_errors(factory):
    worker, _ = factory('bad-tokenize')
    await worker.start()
    with pytest.raises(WorkerProtocolError, match='tokens'):
        await worker.tokenize('hello')
