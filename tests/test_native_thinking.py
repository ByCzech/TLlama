"""Explicit Instella thinking survives native, public API and REPL boundaries."""

import asyncio
import json

import httpx
import pytest

from tllama.cli_run import RunSession
from tllama.client import TLlamaClient
from tllama.helpers.reasoning_split import detect_reasoning_format, completion_reasoning_format
from tllama.main import app
from tllama.native_api import _read_events, read_result
from tllama.native_requests import NativeRequest
from tllama.worker import WorkerProtocolError

from test_native_api import chat_result, collected, sequence
from test_client_status import Terminal
from test_model_information import information


# The original AMD template renders only assistant message.content and ends
# its generation prompt in <｜Assistant｜>, with no prefilled <think>. There is
# no reasoning_content reference for native differential analysis to discover.
# Source: amd/Instella-MoE-16B-A3B-Think tokenizer_config.json,
# revision 74d28c1a8425583d7805b58e2caa9c13d6b62929.
METADATA = {"arch": "instella-moe", "template":
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}"}
REQUEST = NativeRequest("chat", {}, reasoning_format="think_prefix")
THOUGHT = "                \nHmm, uživatel napsal Předveď se!\n"
ANSWER = '\nOdpověď. Literal <think>example</think> stays in the answer.'
TEXT = '<think>' + THOUGHT + '</think>' + ANSWER
ALIASES = ['renamed', 'qwen-alias', 'deepseek-alias', 'gemma-alias', 'granite-alias', 'gpt-oss-alias']
TAG_TEMPLATE = (
    "{% for message in messages %}{{ message.content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<assistant>' }}"
    "{% if enable_thinking is not defined or enable_thinking %}{{ '<think>\\n' }}"
    "{% else %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
    "{# Assistant history uses <think>...</think>. #}"
)


def delta(content="", reasoning="", finish=None):
    return {"choices": [{"delta": {"content": content, "reasoning_content": reasoning},
                         "finish_reason": finish}]}


def test_format_uses_architecture_and_template_without_relying_on_alias():
    assert detect_reasoning_format(METADATA) == "think_prefix"
    assert detect_reasoning_format({"template": "<think>{{ r }}</think>{{ c }}"}) == "think_prefix"
    assert detect_reasoning_format({"template": "{{ content }}"}) == "none"


@pytest.mark.parametrize('arch', ['qwen2', 'qwen3', 'qwen35moe', 'gemma3', 'gemma4', 'granite', 'gpt-oss'])
def test_architecture_without_template_evidence_does_not_imply_thinking(arch):
    metadata = {'arch': arch, 'template': '{{ Prompt }}'}
    assert detect_reasoning_format(metadata) == 'none'
    assert completion_reasoning_format(metadata, '<think>') == 'none'


@pytest.mark.parametrize('template,expected', [
    ('<think>{{ r }}</think>{{ c }}', 'think_prefix'),
    ('<think>{{ r }}</think><response>{{ c }}</response>', 'granite'),
    ('<|channel>thought\n{{ r }}<channel|>{{ c }}', 'gemma_channel'),
    ('<|channel|>analysis<|message|>{{ r }}<|channel|>final<|message|>{{ c }}', 'gpt_oss_channel'),
    ('<THINK>{{ r }}</THINK>', 'none'),
])
def test_template_format_is_independent_of_architecture(template, expected):
    assert detect_reasoning_format({'arch': 'unknown', 'template': template}) == expected


@pytest.mark.parametrize('prefix,expected', [
    (None, 'think_prefix'), ('', 'think_prefix'), ('<assistant><think>\n', 'think_prefill'),
    ('<assistant><think>\n\n</think>\n', 'think_prefix'),
    ('<assistant>', 'think_prefix'),
])
def test_implicit_mode_requires_actual_unclosed_generation_prefix(prefix, expected):
    assert completion_reasoning_format({'template': TAG_TEMPLATE}, prefix) == expected


@pytest.mark.parametrize('alias', ALIASES)
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('think,generated,reasoning,answer', [
    (None, 'reason</think>answer', 'reason', 'answer'),
    (False, 'answer', '', 'answer'),
])
def test_generate_prefill_follows_rendered_template_for_every_alias(
        client, native_reply, alias, stream, think, generated, reasoning, answer):
    state = native_reply(metadata={'arch': 'qwen3', 'template': TAG_TEMPLATE}, content=generated)
    response = client.post('/api/generate', json={'model': alias, 'prompt': 'hi', 'think': think, 'stream': stream})
    assert response.status_code == 200, response.text
    bodies = [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]
    assert ''.join(p.get('thinking', '') for p in bodies) == reasoning
    assert ''.join(p['response'] for p in bodies) == answer
    assert bodies[-1]['done'] and state.refs == 0


@pytest.mark.parametrize('alias', ALIASES)
def test_chat_plain_model_with_misleading_alias_does_not_parse_literal_tags(client, native_reply, alias):
    native_reply(metadata={'arch': 'qwen3', 'template': '{{ Prompt }}'}, content='<think>literal</think>answer')
    response = client.post('/api/chat', json={'model': alias, 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False})
    assert response.json()['message'] == {'role': 'assistant', 'content': '<think>literal</think>answer'}


@pytest.mark.parametrize('extra,generated,reasoning,answer', [
    ({'template': '{{ Prompt }}'}, '<think>literal</think>answer', '', '<think>literal</think>answer'),
    # A prompt suffix written by the user is not a template generation prefill.
    ({'template': '{{ Prompt }}{# <think></think> #}', 'prompt': 'hi<think>'}, 'answer', '', 'answer'),
    ({'raw': True, 'prompt': 'hi<think>\n'}, 'reason</think>answer', 'reason', 'answer'),
    ({'raw': True}, 'answer', '', 'answer'),
])
def test_generate_uses_request_template_and_explicit_raw_prefix(client, native_reply, extra, generated, reasoning, answer):
    native_reply(metadata={'arch': 'qwen3', 'template': TAG_TEMPLATE}, content=generated)
    response = client.post('/api/generate', json={'model': 'qwen-alias', 'prompt': 'hi', 'stream': False, **extra})
    assert response.status_code == 200, response.text
    assert response.json()['response'] == answer
    assert response.json().get('thinking', '') == reasoning


def test_template_ignoring_think_false_still_opens_actual_reasoning(client, native_reply):
    template = '{{ Prompt }}{% if add_generation_prompt %}<think>{% endif %}{# </think> #}'
    native_reply(metadata={'arch': 'qwen3', 'template': template}, content='reason</think>answer')
    response = client.post('/api/generate', json={'model': 'renamed', 'prompt': 'hi', 'think': False, 'stream': False})
    assert response.json()['thinking'] == 'reason' and response.json()['response'] == 'answer'


@pytest.mark.parametrize('template,expected', [('{{ Prompt }}', False), (TAG_TEMPLATE, True)])
async def test_show_thinking_capability_is_alias_independent(information, template, expected):
    manager, client, _, definition, write = information
    write('info.gguf', arch='qwen3', **{'tokenizer.chat_template': template})
    for alias in ALIASES:
        (manager.local_models_dir / (alias + '.toml')).write_text(definition.read_text())
        response = await client.post('/api/show', json={'model': alias})
        assert response.status_code == 200, response.text
        assert ('thinking' in response.json()['capabilities']) is expected


@pytest.mark.parametrize("split", range(len(TEXT) + 1))
async def test_explicit_chat_block_at_every_chunk_boundary_matches_full_result(split):
    result = read_result(chat_result(TEXT, reasoning=""), REQUEST)
    events = await collected([delta(TEXT[:split]), delta(TEXT[split:], finish="stop")], REQUEST)
    assert result.reasoning == THOUGHT == ''.join(e.reasoning for e in events)
    assert result.content == ANSWER == ''.join(e.content for e in events)
    assert events[-1].terminal and events[-1].finish_reason == "stop"


@pytest.mark.parametrize("text,thinking,content", [
    (' \n<think>r</think>a', ' \nr', 'a'),
    ('<think>unfinished', 'unfinished', ''),
    ('<think>unfinished</thi', 'unfinished</thi', ''),
    ('<thi', '', '<thi'), (' \n', '', ' \n'), ('', '', ''),
    ('Plain <think>literal</think>', '', 'Plain <think>literal</think>'),
    ('<think></think>answer', '', 'answer'),
])
async def test_character_deltas_preserve_whitespace_and_truncated_text(text, thinking, content):
    events = await collected([*[delta(char) for char in text], delta(finish="length")], REQUEST)
    result = read_result(chat_result(text, reasoning="", finish="length"), REQUEST)
    assert result.reasoning == thinking == ''.join(e.reasoning for e in events)
    assert result.content == content == ''.join(e.content for e in events)
    assert events[-1].finish_reason == "length"


@pytest.mark.parametrize("structured", [False, True])
async def test_structured_reasoning_and_unknown_formats_leave_literal_tags_alone(structured):
    request = REQUEST if structured else NativeRequest("chat", {})
    reasoning = "native thought" if structured else ""
    events = await collected([delta(reasoning=reasoning), *[delta(c) for c in TEXT], delta(finish="stop")], request)
    result = read_result(chat_result(TEXT, reasoning=reasoning), request)
    assert result.content == TEXT == ''.join(e.content for e in events)
    assert result.reasoning == reasoning == ''.join(e.reasoning for e in events)


async def test_same_delta_native_reasoning_wins_and_state_does_not_leak_between_requests():
    first, second = await asyncio.gather(
        collected([delta(TEXT, "native", finish="stop")], REQUEST),
        collected([delta(TEXT, finish="stop")], REQUEST))
    assert ''.join(e.content for e in first) == TEXT
    assert ''.join(e.reasoning for e in first) == 'native'
    assert ''.join(e.reasoning for e in second) == THOUGHT


async def test_qwen_chat_plain_answer_never_becomes_implicit_reasoning():
    request = NativeRequest("chat", {}, reasoning_format="think_prefill")
    events = await collected([delta('ordinary answer', finish="stop")], request)
    assert ''.join(e.content for e in events) == 'ordinary answer'
    assert not any(e.reasoning for e in events)


@pytest.mark.parametrize("text,reason", [
    ('reason</think>answer', 'reason'), ('<think>reason</think>answer', 'reason'),
    (' \nreason</think>answer', ' \nreason'), (' \n<think>reason</think>answer', ' \nreason'),
])
async def test_qwen_raw_implicit_and_explicit_prefixes_remain_supported(text, reason):
    request = NativeRequest("completion", {}, reasoning_format="think_prefill")
    text += ' <think>literal</think>'
    events = await collected([*[{'content': c} for c in text],
                              {'stop': True, 'stop_type': 'eos'}], request)
    result = read_result({'content': text, 'stop': True, 'stop_type': 'eos'}, request)
    assert result.reasoning == ''.join(e.reasoning for e in events) == reason
    assert result.content == ''.join(e.content for e in events) == 'answer <think>literal</think>'


@pytest.mark.parametrize("outcome", ['missing_finish', 'cancel'])
async def test_interrupted_fallback_does_not_invent_a_success(outcome):
    async def upstream():
        yield delta('<think>partial reasoning')
        if outcome == 'cancel':
            raise asyncio.CancelledError()
    seen = []
    with pytest.raises(asyncio.CancelledError if outcome == 'cancel' else WorkerProtocolError):
        async for event in _read_events(upstream(), REQUEST):
            seen.append(event)
    assert seen and not any(e.terminal for e in seen)


async def test_early_close_closes_native_upstream_without_success():
    from tllama.native_api import NativeAPI
    from contextlib import asynccontextmanager
    closed = []

    class Worker:
        @asynccontextmanager
        async def chat_stream(self, body):
            try:
                yield sequence(delta('<think>partial reasoning'))
            finally:
                closed.append(True)

    async with NativeAPI(Worker(), REQUEST).stream() as events:
        assert (await anext(events)).reasoning
    assert closed == [True]


@pytest.mark.parametrize("path", ['/api/chat', '/api/generate', '/v1/chat/completions'])
@pytest.mark.parametrize("stream", [False, True])
def test_public_instella_channels_without_model_name_hint(client, native_reply, path, stream):
    state = native_reply(metadata=METADATA, content=TEXT)
    request = {"model": "renamed", "stream": stream, "prompt": 'Předveď se!',
               "messages": [{"role": "user", "content": 'Předveď se!'}]}
    response = client.post(path, json=request)
    assert response.status_code == 200
    if path.startswith('/v1'):
        if stream:
            bodies = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            parts = [c['delta'] for body in bodies for c in body['choices']]
        else:
            parts = [response.json()['choices'][0]['message']]
        reasoning_key = 'reasoning_content'
    else:
        bodies = [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]
        assert bodies[-1]['done']
        parts = [body['message'] if path.endswith('chat') else
                 {'content': body.get('response', ''), 'thinking': body.get('thinking', '')} for body in bodies]
        reasoning_key = 'thinking'
    assert ''.join(p.get('content', '') for p in parts) == ANSWER
    assert ''.join(p.get(reasoning_key, '') for p in parts) == THOUGHT
    assert state.refs == 0
    if path != '/api/generate':
        # Internal fallback policy must never replace native's wire setting.
        assert state.worker.calls[-1][1]['reasoning_format'] == 'deepseek'


@pytest.mark.parametrize("hidden", [False, True])
async def test_repl_full_http_path_history_color_hide_and_statistics(native_reply, monkeypatch, hidden):
    native_reply(metadata=METADATA, content='<think>reason</think>answer')
    out, err = Terminal(tty=True), Terminal(tty=True)
    monkeypatch.setattr('sys.stdout', out)
    monkeypatch.setattr('sys.stderr', err)
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    session = RunSession('renamed', 'completion', hide_thinking=hidden, verbose=True)
    async with TLlamaClient(host='http://public', transport=httpx.ASGITransport(app=app)) as client:
        await session.submit(client, 'Předveď se!')
    assert out.getvalue() == 'answer\n'
    assert session.messages[-1] == {'role': 'assistant', 'content': 'answer', 'thinking': 'reason'}
    assert 'Statistics:' in err.getvalue() and 'First text:' in err.getvalue()
    assert '542 tokens' in err.getvalue()
    assert ('Thinking:' in err.getvalue()) is not hidden
    assert ('Answer:' in err.getvalue()) is not hidden
    assert ('\x1b[90mThinking:\nreason\x1b[0m' in err.getvalue()) is not hidden


@pytest.mark.parametrize('value', [False, 'none'])
def test_disable_is_forwarded_but_cannot_hide_reasoning_emitted_by_model(client, native_reply, value):
    state = native_reply(metadata=METADATA, content='<think>reason</think>answer')
    response = client.post('/api/chat', json={'model': 'renamed', 'messages': [{'role': 'user', 'content': 'hi'}],
                                             'think': value, 'stream': False})
    assert response.json()['message'] == {'role': 'assistant', 'content': 'answer', 'thinking': 'reason'}
    body = state.worker.calls[-1][1]
    assert body['reasoning_effort'] == 'none'
    assert body['chat_template_kwargs']['enable_thinking'] is False
