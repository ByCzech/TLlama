"""Offline Tools advertisement follows templates, never model names or loads."""

import json

import gguf
import httpx
import pytest

from tllama.helpers.tool_capability import template_accepts_tools
from tllama.helpers.metadata_cache import get_metadata_cache_path, load_digest_cache, save_digest_cache
from tllama.main import app
from tllama.routers import ollama

from test_native_manager import native_manager


TOOLS = '{% if tools %}{% for tool in tools %}{{ tool | tojson }}{% endfor %}{% endif %}'
PLAIN = '{% for message in messages %}{{ message.content }}{% endfor %}'


@pytest.mark.parametrize('template,expected', [
    (TOOLS, True),
    ('{{ tools | default([]) | tojson }}', True),
    ('{% set available = tools %}{{ available }}', True),
    ('{{ raise_exception("must never execute") }}{{ tools }}', True),
    ('{% for tool in tools %}{% if loop.index > 2 %}{% break %}{% endif %}{{ tool }}{% endfor %}', True),
    (PLAIN, False),
    ('{# {{ tools }} #}{{ messages }}', False),
    ('{{ "tools" }}', False),
    ('{% raw %}{{ tools }}{% endraw %}', False),
    ('{{ message.tools }}', False),
    ('{% set tools = [] %}{{ tools }}', False),
    ('{% macro show(tools) %}{{ tools }}{% endmacro %}', False),
    ('{% for tools in messages %}{{ tools }}{% endfor %}', False),
    ('{% if tools %}', False),
    ('{{ tools | native_only_filter }}', False),
    ('{% native_only_tag %}{{ tools }}', False),
    (' ' * 262145, False),
    ('', False),
    (None, False),
    (['{{ tools }}'], False),
])
def test_external_tools_input_without_rendering(template, expected):
    assert template_accepts_tools(template) is expected


@pytest.fixture
async def offline(native_manager, monkeypatch):
    manager = native_manager()
    # Even a recognisable family/name must not invent Tools support.
    name, toml = manager.test.define('Qwen-tools-instruct')
    weights = manager.local_models_dir / f'{name}.gguf'

    def write(default=None, tool_use=None):
        writer = gguf.GGUFWriter(path=str(weights), arch='qwen3')
        for key, value in [('tokenizer.chat_template', default),
                           ('tokenizer.chat_template.tool_use', tool_use)]:
            if value is not None:
                writer.add_string(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

    monkeypatch.setattr(ollama, 'model_manager', manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://public') as client:
        yield manager, client, name, toml, write
    assert not manager.test.ledger
    assert manager.native_snapshot() == []


@pytest.mark.parametrize('default,tool_use,override,expected', [
    (TOOLS, None, None, True),
    (PLAIN, TOOLS, None, True),
    (None, TOOLS, None, True),
    (TOOLS, PLAIN, None, False),
    (TOOLS, None, PLAIN, False),
    (PLAIN, TOOLS, PLAIN, False),
    (PLAIN, None, TOOLS, True),
    (PLAIN, PLAIN, TOOLS, True),
    (PLAIN, TOOLS, '', True),  # Empty native override falls back to GGUF.
    (PLAIN, None, None, False),
    (None, None, None, False),
    ('{# tools #}{{ messages }}', None, None, False),
    ('{% if tools %}', None, None, False),
])
async def test_show_uses_native_template_precedence_without_worker(offline, default, tool_use, override, expected):
    manager, client, name, toml, write = offline
    write(default, tool_use)
    if override is not None:
        toml.write_text(toml.read_text() + '\n[template]\njinja=' + json.dumps(override))
    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200, response.text
    assert ('tools' in response.json()['capabilities']) is expected
    assert 'completion' in response.json()['capabilities']


async def test_definition_edits_apply_over_cached_named_template(offline, monkeypatch):
    manager, client, name, toml, write = offline
    write(PLAIN, TOOLS)
    original = toml.read_text()
    scans = []
    scan = manager._get_model_metadata_sync

    def counted(path):
        scans.append(path)
        return scan(path)

    monkeypatch.setattr(manager, '_get_model_metadata_sync', counted)
    for override, expected in [(None, True), (PLAIN, False), (TOOLS, True), (None, True)]:
        toml.write_text(original if override is None else
                        original + '\n[template]\njinja=' + json.dumps(override))
        response = await client.post('/api/show', json={'model': name})
        assert response.status_code == 200, response.text
        assert ('tools' in response.json()['capabilities']) is expected
        # Exercise the persistent header cache as well as fresh TOML layering.
        manager._metadata_cache.clear()
    assert len(scans) == 1


async def test_pre_tools_header_cache_is_rebuilt_without_invalidating_weight_digest(offline):
    manager, client, name, _, write = offline
    write(PLAIN, TOOLS)
    weights = manager.local_models_dir / f'{name}.gguf'
    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200 and 'tools' in response.json()['capabilities']
    cache_path = get_metadata_cache_path(manager.metadata_cache_dir, weights)
    old = json.loads(cache_path.read_text())
    old['schema_version'] = 3
    old['metadata'].pop('tool_template')
    old['metadata']['metadata_raw'].pop('tokenizer.chat_template.tool_use')
    cache_path.write_text(json.dumps(old))
    digest = {'sha256': 'a' * 64}
    save_digest_cache(manager.metadata_cache_dir, name, weights, digest)
    manager._metadata_cache.clear()

    response = await client.post('/api/show', json={'model': name})
    assert response.status_code == 200 and 'tools' in response.json()['capabilities']
    assert json.loads(cache_path.read_text())['schema_version'] > 3
    assert load_digest_cache(manager.metadata_cache_dir, weights) == digest
