"""Portable storage preserves public names and rejects ambiguous spellings."""

from pathlib import PurePosixPath
import unicodedata

import pytest

from tllama.repository_names import (encode_component, decode_component,
    definition_path, definition_name, check_portable_collisions)
from tllama.repository_layout import group_paths
from tllama.model_identity import file_group_digest


@pytest.mark.parametrize('logical,physical', [
    ('model:55b', 'Local/model%3A55b.toml'),
    ('můj model:55b', 'Local/můj model%3A55b.toml'),
    ('model%3A55b', 'Local/model%253A55b.toml'),
    ('a+b', 'Local/a+b.toml'),
    ('model.', 'Local/model..toml'),
    ('author./model', 'TLlama/author%2E/model.toml'),
    ('author /model', 'TLlama/author%20/model.toml'),
    ('ns/repo/model', 'HuggingFace/ns/repo/model.toml'),
    ('CON', 'Local/%43ON.toml'),
    ('con.txt', 'Local/%63on.txt.toml'),
    ('CON:55b', 'Local/CON%3A55b.toml'),
    ('NUL/model', 'TLlama/%4EUL/model.toml'),
    ('COM¹', 'Local/%43OM¹.toml'),
    ('CONOUT$', 'Local/%43ONOUT$.toml'),
])
def test_public_name_roundtrip(logical, physical):
    assert definition_path(logical) == PurePosixPath(physical)
    assert definition_name(physical) == logical


def test_percent_is_decoded_once_and_unicode_spelling_is_preserved():
    assert decode_component('m%253A') == 'm%3A'
    for text in ('é', unicodedata.normalize('NFD', 'é'), '東京', '🦙'):
        assert decode_component(encode_component(text)) == text


@pytest.mark.parametrize('value', ['', '.', '..', 'a/b', 'a\\b', 'a\x00b', 'a\x1fb', 'a\x7fb', 'a\x85b', '\ud800'])
def test_unsafe_components_are_rejected(value):
    with pytest.raises(ValueError):
        encode_component(value)


@pytest.mark.parametrize('value', ['%2F', '%5C', '%00', '%2E%2E', '%3a', '%41', '%C3%A9', '%', '%2', 'CON.toml', 'a.', 'a '])
def test_decoder_accepts_only_canonical_storage_names(value):
    with pytest.raises(ValueError):
        decode_component(value)


@pytest.mark.parametrize('value', ['a//b', '/a', 'a/', 'a/../b', 'a/b/c/d', ' a', 'a ', '.'])
def test_invalid_logical_references_are_rejected(value):
    with pytest.raises(ValueError):
        definition_path(value)


@pytest.mark.parametrize('value', ['Local/a/b.toml', 'TLlama/a.toml', 'Local/../a.toml',
    '/Local/a.toml', 'Local/a.TOML', 'Local/%2F.toml', 'HuggingFace/ns/repo/a/b.toml'])
def test_decoder_validates_namespace_depth_and_traversal(value):
    with pytest.raises(ValueError):
        definition_name(value)


def test_limits_are_checked_after_encoding_and_appending_suffix():
    assert len(encode_component('a' * 255)) == 255
    for text in (':' * 86, 'a' * 256, '界' * 86):
        with pytest.raises(ValueError, match='255'):
            encode_component(text)
    with pytest.raises(ValueError, match='255'):
        definition_path('a' * 251)


@pytest.mark.parametrize('paths', [
    ['Local/Model.toml', 'Local/model.toml'],
    ['TLlama/Author/a.toml', 'TLlama/author/b.toml'],
    ['Local/é.toml', 'Local/e\u0301.toml'],
    ['Local/name.toml', 'Local/name.toml/sub.toml'],
])
def test_collisions_are_rejected_even_on_linux(paths):
    with pytest.raises(ValueError, match='collision'):
        check_portable_collisions(paths)


def test_distinct_names_and_shared_directories_are_allowed():
    check_portable_collisions(['Local/a.toml', 'Local/a%3Ab.toml', 'Local/a%253Ab.toml',
                               'TLlama/author/a.toml', 'TLlama/author/b.toml'])


def test_single_group_uses_content_hash_and_canonical_filename():
    assert group_paths(['a' * 64]) == (PurePosixPath('data', 'a' * 64, 'weights.gguf'),)


def test_shards_share_a_directory_and_match_existing_identity():
    digests = ['a' * 64, 'b' * 64, 'c' * 64]
    paths = group_paths(digests)
    assert len({path.parent for path in paths}) == 1
    assert paths[0].parent.name == file_group_digest(digests)
    assert [path.name for path in paths] == [f'weights-{n:05d}-of-00003.gguf' for n in (1, 2, 3)]
    assert group_paths(digests[::-1])[0].parent != paths[0].parent
    assert group_paths([digests[0]])[0].parent != paths[0].parent


@pytest.mark.parametrize('digests', [[], [''], ['g' * 64], ['A' * 64], ['../bad'], [None], ['a' * 64] * 2049])
def test_invalid_group_identities_do_not_form_paths(digests):
    with pytest.raises(ValueError):
        group_paths(digests)
