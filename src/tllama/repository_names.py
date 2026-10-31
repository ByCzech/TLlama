"""Canonical, readable storage names shared by repositories and migration.

This is a filename codec, not URL decoding. Only storage path components
are decoded, exactly once; TOML file paths and user import paths stay literal.
"""

from pathlib import PurePosixPath
import re
import unicodedata


_ESCAPED = frozenset('<>:"|?*%')
_RESERVED = frozenset({'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} |
                      {prefix + number for prefix in ('COM', 'LPT')
                       for number in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '¹', '²', '³')})
_AREAS = {1: 'Local', 2: 'TLlama', 3: 'HuggingFace'}


def _validate_component(value):
    if (not isinstance(value, str) or value in {'', '.', '..'}
            or '/' in value or '\\' in value
            or any(unicodedata.category(char) in {'Cc', 'Cs'} for char in value)):
        raise ValueError('Invalid repository name component.')


def encode_component(value):
    """Encode an actual filename/directory component, including its suffix.

    Ordinary Unicode and internal spaces remain readable. Terminal dots and
    spaces are escaped only at the actual end, after a caller adds .toml.
    Reserved Windows basenames are escaped at their first character.
    """
    _validate_component(value)
    end = len(value.rstrip(' .'))
    encoded = ''.join(f'%{ord(char):02X}' if char in _ESCAPED or index >= end
                      else char for index, char in enumerate(value))
    if encoded.split('.', 1)[0].rstrip(' ').upper() in _RESERVED:
        encoded = f'%{ord(encoded[0]):02X}' + encoded[1:]
    # Common long-name filesystems count either bytes or UTF-16 code units.
    # Absolute path limits still depend on the chosen repository root.
    if len(encoded.encode('utf-8')) > 255 or len(encoded.encode('utf-16-le')) // 2 > 255:
        raise ValueError('Encoded repository name component exceeds 255 bytes or UTF-16 units.')
    return encoded


def decode_component(value):
    """Accept only our canonical spelling; reject traversal and URL escapes."""
    if not isinstance(value, str):
        raise ValueError('Invalid encoded repository name component.')
    decoded = []
    index = 0
    while index < len(value):
        if value[index] == '%':
            token = value[index + 1:index + 3]
            if not re.fullmatch('[0-9A-F]{2}', token):
                raise ValueError('Invalid repository percent escape.')
            decoded.append(chr(int(token, 16)))
            index += 3
        else:
            decoded.append(value[index])
            index += 1
    decoded = ''.join(decoded)
    if encode_component(decoded) != value:
        raise ValueError('Noncanonical encoded repository name component.')
    return decoded


def definition_path(name):
    """Map a public model name to its portable repository-relative TOML path."""
    if not isinstance(name, str) or name != name.strip():
        raise ValueError('Invalid model name.')
    parts = name.split('/')
    if len(parts) not in _AREAS:
        raise ValueError('Model names support one to three path components.')
    for part in parts:
        _validate_component(part)
    return PurePosixPath(_AREAS[len(parts)], *(encode_component(part) for part in parts[:-1]),
                         encode_component(parts[-1] + '.toml'))


def definition_name(relative_path):
    """Inverse of definition_path; never infer a repository's storage format."""
    value = str(relative_path)
    if '\\' in value:
        raise ValueError('Repository-relative paths use POSIX separators.')
    parts = value.split('/')
    if (len(parts) - 1 not in _AREAS or _AREAS[len(parts) - 1] != parts[0]
            or not parts[-1].endswith('.toml')):
        raise ValueError('Invalid repository definition path.')
    logical = [decode_component(part) for part in parts[1:]]
    logical[-1] = logical[-1][:-5]
    name = '/'.join(logical)
    if definition_path(name).as_posix() != value:
        raise ValueError('Noncanonical repository definition path.')
    return name


def check_portable_collisions(paths):
    """Reject case/normalization and file-vs-directory conflicts on every OS.

    Preserve names; normalization is only a conservative collision check.
    It is deliberately stricter than a case-sensitive development filesystem.
    """
    nodes = {}
    for path in paths:
        parts = str(path).split('/')
        key = ()
        for index, part in enumerate(parts):
            _validate_component(part)
            key += (unicodedata.normalize('NFC', part).casefold(),)
            node = ('/'.join(parts[:index + 1]), index == len(parts) - 1)
            previous = nodes.setdefault(key, node)
            if previous != node:
                raise ValueError(f'Portable repository path collision: {previous[0]!r} and {node[0]!r}.')
