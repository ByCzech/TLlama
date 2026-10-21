"""Validate named GGUF import groups using header-declared shard positions."""

from pathlib import Path
import re

from tllama.helpers.gguf_metadata import read_gguf_metadata


def shard_names(filename, metadata):
    count = metadata.get('split.count', 1)
    index = metadata.get('split.no', 0)
    if type(count) is not int or not 1 <= count <= 2048 or type(index) is not int or index != 0:
        raise ValueError('Import must start with the first GGUF shard (at most 2048 shards).')
    if count == 1:
        return [filename]
    match = re.fullmatch(r'(.+)-00001-of-(\d{5})\.gguf', filename)
    if not match or int(match[2]) != count:
        raise ValueError('Split GGUF filenames must use NAME-00001-of-NNNNN.gguf and match the header.')
    return [f'{match[1]}-{number:05d}-of-{count:05d}.gguf' for number in range(1, count + 1)]


def validate_group(names, paths):
    if not names or len(names) != len(paths):
        raise ValueError('An import group must contain its complete ordered GGUF files.')
    try:
        metadata = [read_gguf_metadata(str(path)) for path in paths]
    except Exception as exc:
        raise ValueError(f'Cannot read uploaded GGUF header: {exc}') from exc
    expected = shard_names(names[0], metadata[0])
    if names != expected:
        raise ValueError('Import does not contain the complete ordered GGUF shard set.')
    for index, header in enumerate(metadata):
        if header.get('split.count', 1) != len(names) or header.get('split.no', 0) != index:
            raise ValueError('GGUF shard headers disagree with the import group.')
    return metadata


def local_group(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Import source is not a file: {path}')
    try:
        names = shard_names(path.name, read_gguf_metadata(str(path)))
    except Exception as exc:
        raise ValueError(f'Cannot prepare GGUF import {path}: {exc}') from exc
    paths = [path.parent / name for name in names]
    if any(not item.is_file() for item in paths):
        raise FileNotFoundError('A GGUF continuation shard is missing beside the first shard.')
    validate_group(names, paths)
    return paths
