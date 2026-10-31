"""Ordered files belonging to a model definition, without loading a worker."""

from dataclasses import dataclass
from pathlib import Path
import re

from tllama.helpers.gguf_import import shard_names
from tllama.helpers.model_toml import TomlModelError


def gguf_files(path, read_metadata, root):
    path = Path(path)
    try:
        header = read_metadata(path)
    except (ValueError, OSError):
        # Preserve listings of ordinary files whose optional GGUF metadata
        # cannot be read; the native loader remains authoritative. A split
        # filename cannot safely be reported as a complete single file.
        if re.fullmatch(r'.+-\d{5}-of-\d{5}\.gguf', path.name):
            raise TomlModelError(f'Cannot inspect GGUF shard set: {path}') from None
        header = {}
    raw = header.get('metadata_raw', {})
    names = shard_names(path.name, raw)
    files = tuple(path.parent / name for name in names)
    for index, file in enumerate(files):
        if not file.resolve().is_relative_to(Path(root).resolve()) or not file.is_file():
            raise TomlModelError(f'Missing or external GGUF shard: {file}')
        if len(files) > 1:
            part = read_metadata(file).get('metadata_raw', {})
            if part.get('split.count') != len(files) or part.get('split.no') != index:
                raise TomlModelError(f'Inconsistent GGUF shard header: {file}')
    return files


@dataclass(frozen=True)
class ModelArtifacts:
    llm: tuple[Path, ...]
    projector: tuple[Path, ...] = ()
    adapter: tuple[Path, ...] = ()

    @property
    def files(self):
        # Shared paths/hard links contribute once within one definition.
        found = {}
        for path in self.llm + self.projector + self.adapter:
            stat = path.stat()
            found.setdefault((stat.st_dev, stat.st_ino), path)
        return tuple(found.values())

    @property
    def size(self):
        return sum(path.stat().st_size for path in self.files)
