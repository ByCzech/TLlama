"""Ordered files belonging to a model definition, without loading a worker."""

from dataclasses import dataclass
from pathlib import Path
import re

from tllama.helpers.gguf_import import shard_names
from tllama.helpers.model_toml import TomlModelError


def gguf_files(path, read_metadata, root, *, repository=None):
    path = Path(path)
    # Reusing the first shard's header must not miss a replacement that the
    # former second read would have observed. Other files keep their own
    # header/containment checks; callers still compare full artifact snapshots.
    split_name = re.fullmatch(r'.+-\d{5}-of-\d{5}\.gguf', path.name)
    def revision():
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
    try:
        before = revision() if split_name else None
    except OSError:
        raise TomlModelError(f'Cannot inspect GGUF shard set: {path}') from None
    try:
        header = read_metadata(path)
    except (ValueError, OSError):
        # Preserve listings of ordinary files whose optional GGUF metadata
        # cannot be read; the native loader remains authoritative. A split
        # filename cannot safely be reported as a complete single file.
        if split_name:
            raise TomlModelError(f'Cannot inspect GGUF shard set: {path}') from None
        header = {}
    raw = header.get('metadata_raw', {})
    if ('split.count' not in raw and header.get('arch') in (None, '', 'unknown')
            and split_name):
        raise TomlModelError(f'Cannot inspect GGUF shard set: {path}')
    names = shard_names(path.name, raw)
    files = tuple(path.parent / name for name in names)
    physical_root = repository.physical_root if repository is not None else Path(root).resolve()
    for index, file in enumerate(files):
        if not file.resolve().is_relative_to(physical_root) or not file.is_file():
            raise TomlModelError(f'Missing or external GGUF shard: {file}')
        if len(files) > 1:
            part = (header if index == 0 else read_metadata(file)).get('metadata_raw', {})
            if part.get('split.count') != len(files) or part.get('split.no') != index:
                raise TomlModelError(f'Inconsistent GGUF shard header: {file}')
    if before is not None and before != revision():
        raise TomlModelError(f'GGUF first shard changed during inspection: {path}')
    return files


@dataclass(frozen=True)
class ModelArtifacts:
    llm: tuple[Path, ...]
    projector: tuple[Path, ...] = ()
    adapter: tuple[Path, ...] = ()
    draft: tuple[Path, ...] = ()

    def _file_stats(self):
        """One fresh inode inventory per calculation; never retained on self."""
        found = {}
        for path in self.llm + self.projector + self.adapter + self.draft:
            stat = path.stat()
            found.setdefault((stat.st_dev, stat.st_ino), (path, stat))
        return tuple(found.values())

    @property
    def files(self):
        # Shared paths/hard links contribute once within one definition.
        return tuple(path for path, stat in self._file_stats())

    @property
    def size(self):
        return sum(stat.st_size for path, stat in self._file_stats())
