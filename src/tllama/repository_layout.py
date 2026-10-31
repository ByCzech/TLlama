"""Names for immutable content-addressed GGUF groups, with no filesystem I/O.

File bytes, shard order and the versioned group digest define the directory.
Public model names, source filenames, templates and parameters do not.
"""

from pathlib import PurePosixPath
import re

from tllama.model_identity import file_group_digest


REPOSITORY_FORMAT = {'version': 2, 'names': 'percent-v1', 'data': 'sha256-groups-v1'}


def group_paths(digests):
    """Return canonical relative paths for an ordered complete GGUF group.

    Callers must validate headers and stability of the hashed files before
    publication. A path calculation alone does not verify uploaded content.
    """
    digests = list(digests)
    if (not 1 <= len(digests) <= 2048 or any(not isinstance(value, str)
            or not re.fullmatch('[0-9a-f]{64}', value) for value in digests)):
        raise ValueError('A data group requires 1..2048 lowercase SHA-256 digests.')
    root = PurePosixPath('data', file_group_digest(digests))
    if len(digests) == 1:
        return (root / 'weights.gguf',)
    return tuple(root / f'weights-{number:05d}-of-{len(digests):05d}.gguf'
                 for number in range(1, len(digests) + 1))
