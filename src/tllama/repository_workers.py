"""Repository resources used by one immutable native launch snapshot."""

from pathlib import Path
from dataclasses import replace
import re

from tllama.repository_runtime import read_format
from tllama.repository_store import RepositoryStore


def resolve_launch(launch):
    """Group whole shard directories, including an independent projector/LoRA.

    Physical diagnostics can also use ordinary files outside managed data;
    those files are never candidates for repository GC. Only explicitly
    marked repositories participate, including when reached through an alias.
    """
    resources = {}
    paths = [Path(value).resolve() if value is not None else None
             for value in (launch.model, launch.mmproj, launch.runtime_options.get('lora_path'))]
    for value in paths:
        if value is None:
            continue
        path = value
        if (path.parent.parent.name != 'data'
                or not re.fullmatch('[0-9a-f]{64}', path.parent.name)):
            continue
        root = path.parents[2]
        store = RepositoryStore(root)
        if read_format(store) is not None:
            resources.setdefault(root, set()).add(path.parent.name)
    # Freeze even currently unmanaged inputs: a symlink must not be retargeted
    # into managed data after the decision that no repository pin is needed.
    options = dict(launch.runtime_options)
    if paths[2] is not None:
        options['lora_path'] = str(paths[2])
    launch = replace(launch, model=paths[0], mmproj=paths[1], runtime_options=options)
    return launch, resources


def launch_groups(launch):
    return resolve_launch(launch)[1]
