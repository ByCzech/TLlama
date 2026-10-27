#!/usr/bin/env python3
"""Preview a TOML's native launch without loading weights or starting a process."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from tllama.daemon_config import load_daemon_config
from tllama.helpers.model_toml import parse_model_toml
from tllama.native_config import build_native_sampling, resolve_worker_config


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--ctx", type=int, help="Preview a request num_ctx override")
    args = parser.parse_args()
    try:
        config = load_daemon_config(False, discover=False).backend
        if args.models_dir is not None:
            config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
        root = Path(config.models_dir).expanduser().resolve()
        spec = parse_model_toml(args.toml.read_text(), source=str(args.toml))

        def storage_path(reference):
            if reference is None:
                return None
            path = (root / reference).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Model/projector reference must resolve inside the model store")
            return path

        if spec.llm_model is None:
            raise ValueError("Import the [llm].from definition before previewing its native launch")
        if spec.mmproj_from is not None:
            raise ValueError("Import the [mmproj].from definition before previewing its native launch")
        resolved = resolve_worker_config(config, storage_path(spec.llm_model), spec,
                                         num_ctx=args.ctx, mmproj=storage_path(spec.mmproj_model))
        # Illustrative private paths only; NativeWorker creates the actual files.
        command = (resolved.launch.command(Path("/private/api-key"), Path("/private/chat-template.jinja"))
                   if resolved.launch.device is not None else None)
        print(json.dumps({"dry_run": True, "runtime_fingerprint": resolved.launch.runtime_fingerprint,
                          "runtime": plain(resolved.runtime), "sources": plain(resolved.sources),
                          "argv": command, "native_sampling": build_native_sampling({}, resolved=resolved),
                          "backend_priority": resolved.launch.backend_priority,
                          "automatic_device_selection": resolved.launch.device is None,
                          "system_prompt": resolved.system_prompt, "gguf_metadata_read": False,
                          "private_paths": "illustrative; created by NativeWorker at start"},
                         ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
