"""Foreground server and local repository maintenance entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from tllama.maintenance import add_rebuild_parser
from tllama.version import get_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tllamad", description="TLlama server and local repository maintenance.")
    parser.add_argument("--version", action="version", version=f"tllamad {get_version()}")
    add_rebuild_parser(parser.add_subparsers(dest="command"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tllama.config import ConfigError

    try:
        if args.command is not None:
            return args.handler(args)
        from tllama.main import start_server
        start_server()
        return 0
    except ConfigError as exc:
        print(f"tllamad: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
