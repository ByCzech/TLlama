"""Foreground server and local repository maintenance entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from tllama.maintenance import add_rebuild_parser
from tllama.version import get_version
from tllama.daemon_options import add_daemon_options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tllamad", description="TLlama server and local repository maintenance.")
    parser.add_argument("--version", action="version", version=f"tllamad {get_version()}")
    add_daemon_options(parser)
    add_rebuild_parser(parser.add_subparsers(dest="command"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tllama.config import ConfigError

    try:
        from tllama.daemon_config import load_daemon_config, daemon_configuration
        overrides = {key: value for key, value in vars(args).items() if '.' in key}
        config = load_daemon_config(False if args.no_config else args.config, overrides,
                                    discover=not args.no_config)
        if args.check_config or args.print_config:
            if args.command is not None:
                raise ConfigError('Configuration inspection cannot be combined with a maintenance command')
            print(json.dumps(config.report(), ensure_ascii=False, allow_nan=False, indent=2)
                  if args.print_config else 'Configuration is valid.')
            return 0
        with daemon_configuration(config, overrides):
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
