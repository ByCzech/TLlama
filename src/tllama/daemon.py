"""Foreground server entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from tllama.version import get_version
from tllama.daemon_options import add_daemon_options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tllamad", description="TLlama server.")
    parser.add_argument("--version", action="version", version=f"tllamad {get_version()}")
    add_daemon_options(parser)
    commands = parser.add_subparsers(dest='command')
    repository = commands.add_parser('repo', help='Local model repository maintenance; no HTTP connection.')
    operations = repository.add_subparsers(dest='repo_operation', required=True)
    for name, help_text in (
            ('check', 'Read-only check of the local model repository.'),
            ('prune', 'Preview unused managed groups; --apply removes them locally.')):
        command = operations.add_parser(name, help=help_text)
        selection = command.add_mutually_exclusive_group()
        selection.add_argument('--config', default=argparse.SUPPRESS, metavar='PATH', help='Server TOML file.')
        selection.add_argument('--no-config', action='store_true', default=argparse.SUPPRESS,
                               help='Ignore server TOML selection and discovery.')
        command.add_argument('--models-dir', dest='storage.models_dir', default=argparse.SUPPRESS,
                             metavar='PATH', help='Local model repository path (overrides server configuration).')
        command.add_argument('--json', action='store_true', help='Print the full local report as JSON.')
        if name == 'check':
            command.add_argument('--hashes', action='store_true', help='Independently read and verify full GGUF contents.')
        else:
            command.add_argument('--apply', action='store_true', help='Remove currently proved unused groups; default is preview.')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config and args.no_config:
        parser.error('--config and --no-config cannot be combined')
    if args.command == 'repo' and (args.check_config or args.print_config or args.list_devices):
        parser.error('repo commands cannot be combined with configuration/device inspection flags')
    from tllama.config import ConfigError

    try:
        from tllama.daemon_config import load_daemon_config, daemon_configuration
        overrides = {key: value for key, value in vars(args).items() if '.' in key}
        config = load_daemon_config(False if args.no_config else args.config, overrides,
                                    discover=not args.no_config)
        if args.command == 'repo':
            if args.repo_operation == 'check':
                from tllama.repository_check_cli import run_check
                return run_check(config.backend.models_dir, hashes=args.hashes, as_json=args.json)
            from tllama.repository_check_cli import run_prune
            return run_prune(config.backend.models_dir, apply=args.apply, as_json=args.json)
        if args.list_devices:
            from tllama.daemon_devices import list_devices
            return list_devices(config.values['engine.executable'])
        if args.check_config or args.print_config:
            print(json.dumps(config.report(), ensure_ascii=False, allow_nan=False, indent=2)
                  if args.print_config else 'Configuration is valid.')
            return 0
        with daemon_configuration(config, overrides):
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
