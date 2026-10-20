"""Client entry point; importing it never constructs a server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from typing import Sequence

from tllama.version import get_version


def _positive_timeout(value):
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('timeout must be a finite number greater than zero')


def _client_options(parser):
    # Suppressed subparser defaults preserve flags supplied before the command.
    parser.add_argument('--host', default=argparse.SUPPRESS,
                        help='Server HTTP(S) address (overrides TLLAMA_HOST; default 127.0.0.1:54800).')
    parser.add_argument('--timeout', type=_positive_timeout, default=argparse.SUPPRESS,
                        help='HTTP inactivity timeout in seconds (default 30; connect at most 10).')
    parser.add_argument('--json', action='store_true', default=argparse.SUPPRESS,
                        help='Print the full server response as JSON.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='tllama', description='TLlama HTTP client.')
    parser.add_argument('--version', action='version', version=f'tllama {get_version()}')
    _client_options(parser)
    commands = parser.add_subparsers(dest='command')
    for name, aliases, help_text in [
        ('list', ['ls'], 'List models available on the server.'),
        ('show', [], 'Show model information or its definition.'),
        ('ps', [], 'Show resident models and reported memory placement.'),
        ('version', [], 'Show client and server versions.'),
    ]:
        command = commands.add_parser(name, aliases=aliases, help=help_text)
        _client_options(command)
        command.set_defaults(operation=name)
        if name == 'show':
            command.add_argument('model')
            fields = command.add_mutually_exclusive_group()
            for field in ('modelfile', 'parameters', 'template', 'system', 'license'):
                fields.add_argument('--' + field, dest='field', action='store_const', const=field,
                                    help=f'Print only the {field} text.')
    return parser


async def _execute(args):
    from tllama.client import TLlamaClient
    from tllama.cli_output import models_text, processes_text, show_text

    async with TLlamaClient(getattr(args, 'host', None), timeout=getattr(args, 'timeout', 30.0)) as client:
        if args.operation == 'list':
            data = await client.list_models()
            text = models_text(data)
        elif args.operation == 'ps':
            data = await client.ps()
            text = processes_text(data)
        elif args.operation == 'show':
            data = await client.show(args.model)
            text = show_text(args.model, data, args.field)
        else:
            data = {'client_version': get_version(), 'server_version': await client.version()}
            text = f'tllama client: {data["client_version"]}\ntllama server: {data["server_version"]}'
        if getattr(args, 'json', False):
            print(json.dumps(data, ensure_ascii=False))
        elif args.operation == 'show' and args.field:
            # Preserve a retrieved definition verbatim, including its final newline.
            sys.stdout.write(text)
        else:
            print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if getattr(args, 'json', False) and getattr(args, 'field', None):
        parser.error('--json cannot be combined with a show text selector')
    from tllama.client import ClientConfigError, ClientError

    try:
        return asyncio.run(_execute(args))
    except ClientConfigError as exc:
        print(f'tllama: {exc}', file=sys.stderr)
        return 2
    except ClientError as exc:
        print(f'tllama: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
