"""Client entry point; importing it never constructs a server."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import json
import math
import sys
from typing import Sequence

from tllama.version import get_version

WAIT_NOTICE_AFTER = 2.0


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
                        help='HTTP inactivity limit in seconds (default: no read limit for list/show/pull, otherwise 30; connect at most 10).')
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
        ('pull', [], 'Download a Hugging Face model through the server.'),
        ('rm', [], 'Delete model definitions; keep their weight files.'),
        ('stop', [], 'Request unload after active inference leases drain.'),
    ]:
        command = commands.add_parser(name, aliases=aliases, help=help_text)
        _client_options(command)
        command.set_defaults(operation=name)
        if name in {'pull', 'stop'}:
            command.add_argument('model')
        elif name == 'rm':
            command.add_argument('models', nargs='+')
        elif name == 'show':
            command.add_argument('model')
            fields = command.add_mutually_exclusive_group()
            for field in ('modelfile', 'parameters', 'template', 'system', 'license'):
                fields.add_argument('--' + field, dest='field', action='store_const', const=field,
                                    help=f'Print only the {field} text.')
    return parser


async def _wait_for_model_info(awaitable, *, notice):
    if not notice:
        return await awaitable
    task = asyncio.create_task(awaitable)
    try:
        await asyncio.wait({task}, timeout=WAIT_NOTICE_AFTER)
        if not task.done():
            print('Waiting for TLlama to return model information; metadata indexing may take time. Ctrl+C cancels.',
                  file=sys.stderr, flush=True)
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _execute(args):
    from tllama.client import ClientError, TLlamaClient
    from tllama.cli_output import models_text, processes_text, show_text

    as_json = getattr(args, 'json', False)
    async with TLlamaClient(getattr(args, 'host', None), timeout=getattr(args, 'timeout', None)) as client:
        if args.operation == 'pull':
            previous = None
            async with aclosing(client.pull(args.model)) as progress:
                async for event in progress:
                    if as_json:
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    else:
                        from tllama.cli_output import size_text
                        text = event['status']
                        if 'completed' in event:
                            text += f": {size_text(event['completed'])} / {size_text(event.get('total'))}"
                        if text != previous:
                            print(text, file=sys.stderr, flush=True)
                            previous = text
            if not as_json:
                print(f'pulled: {args.model}')
            return 0
        if args.operation == 'rm':
            failed = False
            for model in args.models:
                try:
                    data = await client.remove(model)
                except ClientError as exc:
                    failed = True
                    print(f'tllama: {model}: {exc}', file=sys.stderr)
                    if as_json:
                        print(json.dumps({'model': model, 'error': str(exc)}, ensure_ascii=False))
                else:
                    if as_json:
                        print(json.dumps({'model': model, 'response': data}, ensure_ascii=False))
                    else:
                        suffix = ' (weights kept)' if data.get('kept_model_file') else ''
                        print(f'deleted definition: {model}{suffix}')
            return 1 if failed else 0
        if args.operation == 'stop':
            data = await client.stop(args.model)
            text = f'unload requested: {args.model}'
        elif args.operation == 'list':
            data = await _wait_for_model_info(client.list_models(), notice=not as_json)
            text = '' if as_json else models_text(data)
        elif args.operation == 'ps':
            data = await client.ps()
            text = '' if as_json else processes_text(data)
        elif args.operation == 'show':
            data = await _wait_for_model_info(client.show(args.model), notice=not as_json)
            text = '' if as_json else show_text(args.model, data, args.field)
        else:
            data = {'client_version': get_version(), 'server_version': await client.version()}
            text = f'tllama client: {data["client_version"]}\ntllama server: {data["server_version"]}'
        if as_json:
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
