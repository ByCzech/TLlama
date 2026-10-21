"""Model-mode dispatch and conversation ownership, independent of terminal UI."""

from contextlib import aclosing
from dataclasses import dataclass, field
import argparse
import json
import math
import sys
import time

from tllama.client import ClientError
from tllama.cli_status import WaitStatus
from tllama.cli_reply import ReplyText
from tllama.cli_metrics import statistics


def load_json(text):
    def finite(value):
        raise ValueError('JSON numbers must be finite')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('JSON numbers must be finite')
        return result
    return json.loads(text, parse_constant=finite, parse_float=number, object_pairs_hook=unique)


def option(text):
    name, sep, value = text.partition('=')
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError('use NAME=JSON, e.g. temperature=0.7')
    try:
        return name, load_json(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def response_format(text):
    if text == 'json':
        return text
    try:
        value = load_json(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('format must be json or a JSON schema object')


def think_value(text):
    return {'true': True, 'false': False}.get(text.lower(), text)


def add_run_options(parser):
    parser.add_argument('model')
    parser.add_argument('prompt', nargs='*', help='Prompt text (stdin is prepended when piped).')
    parser.add_argument('--system', help='Session system prompt; does not change the model definition.')
    parser.add_argument('--option', action='append', type=option, default=[], metavar='NAME=JSON')
    parser.add_argument('--think', type=think_value, help='true, false, or a server-supported reasoning level.')
    parser.add_argument('--format', type=response_format, help='json or a JSON schema object.')
    parser.add_argument('--keep-alive', help='Server model residence duration, e.g. 5m or 0.')
    parser.add_argument('--hide-thinking', action='store_true', help='Hide reasoning text; does not disable thinking.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print measured timings and token statistics to stderr.')


@dataclass
class RunSession:
    model: str
    mode: str
    controls: dict = field(default_factory=dict)
    system: str | None = None
    messages: list = field(default_factory=list)
    as_json: bool = False
    hide_thinking: bool = False
    verbose: bool = False

    @classmethod
    async def create(cls, client, args):
        from tllama.cli import _wait_for_model_info
        info = await _wait_for_model_info(client.show(args.model), notice=not getattr(args, 'json', False))
        capabilities = info.get('capabilities')
        if not isinstance(capabilities, list) or any(not isinstance(c, str) for c in capabilities):
            raise ClientError('Model information is missing its capabilities array.')
        modes = set(capabilities) & {'completion', 'embedding', 'decision'}
        if len(modes) != 1:
            raise ClientError('Model must advertise exactly one supported mode: completion, embedding, or decision.')
        controls = {key: value for key, value in (
            ('options', dict(args.option) or None), ('think', args.think),
            ('format', args.format), ('keep_alive', args.keep_alive)) if value is not None}
        mode = modes.pop()
        if mode != 'completion' and (args.system is not None or args.hide_thinking
                                      or set(controls) & {'think', 'format'}):
            raise ClientError('System, thinking and format controls require a completion model.')
        if mode == 'decision' and controls:
            raise ClientError('Decision models do not accept generation or residence controls.')
        return cls(args.model, mode, controls, args.system, as_json=getattr(args, 'json', False),
                   hide_thinking=args.hide_thinking, verbose=getattr(args, 'verbose', False))

    async def submit(self, client, text):
        started = time.perf_counter()
        if self.mode == 'embedding':
            async with WaitStatus('Waiting for response…', enabled=not self.as_json):
                data = await client.embed(self.model, text, **self.controls)
            print(json.dumps(data, ensure_ascii=False), flush=True)
            self._statistics(data, started)
            return
        if self.mode == 'decision':
            try:
                payload = load_json(text)
            except ValueError as exc:
                raise ClientError(f'Decision input must be JSON: {exc}') from None
            if not isinstance(payload, dict) or set(payload) != {'state', 'questions'}:
                raise ClientError('Decision input must be an object with exactly state and questions.')
            async with WaitStatus('Waiting for response…', enabled=not self.as_json):
                data = await client.decision(self.model, payload['state'], payload['questions'])
            print(json.dumps(data, ensure_ascii=False), flush=True)
            self._statistics(data, started)
            return
        request = ([{'role': 'system', 'content': self.system}] if self.system is not None else [])
        request += self.messages + [{'role': 'user', 'content': text}]
        reply = {'role': 'assistant', 'content': ''}
        renderer = ReplyText()
        terminal = None
        first_text = None
        try:
            async with WaitStatus('Waiting for response…', enabled=not self.as_json) as status, \
                    aclosing(client.chat(self.model, request, **self.controls)) as stream:
                async for data in stream:
                    message = data['message']
                    if first_text is None and (message.get('content') or message.get('thinking')):
                        first_text = time.perf_counter() - started
                    if data['done']:
                        terminal = data
                    for key in ('content', 'thinking'):
                        if key in message:
                            reply[key] = reply.get(key, '') + message[key]
                    if message.get('tool_calls'):
                        reply.setdefault('tool_calls', []).extend(message['tool_calls'])
                    if self.as_json:
                        print(json.dumps(data, ensure_ascii=False), flush=True)
                    else:
                        if (message.get('content') or message.get('tool_calls')
                                or message.get('thinking') and not self.hide_thinking or data['done']):
                            status.stop()
                        if message.get('thinking') and not self.hide_thinking:
                            renderer.thinking(message['thinking'])
                        renderer.answer(message.get('content', ''))
                        renderer.tools(message.get('tool_calls'))
        finally:
            renderer.close()
        # Commit only a completed turn. Failed/cancelled replies must not
        # silently become context for the next interactive request.
        self.messages.extend([{'role': 'user', 'content': text}, reply])
        self._statistics(terminal or {}, started, first_text=first_text)

    def _statistics(self, data, started, *, first_text=None):
        if self.verbose:
            print(statistics(data, elapsed=time.perf_counter() - started, first_text=first_text),
                  file=sys.stderr, flush=True)

    async def preload(self, client):
        started = time.perf_counter()
        async with WaitStatus('Loading model…', enabled=not self.as_json):
            loaded = await client.preload(self.model, options=self.controls.get('options'),
                                           keep_alive=self.controls.get('keep_alive'))
        if not loaded and not self.as_json:
            print('Preload not retained (keep_alive=0); the model will load on request.', file=sys.stderr)
        if self.verbose:
            print(f'Preload request: {time.perf_counter() - started:.3f} s'
                  + ('' if loaded else ' (not retained)'), file=sys.stderr, flush=True)


async def run(client, args):
    text = ' '.join(args.prompt) if args.prompt else None
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        text = piped + ('\n\n' + text if text is not None else '')
    if text is None:
        from tllama.cli_repl import repl
        session = await RunSession.create(client, args)
        await session.preload(client)
        return await repl(client, session)
    session = await RunSession.create(client, args)
    await session.submit(client, text)
    return 0
