"""Small async shell: terminal editing stays separate from conversation state."""

import asyncio
import argparse
import json
import signal
import sys

from tllama.cli_run import load_json, response_format, think_value
from tllama.client import ClientError, ClientConfigError

HELP = """Enter a prompt. Alt+Enter inserts a newline; Enter submits. Pasted multiline text is one prompt.
Ctrl+C cancels input or the active request; Ctrl+D exits at an empty prompt.
/help, /?                 Show this help
/bye, /exit               End this session
/clear                    Clear conversation messages; keep session settings
/save NAME                Save conversation and settings locally (overwrites NAME)
/load NAME                Replace conversation and settings from a local save
/sessions                 List saves for this server/model
/forget NAME              Remove one local save
/show model|parameters|system
/set parameter NAME JSON  Set a request option (e.g. temperature 0.7)
/set system TEXT          Replace the session system prompt and clear conversation
/set think VALUE          true, false, or a server-supported level
/set format VALUE         json, a schema object, or off
/set keep_alive VALUE     Server model residence duration (e.g. 5m or 0)
/set verbose, /set quiet  Enable/disable measured request statistics
Use //text to send a literal /text prompt. Input history is private per server/model;
--no-history disables persistence and --clear-history clears saved inputs on startup.
"""


def terminal(*, history=None):
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import NestedCompleter
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.output.defaults import create_output
    except ImportError:
        raise ClientConfigError('Interactive run requires prompt_toolkit>=3.0.51; refresh the TLlama installation.') from None
    keys = KeyBindings()
    @keys.add('escape', 'enter')
    def newline(event):
        event.current_buffer.insert_text('\n')
    completer = NestedCompleter.from_nested_dict({
        '/help': None, '/?': None, '/bye': None, '/exit': None, '/clear': None,
        '/save': None, '/load': None, '/sessions': None, '/forget': None,
        '/show': {'model': None, 'parameters': None, 'system': None},
        '/set': {'parameter': None, 'system': None, 'think': None, 'format': None, 'keep_alive': None,
                 'verbose': None, 'quiet': None},
    })
    return PromptSession(history=history if history is not None else InMemoryHistory(), completer=completer, key_bindings=keys,
                         complete_while_typing=False, prompt_continuation='... ',
                         output=create_output(stdout=sys.stderr))


async def command(client, session, text):
    """Return False only for exit; command parsing never touches prompt text."""
    name, *rest = text.split(maxsplit=1)
    argument = rest[0] if rest else ''
    if name in {'/bye', '/exit'}:
        if argument:
            raise ClientError(f'{name} does not accept arguments.')
        return False
    if name in {'/help', '/?'} and not argument:
        print(HELP, end='', file=sys.stderr)
    elif name == '/clear' and not argument:
        session.messages.clear()
        print('Conversation cleared.', file=sys.stderr)
    elif name in {'/save', '/load', '/forget', '/sessions'}:
        from tllama.cli_sessions import SessionStore
        store = SessionStore(getattr(client, 'url', None), session.model, session.mode)
        if name == '/sessions' and not argument:
            print('\n'.join(store.names()) or 'No saved sessions.', file=sys.stderr)
        elif name == '/save' and argument:
            store.save(argument, session)
            print(f'Session saved: {argument}', file=sys.stderr)
        elif name == '/load' and argument:
            store.load(argument, session)
            print(f'Session loaded: {argument}', file=sys.stderr)
        elif name == '/forget' and argument:
            store.forget(argument)
            print(f'Session removed: {argument}', file=sys.stderr)
        else:
            raise ClientError('Use /save NAME, /load NAME, /forget NAME or /sessions.')
    elif name == '/show':
        if argument == 'model':
            from tllama.cli import _wait_for_model_info
            data = await _wait_for_model_info(client.show(session.model), notice=True)
        elif argument == 'parameters':
            data = session.controls
        elif argument == 'system':
            data = session.system
        else:
            raise ClientError('Use /show model, /show parameters, or /show system.')
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
    elif name == '/set':
        if argument in {'verbose', 'quiet'}:
            session.verbose = argument == 'verbose'
            print('Verbose mode ' + ('enabled.' if session.verbose else 'disabled.'), file=sys.stderr)
            return True
        if not argument:
            raise ClientError('Use /set SETTING VALUE; see /help.')
        setting, *values = argument.split(maxsplit=1)
        if not values:
            raise ClientError('Use /set SETTING VALUE; see /help.')
        value = values[0]
        if session.mode != 'completion' and setting not in {'parameter', 'keep_alive'}:
            raise ClientError('This setting requires a completion model.')
        if session.mode == 'decision':
            raise ClientError('Decision models do not accept generation or residence controls.')
        if setting == 'parameter':
            key, *parts = value.split(maxsplit=1)
            if not parts:
                raise ClientError('Use /set parameter NAME JSON.')
            session.controls.setdefault('options', {})[key] = load_json(parts[0])
        elif setting == 'system':
            session.system = value
            session.messages.clear()
        elif setting == 'think':
            session.controls['think'] = think_value(value)
        elif setting == 'format':
            if value == 'off':
                session.controls.pop('format', None)
            else:
                session.controls['format'] = response_format(value)
        elif setting == 'keep_alive':
            session.controls['keep_alive'] = value
        else:
            raise ClientError('Unknown session setting; see /help.')
        print('Session setting updated.', file=sys.stderr)
    else:
        raise ClientError('Unknown command or arguments; see /help. Use // to send a literal slash.')
    return True


async def submit(client, session, text):
    await _interruptible(session.submit(client, text))


async def _interruptible(awaitable):
    """SIGINT cancels the owned request without cancelling the REPL task."""
    task = asyncio.create_task(awaitable)
    interrupted = False
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        nonlocal interrupted
        # Repeated Ctrl+C must not interrupt the first cancellation's
        # response cleanup before all HTTP iterator layers unwind.
        if not interrupted:
            interrupted = True
            loop.call_soon_threadsafe(task.cancel)
    installed = False
    try:
        signal.signal(signal.SIGINT, interrupt)
        installed = True
        return await task
    except asyncio.CancelledError:
        if not interrupted:
            raise
        print('Request cancelled.', file=sys.stderr)
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def repl(client, session, *, reader=None):
    if reader is None:
        from tllama.cli_history import input_history
        reader = terminal(history=input_history(getattr(client, 'url', None), session.model,
                          enabled=session.history_enabled, clear=session.clear_history))
    print(f'TLlama: {session.model} ({session.mode}). /help for commands.', file=sys.stderr)
    while True:
        try:
            text = await reader.prompt_async('>>> ')
        except EOFError:
            return 0
        except KeyboardInterrupt:
            continue
        if not text.strip():
            continue
        try:
            if text.startswith('//'):
                await submit(client, session, text[1:])
            elif text.startswith('/'):
                if await _interruptible(command(client, session, text)) is False:
                    return 0
            else:
                await submit(client, session, text)
        except (ClientError, ValueError, argparse.ArgumentTypeError) as exc:
            print(f'tllama: {exc}', file=sys.stderr)
            session._notify('error')
