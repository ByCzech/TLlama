"""Explicit, versioned conversation saves; loading never contacts the server."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from tllama.client import ClientError, server_url
from tllama.client_config import DEFAULT_SESSION_MAX_FILE_MIB, DEFAULT_IMAGE_MAX_FILE_MIB, validate_size_limit_mib
from tllama.client_state import private_directory, state_directory


class SessionStore:
    def __init__(self, server, model, mode, *, max_file_mib=DEFAULT_SESSION_MAX_FILE_MIB,
                 max_image_file_mib=DEFAULT_IMAGE_MAX_FILE_MIB):
        try:
            self.max_file_mib = validate_size_limit_mib(max_file_mib)
        except ValueError as exc:
            raise ClientError(f'Session file limit {exc}.') from None
        try:
            self.max_image_file_mib = validate_size_limit_mib(max_image_file_mib)
        except ValueError as exc:
            raise ClientError(f'Image file limit {exc}.') from None
        self.max_bytes = self.max_file_mib * 1024 * 1024
        self.server, self.model, self.mode = server_url(server), model, mode
        self.directory = state_directory(server, model, 'sessions')

    def path(self, name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
            raise ClientError('Session name must be 1..64 letters, digits, dots, underscores or hyphens, starting with a letter/digit.')
        return self.directory / (name + '.json')

    def save(self, name, session):
        path = self.path(name)
        data = {'version': 1, 'server': self.server, 'model': self.model, 'mode': self.mode,
                'system': session.system, 'controls': session.controls, 'messages': session.messages,
                'display': {'verbose': session.verbose, 'hide_thinking': session.hide_thinking}}
        self.validate(data)
        temporary = None
        try:
            encoded = (json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            if len(encoded) > self.max_bytes:
                raise ClientError(f'Session exceeds the {self.max_file_mib} MiB file limit.')
            private_directory(self.directory)
            fd, temporary = tempfile.mkstemp(prefix='.session-', dir=self.directory)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
            os.replace(temporary, path)
        except (OSError, ValueError) as exc:
            raise ClientError(f'Cannot save session: {exc}') from None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def validate(self, data):
        from tllama.cli_run import load_json
        def require(condition):
            if not condition:
                raise ValueError("Invalid session")
        # Finite JSON values and duplicate-free decoding are shared with CLI controls.
        try:
            load_json(json.dumps(data, allow_nan=False))
            require(isinstance(data, dict) and set(data) == {
                'version', 'server', 'model', 'mode', 'system', 'controls', 'messages', 'display'})
            require(type(data['version']) is int and data['version'] == 1)
            require((data['server'], data['model'], data['mode']) == (self.server, self.model, self.mode))
            require(data['system'] is None or isinstance(data['system'], str))
            controls = data['controls']
            require(isinstance(controls, dict) and set(controls) <= {'options', 'think', 'format', 'keep_alive'})
            if 'options' in controls:
                require(isinstance(controls['options'], dict))
            if 'think' in controls:
                require(type(controls['think']) is bool or isinstance(controls['think'], str))
            if 'format' in controls:
                require(controls['format'] == 'json' or isinstance(controls['format'], dict))
            if 'keep_alive' in controls:
                require(type(controls['keep_alive']) in {int, float, str})
            messages = data['messages']
            require(isinstance(messages, list))
            for message in messages:
                require(isinstance(message, dict) and set(message) <= {'role', 'content', 'thinking', 'tool_calls', 'images'})
                require(message.get('role') in {'user', 'assistant'} and isinstance(message.get('content'), str))
                if 'images' in message:
                    from tllama.cli_images import validate_image
                    require(message['role'] == 'user' and isinstance(message['images'], list) and bool(message['images']))
                    for image in message['images']:
                        try:
                            validate_image(image, max_file_mib=self.max_image_file_mib)
                        except ValueError as exc:
                            raise ClientError(f'Invalid session image: {exc}') from None
                if 'thinking' in message:
                    require(isinstance(message['thinking'], str))
                if 'tool_calls' in message:
                    require(isinstance(message['tool_calls'], list) and all(isinstance(t, dict) for t in message['tool_calls']))
            display = data['display']
            require(isinstance(display, dict) and set(display) == {'verbose', 'hide_thinking'})
            require(all(type(v) is bool for v in display.values()))
            if self.mode != 'completion':
                require(not messages and data['system'] is None and not display['hide_thinking'])
                require(not (set(controls) & {'think', 'format'}))
            if self.mode == 'decision':
                require(not controls)
        except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
            raise ClientError('Invalid session format, server/model/mode mismatch or unsupported version.') from None

    def load(self, name, session):
        from tllama.cli_run import load_json
        try:
            fd = os.open(self.path(name), os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ClientError('Session must be a regular file.')
                if info.st_size > self.max_bytes:
                    raise ClientError(f'Session exceeds the {self.max_file_mib} MiB file limit.')
                raw = stream.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                raise ClientError(f'Session exceeds the {self.max_file_mib} MiB file limit.')
            data = load_json(raw.decode('utf-8'))
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            raise ClientError(f'Cannot load session: {exc}') from None
        self.validate(data)
        if any(message.get('images') for message in data['messages']) and not session.vision:
            raise ClientError('Saved images require a completion model advertising vision; check its projector configuration.')
        # Validate everything before replacing any live conversation state.
        session.system = data['system']
        session.controls = deepcopy(data['controls'])
        session.messages = deepcopy(data['messages'])
        session.pending_images.clear()
        session.verbose, session.hide_thinking = data['display']['verbose'], data['display']['hide_thinking']

    def names(self):
        try:
            return sorted(p.stem for p in self.directory.glob('*.json') if not p.is_symlink() and p.is_file())
        except OSError as exc:
            raise ClientError(f'Cannot list sessions: {exc}') from None

    def forget(self, name):
        try:
            self.path(name).unlink()
        except OSError as exc:
            raise ClientError(f'Cannot remove session: {exc}') from None
