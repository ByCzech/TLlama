"""Client-only TOML defaults; importing this module never loads the daemon."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

import tomlkit

from tllama.client import ClientConfigError, DEFAULT_HOST, server_url


SETTINGS = {'client.host': ('host', str), 'client.timeout': ('timeout', float),
            'display.decimal_units': ('decimal_units', bool),
            'display.hide_thinking': ('hide_thinking', bool),
            'display.verbose': ('verbose', bool),
            'history.enabled': ('history_enabled', bool)}


@dataclass(frozen=True)
class ClientConfig:
    host: str = DEFAULT_HOST
    timeout: float | None = None
    decimal_units: bool = False
    hide_thinking: bool = False
    verbose: bool = False
    history_enabled: bool = True


def default_config_paths(environ):
    user = Path(environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    if os.name == 'nt':
        user = Path(environ.get('APPDATA') or user)
    return Path('/etc/tllama/client.toml'), user / 'tllama' / 'client.toml'


def _validate(key, value, origin):
    expected = SETTINGS[key][1]
    if expected is bool and type(value) is not bool:
        raise ClientConfigError(f'{origin}: {key} must be true or false.')
    if expected is str:
        if not isinstance(value, str) or not value.strip() or '\0' in value:
            raise ClientConfigError(f'{origin}: {key} must be a nonempty string.')
        try:
            server_url(value)
        except ClientConfigError as exc:
            raise ClientConfigError(f'{origin}: {key}: {exc}') from None
        return value.strip()
    if expected is float:
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ClientConfigError(f'{origin}: {key} must be a finite number greater than zero.')
    return value


def load_client_config(config_path=None, overrides=None, *, environ=None):
    """Layer system then user settings, or read only an explicit file.

    TLLAMA_HOST and explicit CLI values override file defaults. The HTTP library
    does not discover files: this loader is called only by the CLI entry point.
    """
    env = os.environ if environ is None else environ
    overrides = dict(overrides or {})
    values = {}
    explicit = config_path is not None
    paths = (Path(config_path).expanduser(),) if explicit else default_config_paths(env)
    sections = {key.split('.')[0] for key in SETTINGS}
    for path in paths:
        try:
            text = path.read_text(encoding='utf-8')
        except FileNotFoundError as exc:
            if not explicit:
                continue
            raise ClientConfigError(f'Cannot read client config {path}: {exc}') from None
        except (OSError, UnicodeError) as exc:
            raise ClientConfigError(f'Cannot read client config {path}: {exc}') from None
        try:
            document = tomlkit.parse(text).unwrap()
        except ValueError as exc:
            raise ClientConfigError(f'Invalid client config {path}: {exc}') from None
        for section, table in document.items():
            if section not in sections or not isinstance(table, dict):
                raise ClientConfigError(f'{path}: unknown table or invalid section {section!r}.')
            for name, value in table.items():
                key = f'{section}.{name}'
                if key not in SETTINGS:
                    raise ClientConfigError(f'{path}: unknown client setting {key!r}.')
                field = SETTINGS[key][0]
                values[field] = _validate(key, value, path)
    if 'host' not in overrides and env.get('TLLAMA_HOST', '').strip():
        values['host'] = _validate('client.host', env['TLLAMA_HOST'], 'TLLAMA_HOST')
    for key, (field, _) in SETTINGS.items():
        if field in overrides:
            values[field] = _validate(key, overrides.pop(field), 'CLI')
    if overrides:
        raise ClientConfigError(f'Unknown client overrides: {", ".join(sorted(overrides))}.')
    return ClientConfig(**values)
