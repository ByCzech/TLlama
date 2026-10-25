"""Resolve daemon defaults, server TOML, environment and CLI before serving."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping

import tomlkit

from tllama.config import (AppConfig, BackendConfig, ConfigError, DEFAULT_MODELS_DIR,
                           _parse_host_port, resolve_cors_origins)
from tllama.backend_policy import DEFAULT_BACKEND_PRIORITY, backend_priority
from tllama.worker_config import DEFAULT_EXECUTABLE
from tllama.native_params import RUNTIME_TYPES, SAMPLING_TYPES, runtime_problem, sampling_problem, kv_type_name
from tllama.helpers.runtime_params import coerce_runtime_overrides
from tllama.helpers.sampling_params import coerce_sampling_overrides
from tllama.helpers.common import normalize_keep_alive


# One canonical key per setting; aliases are expanded before layers are merged.
# kind, default, environment variable, BackendConfig field (if applicable).
SETTINGS = {
    'server.host': ('str', '127.0.0.1', 'TLLAMA_HOST', None),
    'server.port': ('int', 54800, 'TLLAMA_PORT', None),
    'server.reload': ('bool', False, 'TLLAMA_APP_RELOAD', None),
    'server.log_level': ('str', 'info', 'TLLAMA_LOG_LEVEL', None),
    'server.origins': ('names', [], 'TLLAMA_ORIGINS', None),
    'storage.models_dir': ('str', DEFAULT_MODELS_DIR, 'TLLAMA_MODELS', 'models_dir'),
    'lifecycle.keep_alive': ('duration', '5m', 'TLLAMA_KEEP_ALIVE', 'keep_alive'),
    'lifecycle.max_loaded_models': ('int', 1, 'TLLAMA_MAX_LOADED_MODELS', 'max_loaded_models'),
    'lifecycle.janitor_interval': ('float', 10.0, 'TLLAMA_JANITOR_INTERVAL', 'janitor_interval_seconds'),
    'cache.metadata_ttl': ('float', 300.0, 'TLLAMA_METADATA_CACHE_TTL', 'metadata_cache_ttl_seconds'),
    'engine.executable': ('str', str(DEFAULT_EXECUTABLE), 'TLLAMA_RUNTIME_EXECUTABLE', 'runtime_executable'),
    'engine.version_policy': ('str', 'warn', 'TLLAMA_RUNTIME_VERSION_POLICY', 'runtime_version_policy'),
    'engine.backend_priority': ('names', list(DEFAULT_BACKEND_PRIORITY), 'TLLAMA_BACKEND_PRIORITY', 'backend_priority'),
    'engine.ignore_unsupported_runner': ('bool', False, 'TLLAMA_IGNORE_UNSUPPORTED_RUNNER', 'ignore_unsupported_runner'),
    'telemetry.memory_source': ('str', 'auto', 'TLLAMA_MEMORY_SOURCE', 'memory_source'),
    'telemetry.memory_compare': ('bool', False, 'TLLAMA_MEMORY_COMPARE', 'memory_compare'),
    'runtime.n_ctx': ('int', 0, 'TLLAMA_CONTEXT_LENGTH', 'context_length'),
    'runtime.type_kv': ('str', None, 'TLLAMA_KV_CACHE_TYPE', 'kv_cache_type'),
    'runtime.type_k': ('str', None, 'TLLAMA_K_CACHE_TYPE', 'k_cache_type'),
    'runtime.type_v': ('str', None, 'TLLAMA_V_CACHE_TYPE', 'v_cache_type'),
}
ENV_KEYS = {spec[2]: key for key, spec in SETTINGS.items()}
_BRIDGE = '_TLLAMA_DAEMON_OPTIONS'
_ACTIVE = ContextVar('tllama_daemon_config', default=None)


def _convert(key, value, *, text=False):
    if key.startswith('runtime.') and key not in SETTINGS:
        name = key[8:]
        if text and isinstance(value, str):
            return coerce_runtime_overrides({name: value})[name]
        problem = runtime_problem(name, value)
        if problem:
            raise ConfigError(problem)
        return value
    if key.startswith('sampling.'):
        name = key[9:]
        if name in {'stop', 'logit_bias'}:
            raise ConfigError(f'{name} is a per-model [sampling] setting, not a daemon default')
        if text and isinstance(value, str):
            return coerce_sampling_overrides({name: value})[name]
        problem = sampling_problem(name, value)
        if problem:
            raise ConfigError(problem)
        return value
    kind = SETTINGS[key][0]
    if text:
        if kind == 'bool':
            lowered = str(value).strip().lower()
            if lowered not in {'1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'}:
                raise ConfigError('expected true/false or 1/0, yes/no, on/off')
            value = lowered in {'1', 'true', 'yes', 'on'}
        elif kind == 'int':
            value = int(value)
        elif kind == 'float':
            value = float(value)
        elif kind == 'names':
            value = [part.strip() for part in value.split(',') if part.strip()]
    good = {'str': isinstance(value, str), 'int': type(value) is int,
            'float': type(value) in (int, float), 'bool': type(value) is bool,
            'duration': type(value) in (str, int, float),
            'names': isinstance(value, list) and all(isinstance(item, str) for item in value)}[kind]
    if not good or isinstance(value, str) and '\0' in value:
        raise ConfigError(f'expected {kind}')
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ConfigError('expected a finite number')
    return value


def _known(key):
    return key in SETTINGS or (key.startswith('runtime.') and key[8:] in RUNTIME_TYPES
                              and key[8:] != 'model_path') or (
                                  key.startswith('sampling.') and key[9:] in SAMPLING_TYPES)


def default_config_paths(environ):
    home = Path.home()
    user = Path(environ.get('XDG_CONFIG_HOME') or home / '.config')
    if os.name == 'nt':
        user = Path(environ.get('APPDATA') or user)
    return (user / 'tllama' / 'server.toml', Path('/etc/tllama/server.toml'))


@dataclass(frozen=True)
class DaemonConfig:
    app: AppConfig
    backend: BackendConfig
    cors_origins: tuple[str, ...]
    values: Mapping
    sources: Mapping[str, str]
    config_path: Path | None

    def report(self):
        # Only schema-owned settings are reported, never the process environment.
        return {'config_file': str(self.config_path) if self.config_path else None,
                'settings': {key: {'value': value, 'source': self.sources[key]}
                             for key, value in sorted(self.values.items())}}


def load_daemon_config(config_path=None, overrides=None, *, environ=None, discover=True):
    env = os.environ if environ is None else environ
    overrides = dict(overrides or {})
    selected = '' if config_path is False else config_path or env.get('TLLAMA_CONFIG', '').strip()
    path = Path(selected).expanduser().absolute() if selected else None
    if path is None and discover:
        path = next((p.absolute() for p in default_config_paths(env) if p.exists()), None)
    values = {key: spec[1] for key, spec in SETTINGS.items()}
    sources = {key: 'default' for key in values}

    def put(key, value, origin, text=False):
        if not _known(key):
            raise ConfigError(f'{origin}: unknown daemon setting {key!r}')
        try:
            values[key] = _convert(key, value, text=text)
            sources[key] = origin
        except (ValueError, TypeError, OverflowError) as exc:
            raise ConfigError(f'{origin} ({key}): {exc}') from None

    def layer(items, origin, text=False):
        # host:port and the dedicated port setting share the same precedence.
        # A dedicated port at this layer wins over a port inside host.
        items = dict(items)
        if 'runtime.type_kv' in items:
            for side in ('runtime.type_k', 'runtime.type_v'):
                items.setdefault(side, items['runtime.type_kv'])
        if 'server.host' in items:
            raw = items.pop('server.host')
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigError(f'{origin}: server.host must be a nonempty string')
            host, port = _parse_host_port(origin, raw, values['server.host'], values['server.port'])
            if any(c.isspace() for c in host) or any(c in host for c in '@/\\?#'):
                raise ConfigError(f'{origin}: server.host must be a listener address without credentials')
            put('server.host', host, origin)
            # A host-only override retains the lower layer's port and source.
            from tllama.config import _strip_scheme_and_path, _split_host_port
            _, given_port = _split_host_port(origin, raw, _strip_scheme_and_path(origin, raw, raw.strip()))
            if given_port is not None:
                put('server.port', port, origin)
        for key, value in items.items():
            put(key, value, origin, text=text)

    if path is not None:
        try:
            document = tomlkit.parse(path.read_text(encoding='utf-8')).unwrap()
        except (OSError, ValueError) as exc:
            raise ConfigError(f'Cannot read daemon config {path}: {exc}') from None
        flat = {}
        sections = {key.split('.')[0] for key in SETTINGS} | {'sampling'}
        for section, table in document.items():
            if section not in sections or not isinstance(table, dict):
                raise ConfigError(f'{path}: unknown table or invalid section {section!r}')
            for name, value in table.items():
                flat[f'{section}.{name}'] = value
        layer(flat, f'TOML {path}')

    # Convert each environment value independently for useful variable names,
    # but apply them together for deterministic host/port alias handling.
    environment = {}
    for name, raw in env.items():
        if not name.startswith('TLLAMA_') or name == 'TLLAMA_CONFIG':
            continue
        if not raw.strip() and name != 'TLLAMA_BACKEND_PRIORITY':
            continue
        key = ENV_KEYS.get(name)
        if key is None and name.startswith('TLLAMA_RUNTIME_'):
            runtime_name = name[len('TLLAMA_RUNTIME_'):].lower()
            if runtime_name in {'n_ctx', 'type_k', 'type_v', 'type_kv'}:
                raise ConfigError(f'{name}: use dedicated context/K/V environment variables')
            key = 'runtime.' + runtime_name
        elif key is None and name.startswith('TLLAMA_SAMPLING_'):
            key = 'sampling.' + name[len('TLLAMA_SAMPLING_'):].lower()
        if key is None:
            raise ConfigError(f'{name}: unknown daemon environment variable')
        environment[key] = (raw, name)
    host_item = environment.pop('server.host', None)
    if host_item:
        layer({'server.host': host_item[0]}, host_item[1], text=True)
    if 'runtime.type_kv' in environment:
        for side in ('runtime.type_k', 'runtime.type_v'):
            environment.setdefault(side, environment['runtime.type_kv'])
    for key, (raw, name) in environment.items():
        put(key, raw, name, text=True)
    layer(overrides, 'CLI', text=True)

    def invalid(key, message):
        raise ConfigError(f'{sources[key]} ({key}): {message}; got {values[key]!r}')

    for key, minimum, inclusive in [('server.port', 1, True), ('runtime.n_ctx', 0, True),
                                   ('lifecycle.max_loaded_models', 1, True),
                                   ('lifecycle.janitor_interval', 0, False), ('cache.metadata_ttl', 0, True)]:
        value = values[key]
        if value < minimum or not inclusive and value == minimum:
            invalid(key, f'expected a value {">=" if inclusive else ">"} {minimum}')
    if values['server.port'] > 65535:
        invalid('server.port', 'expected a port from 1 to 65535')
    for key in ('engine.executable', 'storage.models_dir'):
        value = values[key]
        if not value.strip():
            invalid(key, 'expected a nonempty path')
        path_value = Path(value).expanduser()
        if key == 'engine.executable' and not path_value.is_absolute():
            invalid(key, 'expected an absolute path')
        values[key] = str(path_value.absolute())
    for key in ('engine.version_policy', 'telemetry.memory_source', 'server.log_level'):
        values[key] = values[key].strip().lower()
    for key, allowed in [('engine.version_policy', {'warn', 'strict'}),
                         ('telemetry.memory_source', {'auto', 'native', 'log'})]:
        if values[key] not in allowed:
            invalid(key, f'expected one of {", ".join(sorted(allowed))}')
    if values['server.log_level'] not in {'critical', 'error', 'warning', 'info', 'debug', 'trace'}:
        invalid('server.log_level', 'expected critical/error/warning/info/debug/trace')
    try:
        duration = values['lifecycle.keep_alive']
        if isinstance(duration, str):
            raw_duration = duration.strip().lower()
            number = raw_duration[:-1] if raw_duration.endswith(('s', 'm', 'h')) else raw_duration
            if not math.isfinite(float(number)):
                raise ValueError('nonfinite duration')
        normalize_keep_alive(values['lifecycle.keep_alive'])
    except (ValueError, OverflowError):
        invalid('lifecycle.keep_alive', 'expected seconds or a duration with s/m/h suffix')
    for key in ('runtime.type_kv', 'runtime.type_k', 'runtime.type_v'):
        if values[key] is not None:
            try:
                values[key] = kv_type_name(values[key].strip())
            except ValueError as exc:
                invalid(key, str(exc))
    values['engine.backend_priority'] = list(backend_priority(values['engine.backend_priority']))
    runtime = {key[8:]: value for key, value in values.items() if key.startswith('runtime.') and key not in SETTINGS}
    sampling = {key[9:]: value for key, value in values.items() if key.startswith('sampling.')}
    backend_values = {spec[3]: values[key] for key, spec in SETTINGS.items() if spec[3]}
    try:
        backend = BackendConfig(**backend_values, runtime_overrides=runtime, sampling_overrides=sampling)
        origins = resolve_cors_origins(','.join(values['server.origins']))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    app = AppConfig(host=values['server.host'], port=values['server.port'],
                    reload=values['server.reload'],
                    log_level=values['server.log_level'])
    return DaemonConfig(app, backend, origins, values, sources, path)


def current_daemon_config():
    active = _ACTIVE.get()
    if active is not None:
        return active
    raw = os.environ.get(_BRIDGE)
    if raw:
        try:
            options = json.loads(raw)
            return load_daemon_config(**options)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f'Invalid daemon reload configuration: {exc}') from None
    return load_daemon_config()


@contextmanager
def daemon_configuration(config, overrides):
    """Keep one snapshot in-process and replay CLI sources in reload children."""
    token = _ACTIVE.set(config)
    previous = os.environ.get(_BRIDGE)
    os.environ[_BRIDGE] = json.dumps({'config_path': str(config.config_path) if config.config_path else False,
                                    'overrides': overrides, 'discover': False})
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        if previous is None:
            os.environ.pop(_BRIDGE, None)
        else:
            os.environ[_BRIDGE] = previous
