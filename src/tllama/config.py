import re

from dataclasses import dataclass, field
from pathlib import Path

from typing import Mapping

from tllama.worker_config import DEFAULT_EXECUTABLE
from tllama.backend_policy import DEFAULT_BACKEND_PRIORITY, backend_priority

DEFAULT_MODELS_DIR = "/var/lib/tllama/models"


class ConfigError(ValueError):
    """An invalid configuration source, reported before serving requests."""


def _reject(name: str, value: str, expected: str) -> "ConfigError":
    return ConfigError(f"{name}={value!r} is not valid: expected {expected}.")


_SUPPORTED_SCHEMES = ("http://", "https://")


def _strip_scheme_and_path(name: str, value: str, raw: str) -> str:
    """Accept the scheme://host:port spelling OLLAMA_HOST accepts.

    A bare host:port is the usual form, but a URL is a documented way to
    write OLLAMA_HOST and someone moving across writes what they had.
    Without this the scheme stayed glued to the host and uvicorn was asked
    to bind to "http://0.0.0.0", which fails at startup with nothing
    pointing at the variable that caused it.

    An unrecognised scheme is refused rather than stripped. Anything
    before "://" could be dropped blindly, but ftp:// is a mistake, and
    silently accepting it would recreate exactly the swallowing this
    module has just stopped doing.
    """
    if "://" not in raw:
        return raw

    for scheme in _SUPPORTED_SCHEMES:
        if raw.lower().startswith(scheme):
            raw = raw[len(scheme):]
            break
    else:
        scheme_given = raw.split("://", 1)[0]
        raise _reject(
            name,
            value,
            f"no scheme or one of http:// and https://, not {scheme_given}://",
        )

    # A URL may carry a path; there is nothing here that could use one.
    return raw.split("/", 1)[0]


def _split_host_port(name: str, value: str, raw: str) -> tuple[str, str | None]:
    """Separate host from port, for IPv6 as well as IPv4 and names.

    Splitting on the last colon is right for "host:port" and wrong for
    every bare IPv6 address, which is full of them: "::1" came out as the
    host ":" on port 1, quietly, and "[::]" failed to parse at all
    because the last colon leaves "]" as the port. A bracketed address
    delimits itself, so the port is whatever follows the closing bracket;
    an unbracketed address with more than one colon cannot carry a port
    and is taken whole.
    """
    if raw.startswith("["):
        closing = raw.find("]")
        if closing == -1:
            raise _reject(name, value, "a closing ] after a bracketed IPv6 address")

        # The brackets are how the written form keeps the address apart
        # from the port; they are not part of the address. getaddrinfo
        # refuses "[::1]" and accepts "::1", so what is returned here has
        # to be the latter.
        host = raw[1:closing]
        rest = raw[closing + 1:]

        if rest == "":
            return host, None
        if rest.startswith(":"):
            return host, rest[1:]
        raise _reject(name, value, "either [ADDRESS] or [ADDRESS]:PORT")

    if raw.count(":") > 1:
        # An unbracketed IPv6 address. Written this way it has no room
        # for a port, so all of it is the host.
        return raw, None

    if ":" not in raw:
        return raw, None

    host, port_str = raw.rsplit(":", 1)
    return host, port_str


# Built-in browser origins; configured origins extend this list.
BUILTIN_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost",
    "https://localhost",
    "http://localhost:*",
    "https://localhost:*",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://127.0.0.1:*",
    "https://127.0.0.1:*",
    "http://[::1]",
    "https://[::1]",
    "http://[::1]:*",
    "https://[::1]:*",
    "vscode-webview://*",
    "vscode-file://*",
)

# scheme://host, and nothing after it. A browser never puts a path, query
# or fragment in an Origin header, so anything past the authority is a
# mistake in the configuration rather than a form this could ever match.
_ORIGIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^/?#]+\Z")


def resolve_cors_origins(value: str = "") -> tuple[str, ...]:
    """The origins a browser may reach TLlama from.

    TLLAMA_ORIGINS adds to the built-in list rather than replacing it, so
    reaching a local server from a local page keeps working whatever else
    is configured.

    A malformed entry stops the server. An origin that a browser can never
    send is a setting that silently does nothing, and a service that comes
    up in a state its operator did not ask for is worse than one that
    refuses to come up at all.
    """
    configured = []

    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            # 'a,,b' and a trailing comma are formatting, not a mistake
            # worth refusing to start over.
            continue

        if not _ORIGIN_RE.fullmatch(entry):
            raise _reject(
                "TLLAMA_ORIGINS",
                entry,
                "a comma separated list of origins of the form "
                "scheme://host[:port], with no path (a '*' may stand in "
                "for any part)",
            )

        configured.append(entry)

    seen = set()
    origins = []
    for origin in (*BUILTIN_CORS_ORIGINS, *configured):
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)

    return tuple(origins)


def _parse_host_port(
    name: str, value: str, default_host: str, default_port: int
) -> tuple[str, int]:
    if not value or value.strip() == "":
        return default_host, default_port

    raw = _strip_scheme_and_path(name, value, value.strip())

    host, port_str = _split_host_port(name, value, raw)

    if port_str is None:
        port = default_port
    else:
        try:
            port = int(port_str)
        except ValueError:
            raise _reject(name, value, "HOST:PORT with a numeric port") from None

        # A port outside this range never reaches a working server, and
        # each end of it fails differently. Above 65535 or below zero,
        # bind() raises somewhere inside uvicorn, far from the variable
        # that caused it. Zero is worse, because it succeeds: the kernel
        # hands out an arbitrary free port, so the service starts and is
        # listening on an address nobody can predict, including whoever
        # configured it.
        if not 1 <= port <= 65535:
            raise _reject(name, value, "a port from 1 to 65535")

    if not host:
        host = default_host

    return host, port


@dataclass(frozen=True)
class BackendConfig:
    models_dir: str = DEFAULT_MODELS_DIR
    context_length: int = 0
    keep_alive: str | int | float | None = "5m"
    max_loaded_models: int = 1
    janitor_interval_seconds: float = 10.0
    metadata_cache_ttl_seconds: float = 300.0
    # Global values are typed by the shared daemon configuration loader.
    # A model's [runtime] overrides these globals.
    runtime_overrides: Mapping[str, object] = field(default_factory=dict)

    # Explicit typed sampling defaults set server-wide.
    # Empty by default
    # and deliberately so: a global sampling value overrules what a model
    # recommends, for every model at once, so it has to be something a
    # person asked for rather than a number TLlama shipped.
    sampling_overrides: Mapping[str, object] = field(default_factory=dict)

    # kv_cache_type sets both sides; k_cache_type/v_cache_type override it
    # for one side each, mirroring type_kv/type_k/type_v in a .toml
    # [runtime]. Each layer expands its shorthand before the next overrides
    # it; native_config owns the precedence for global and model settings.
    kv_cache_type: str | None = None
    k_cache_type: str | None = None
    v_cache_type: str | None = None
    ignore_unsupported_runner: bool = False
    runtime_executable: Path = DEFAULT_EXECUTABLE
    runtime_version_policy: str = "warn"
    backend_priority: tuple[str, ...] = DEFAULT_BACKEND_PRIORITY
    memory_source: str = "auto"
    memory_compare: bool = False

    def __post_init__(self):
        object.__setattr__(self, "backend_priority", backend_priority(self.backend_priority))
        path = Path(self.runtime_executable)
        if not path.is_absolute() or "\0" in str(path):
            raise ConfigError("TLLAMA_RUNTIME_EXECUTABLE must be an absolute path")
        object.__setattr__(self, "runtime_executable", path)
        if self.runtime_version_policy not in ("warn", "strict"):
            raise ConfigError("TLLAMA_RUNTIME_VERSION_POLICY must be warn or strict")
        if self.memory_source not in ("auto", "native", "log"):
            raise ConfigError("TLLAMA_MEMORY_SOURCE must be auto, native or log")
        if type(self.memory_compare) is not bool:
            raise ConfigError("TLLAMA_MEMORY_COMPARE must be a boolean")


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 54800
    reload: bool = False
    debug: bool = False
    log_level: str = "info"
