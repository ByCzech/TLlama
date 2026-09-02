import os
import re

from dataclasses import dataclass, field

from typing import Mapping

DEFAULT_MODELS_DIR = "/var/lib/tllama/models"


class ConfigError(ValueError):
    """An environment variable holds something that cannot be used.

    Every one of these used to be swallowed: a value that failed to parse
    returned the default and the server came up as if nothing had been
    set. For a service started from a systemd unit that is the worst of
    the available behaviours, because the only evidence of a typo is that
    the setting quietly does not apply. These are raised instead, and the
    entry point turns them into a message and a non-zero exit.
    """


def _reject(name: str, value: str, expected: str) -> "ConfigError":
    return ConfigError(f"{name}={value!r} is not valid: expected {expected}.")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _reject(name, value, "a whole number") from None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _reject(name, value, "a number") from None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise _reject(name, value, "one of 1/true/yes/on or 0/false/no/off")


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


RUNTIME_ENV_PREFIX = "TLLAMA_RUNTIME_"

# Variables kept from before the generic passthrough existed. Each names a
# Llama() parameter that TLLAMA_RUNTIME_<NAME> now also reaches; they stay
# so a working configuration keeps working, and because the alias is the
# spelling Ollama uses.
_RUNTIME_ALIASES = {
    "TLLAMA_FLASH_ATTENTION": "flash_attn",
}


def _collect_runtime_overrides() -> dict[str, str]:
    """Gather the raw strings; nothing is converted or checked here.

    Deliberately no llama_cpp: the entry point reads configuration before
    the library is loaded, and what a value has to become is decided from
    Llama()'s own signature, in a place that can see it.

    An explicit TLLAMA_RUNTIME_* wins over an alias for the same
    parameter, so the specific spelling beats the general one.
    """
    collected: dict[str, str] = {}

    for variable, parameter in _RUNTIME_ALIASES.items():
        value = os.getenv(variable)
        if value is not None and value.strip() != "":
            collected[parameter] = value

    for variable, value in os.environ.items():
        if not variable.startswith(RUNTIME_ENV_PREFIX):
            continue
        if value.strip() == "":
            continue
        collected[variable[len(RUNTIME_ENV_PREFIX):].lower()] = value

    return collected


def _env_cache_type(name: str) -> str | None:
    return _env_str(name, "").strip().lower() or None


# Origins a browser may reach TLlama from before TLLAMA_ORIGINS adds to
# them. Deliberately shorter than Ollama's equivalent list rather than a
# copy of it, because several of its entries do not apply here:
#
#   0.0.0.0    an address to listen on, not one to connect to. A browser
#              only ever sends it as an Origin if someone typed it into
#              the address bar.
#   file://    Chrome sends 'null' for a page opened off disk, not a
#              file:// origin, so the entry catches nothing there -- and
#              allowing 'null' instead would be far worse, since every
#              sandboxed iframe on the web gets that same origin.
#   app://     schemes belonging to Ollama's own desktop application.
#   tauri://   TLlama has no client of its own yet; when it gets one, its
#              scheme belongs here and not before.
#
# [::1] goes the other way: Ollama does not list it, and TLlama binds to
# IPv6 perfectly well, so a browser pointed at the loopback address it
# actually resolved to must not be turned away.
#
# vscode-webview:// and vscode-file:// stay. A VS Code extension draws its
# UI in a Chromium webview served under those schemes, so an extension
# panel calling a local server is a browser request with a real origin,
# and IDE integration is a stated goal rather than an inherited one. The
# guid in a webview origin differs per panel, hence the wildcard.
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


def resolve_cors_origins() -> tuple[str, ...]:
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

    for entry in _env_str("TLLAMA_ORIGINS", "").split(","):
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
    model_scan_timeout_seconds: float = 5.0
    metadata_cache_ttl_seconds: float = 300.0
    # Llama() keyword arguments set server-wide, still as strings: what
    # each has to become is read off Llama()'s signature, which this
    # module deliberately cannot see. A model's [runtime] overrides these.
    runtime_overrides: Mapping[str, str] = field(default_factory=dict)

    # kv_cache_type sets both sides; k_cache_type/v_cache_type override it
    # for one side each, mirroring type_kv/type_k/type_v in a .toml
    # [runtime]. The names are resolved together, by the same function that
    # resolves the .toml ones, so the precedence cannot drift apart from it.
    kv_cache_type: str | None = None
    k_cache_type: str | None = None
    v_cache_type: str | None = None


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 54800
    reload: bool = False
    debug: bool = False


def load_backend_config_from_env() -> BackendConfig:
    return BackendConfig(
        models_dir=_env_str("TLLAMA_MODELS", DEFAULT_MODELS_DIR),
        context_length=_env_int("TLLAMA_CONTEXT_LENGTH", 0),
        keep_alive=_env_str("TLLAMA_KEEP_ALIVE", "5m"),
        max_loaded_models=_env_int("TLLAMA_MAX_LOADED_MODELS", 1),
        janitor_interval_seconds=_env_float("TLLAMA_JANITOR_INTERVAL", 10.0),
        model_scan_timeout_seconds=_env_float("TLLAMA_MODEL_SCAN_TIMEOUT", 5.0),
        metadata_cache_ttl_seconds=_env_float("TLLAMA_METADATA_CACHE_TTL", 300.0),
        runtime_overrides=_collect_runtime_overrides(),
        kv_cache_type=_env_cache_type("TLLAMA_KV_CACHE_TYPE"),
        k_cache_type=_env_cache_type("TLLAMA_K_CACHE_TYPE"),
        v_cache_type=_env_cache_type("TLLAMA_V_CACHE_TYPE"),
    )


def load_app_config_from_env() -> AppConfig:
    host, port = _parse_host_port(
        "TLLAMA_HOST",
        _env_str("TLLAMA_HOST", "127.0.0.1:54800"),
        default_host="127.0.0.1",
        default_port=54800,
    )

    return AppConfig(
        host=host,
        port=port,
        reload=_env_bool("TLLAMA_APP_RELOAD", False),
        debug=_env_bool("TLLAMA_DEBUG", False),
    )
