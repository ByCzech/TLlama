import os

from dataclasses import dataclass

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


def _env_cache_type(name: str) -> str | None:
    return _env_str(name, "").strip().lower() or None


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
    flash_attention: bool = False

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
        flash_attention=_env_bool("TLLAMA_FLASH_ATTENTION", False),
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
