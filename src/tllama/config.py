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


def _parse_host_port(
    name: str, value: str, default_host: str, default_port: int
) -> tuple[str, int]:
    if not value or value.strip() == "":
        return default_host, default_port

    raw = value.strip()

    if ":" not in raw:
        return raw, default_port

    host, port_str = raw.rsplit(":", 1)

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
    kv_cache_type: str | None = None


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
        kv_cache_type=(
            _env_str("TLLAMA_KV_CACHE_TYPE", "").strip().lower() or None
        ),
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
