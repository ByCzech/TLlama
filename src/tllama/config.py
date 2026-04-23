import os

from dataclasses import dataclass

DEFAULT_MODELS_DIR = "/var/lib/tllama/models"


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
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


def _parse_host_port(value: str, default_host: str, default_port: int) -> tuple[str, int]:
    if not value or value.strip() == "":
        return default_host, default_port

    raw = value.strip()

    if ":" not in raw:
        return raw, default_port

    host, port_str = raw.rsplit(":", 1)

    try:
        port = int(port_str)
    except ValueError:
        return default_host, default_port

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
    )


def load_app_config_from_env() -> AppConfig:
    host, port = _parse_host_port(
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
