import pytest

from tllama.helpers.common import DEFAULT_KEEP_ALIVE_SECONDS, normalize_keep_alive
from tllama.schemas.ollama import OllamaChatRequest, OllamaGenerateRequest


@pytest.mark.parametrize(
    "value, expected",
    [
        ("5m", 300),
        ("30s", 30),
        ("1h", 3600),
        ("0.5h", 1800),
        (3600, 3600),
        ("3600", 3600),
        (0, 0),
        ("0", 0),
        (None, DEFAULT_KEEP_ALIVE_SECONDS),
        ("", DEFAULT_KEEP_ALIVE_SECONDS),
    ],
)
def test_normalize_keep_alive_accepts_ollama_syntax(value, expected):
    assert normalize_keep_alive(value) == expected


@pytest.mark.parametrize("value", [-1, "-1", "-1m", -0.5])
def test_negative_keep_alive_means_infinite(value):
    assert normalize_keep_alive(value) is None


@pytest.mark.parametrize("value", ["abc", "5x", "m", "--"])
def test_malformed_keep_alive_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_keep_alive(value)


@pytest.mark.parametrize(
    "configured, expected",
    [("5m", 300), ("30s", 30), ("-1", None), ("0", 0)],
)
def test_configured_keep_alive_applies_when_the_client_is_silent(
    make_manager, configured, expected
):
    manager = make_manager(keep_alive=configured)

    assert manager.resolve_keep_alive(None) == expected


@pytest.mark.parametrize(
    "configured, requested, expected",
    [("30s", "10m", 600), ("-1", 0, 0), ("5m", -1, None)],
)
def test_request_overrides_the_configured_default(
    make_manager, configured, requested, expected
):
    manager = make_manager(keep_alive=configured)

    assert manager.resolve_keep_alive(requested) == expected


@pytest.mark.parametrize("schema", [OllamaChatRequest, OllamaGenerateRequest])
def test_an_omitted_keep_alive_reaches_the_backend_as_none(schema):
    """None is the only signal that lets the configured default apply."""
    payload = {"messages": []} if schema is OllamaChatRequest else {"prompt": "x"}

    assert schema(model="m", **payload).keep_alive is None


@pytest.mark.parametrize("schema", [OllamaChatRequest, OllamaGenerateRequest])
@pytest.mark.parametrize("value", ["10m", 0, -1])
def test_an_explicit_keep_alive_survives_the_schema(schema, value):
    payload = {"messages": []} if schema is OllamaChatRequest else {"prompt": "x"}

    assert schema(model="m", keep_alive=value, **payload).keep_alive == value
