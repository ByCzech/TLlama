import json

import pytest

from tllama.errors import (
    ollama_stream_error_line,
    openai_error_type,
    openai_stream_error_frame,
)


def as_an_ollama_client_reads_it(body_text):
    """The extraction ollama-python performs on an error body."""
    try:
        return json.loads(body_text).get("error", body_text)
    except json.JSONDecodeError:
        return body_text


@pytest.mark.parametrize(
    "method, path, payload, status",
    [
        ("POST", "/api/generate", {"prompt": "x"}, 400),
        ("POST", "/api/chat", {"messages": []}, 400),
        ("POST", "/api/show", {}, 400),
        ("POST", "/api/show", {"name": "absent/model/here"}, 404),
        ("GET", "/api/no-such-route", None, 404),
    ],
)
def test_api_errors_are_a_flat_string_under_error(client, method, path, payload, status):
    response = client.request(method, path, json=payload)
    body = response.json()

    assert response.status_code == status
    assert list(body) == ["error"]
    assert isinstance(body["error"], str)
    assert as_an_ollama_client_reads_it(response.text) == body["error"]


def test_a_schema_violation_on_api_is_a_bad_request(client):
    """Ollama answers 400 where FastAPI would default to 422."""
    assert client.post("/api/generate", json={"prompt": "x"}).status_code == 400


@pytest.mark.parametrize(
    "payload, status, error_type, param",
    [
        ({"messages": [{"role": "user", "content": "x"}]}, 400, "invalid_request_error", "model"),
        ({"model": "m", "messages": "not-a-list"}, 400, "invalid_request_error", "messages"),
        ({"model": "absent/model/here", "messages": []}, 404, "not_found_error", None),
    ],
)
def test_v1_errors_carry_the_openai_object(client, payload, status, error_type, param):
    response = client.post("/v1/chat/completions", json=payload)
    error = response.json()["error"]

    assert response.status_code == status
    assert sorted(error) == ["code", "message", "param", "type"]
    assert error["type"] == error_type
    assert error["param"] == param


def test_the_two_surfaces_keep_different_shapes(client):
    """One server, two conventions, because two different clients read them."""
    ollama = client.post("/api/show", json={}).json()
    openai = client.post("/v1/chat/completions", json={}).json()

    assert isinstance(ollama["error"], str)
    assert isinstance(openai["error"], dict)


@pytest.mark.parametrize(
    "status, expected",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_openai_error_type_follows_the_status(status, expected):
    assert openai_error_type(status) == expected


def test_a_stream_error_line_uses_the_key_the_client_looks_for():
    """ollama-python checks part.get('error'); a status prefix is invisible to it."""
    line = ollama_stream_error_line("boom")

    assert line.endswith("\n")
    assert json.loads(line) == {"error": "boom"}


def test_a_stream_error_frame_is_a_plain_sse_data_frame():
    """The OpenAI client inspects data frames carrying no event line."""
    frame = openai_stream_error_frame("boom", status_code=500)

    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert "event:" not in frame

    payload = json.loads(frame[len("data: "):])
    assert sorted(payload["error"]) == ["code", "message", "param", "type"]
    assert payload["error"]["message"] == "boom"
