import json
from datetime import datetime, timezone

import pytest

from tllama.helpers.common import NEVER_EXPIRES_SECONDS, never_expires_at


BODY = json.dumps({"model": "absent/model/here", "prompt": "", "keep_alive": -1})


def body_was_parsed(response):
    """A body that never became a dictionary produces that specific complaint."""
    return "valid dictionary" not in json.dumps(response.json())


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "application/x-www-form-urlencoded",
        "application/x-www-form-urlencoded; charset=utf-8",
        "text/plain",
        "text/plain; charset=utf-8",
        "application/json",
        "application/json; charset=utf-8",
        "application/vnd.api+json",
    ],
)
def test_a_json_body_is_accepted_however_it_is_declared(client, content_type):
    """curl -d sends form-urlencoded, and the Ollama documentation shows exactly that."""
    headers = {"Content-Type": content_type} if content_type else {}

    assert body_was_parsed(client.post("/api/generate", content=BODY, headers=headers))


UNDECLARED = [
    None,
    "application/x-www-form-urlencoded",
    "text/plain",
    "text/plain; charset=utf-8",
]


@pytest.mark.parametrize("content_type", UNDECLARED)
def test_a_browser_has_to_declare_json(client, content_type):
    """Those content types are exactly the ones that skip preflight.

    A POST carrying one of them is a "simple request" in the CORS sense:
    it arrives and runs whatever the origin list says, and only the reply
    is withheld from the page. Accepting them as JSON would let any page a
    person happens to visit drive inference or start a download. Requiring
    application/json puts the request back behind a preflight, where the
    origin list applies.
    """
    headers = {"Origin": "https://evil.example"}
    if content_type:
        headers["Content-Type"] = content_type

    response = client.post("/api/generate", content=BODY, headers=headers)

    assert response.status_code == 415


def test_the_refusal_names_the_thing_that_is_wrong(client):
    """Not a schema complaint about a body that was never the problem.

    Whether a body with no Content-Type reaches the route as JSON is
    FastAPI's decision -- strict_content_type -- and its default has
    differed between versions. Refusing here gives the same answer on
    every installation and one that can be acted on.
    """
    response = client.post(
        "/api/generate",
        content=BODY,
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )

    assert "Content-Type: application/json" in response.json()["error"]


def test_the_refusal_takes_the_shape_of_the_surface_addressed(client):
    """/v1 has its own error shape, and this runs before routing."""
    response = client.post(
        "/v1/chat/completions",
        content=BODY,
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert "Content-Type: application/json" in response.json()["error"]["message"]


def test_a_refused_browser_request_can_still_read_the_refusal(client):
    """An opaque network error would leave nothing to debug from."""
    response = client.post(
        "/api/generate",
        content=BODY,
        headers={"Origin": "http://localhost:3000", "Content-Type": "text/plain"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.parametrize("content_type", UNDECLARED)
def test_the_same_request_without_an_origin_is_still_accepted(client, content_type):
    """The two layers against each other, which is the whole rule.

    Origin is the discriminator because a browser always sends it on a
    request like this and script cannot forge it, while curl -- the client
    the leniency exists for, and the one Ollama's documentation shows --
    never sends it at all.
    """
    headers = {"Content-Type": content_type} if content_type else {}

    assert body_was_parsed(client.post("/api/generate", content=BODY, headers=headers))


def test_a_browser_declaring_json_is_served(client):
    """Not a block on browsers: a block on requests that dodge preflight."""
    response = client.post(
        "/api/generate",
        content=BODY,
        headers={"Origin": "http://localhost:3000", "Content-Type": "application/json"},
    )

    assert body_was_parsed(response)


def test_a_request_without_a_body_is_unaffected_by_an_origin(client):
    assert client.get("/api/tags", headers={"Origin": "http://localhost:3000"}).status_code == 200


def test_a_deliberate_multipart_body_is_left_alone(client):
    """Rewriting it would turn a clear mistake into a confusing parse error."""
    response = client.post(
        "/api/generate",
        content=BODY,
        headers={"Content-Type": "multipart/form-data; boundary=xyz"},
    )

    assert not body_was_parsed(response)


def test_a_body_that_really_is_not_json_still_fails(client):
    response = client.post(
        "/api/generate",
        content="model=x&prompt=y",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/", "/health", "/api/tags", "/api/version", "/v1/models"])
def test_requests_without_a_body_are_untouched(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize(
    "relative_path, expected_reference",
    [
        ("Local/plain.gguf", "plain"),
        ("TLlama/ByCzech/qwen3.gguf", "ByCzech/qwen3"),
        ("HuggingFace/ns/repo/qwen3.gguf", "ns/repo/qwen3"),
        ("HuggingFace/ns/repo/sub/qwen3.gguf", "ns/repo/sub/qwen3"),
    ],
)
def test_a_stored_model_resolves_back_to_its_own_path(
    manager, gguf_file, relative_path, expected_reference
):
    """One segment is Local, two are TLlama, three or more are HuggingFace.

    Naming only -- whether that reference is actually loadable now depends
    on a matching .toml (tests/test_virtual_model_scan.py), which is a
    separate concern from the naming scheme itself.
    """
    path = gguf_file(relative_path)

    reference = manager._build_model_ref_from_path(path)

    assert reference == expected_reference
    assert manager.resolve_model_storage_path(reference) == path


def test_the_never_expires_horizon_matches_the_one_ollama_reports():
    """Ollama turns a negative keep_alive into math.MaxInt64 nanoseconds."""
    assert NEVER_EXPIRES_SECONDS == (2 ** 63 - 1) // 1_000_000_000


def test_a_pinned_model_is_reported_far_enough_ahead_to_read_as_forever():
    """The Ollama CLI prints Forever beyond twenty years and Stopping... for the past."""
    ahead = datetime.fromisoformat(never_expires_at()) - datetime.now(timezone.utc)

    assert ahead.days / 365 > 21


def test_the_janitor_never_reaps_a_model_without_an_expiry(manager):
    assert manager._is_model_entry_expired({"expires_at": None}) is False
    assert manager._is_model_entry_expired({"expires_at": "2000-01-01T00:00:00+00:00"}) is True

