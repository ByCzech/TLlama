"""A browser has to be able to reach TLlama, and only from known origins.

Checked through the running application rather than against the middleware's
arguments: what decides is the headers a browser actually receives, and the
translation from the configured patterns to what Starlette compares against
is exactly the step that could be wrong while the arguments look right.
"""

import pytest

from tllama.config import BUILTIN_CORS_ORIGINS, ConfigError, resolve_cors_origins
from tllama.helpers.cors import split_origin_patterns


def preflight(client, origin, method="POST", path="/api/tags"):
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "Content-Type",
        },
    )


class TestWhoGetsIn:
    def test_a_local_page_is_answered(self, client):
        response = preflight(client, "http://localhost")

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost"

    def test_a_local_page_on_any_port_is_answered(self, client):
        """The case the whole wildcard translation exists for.

        A development server picks its own port, so no list written in
        advance can name it. Exact matching -- what Starlette does with
        allow_origins -- would turn this away.
        """
        response = preflight(client, "http://localhost:5173")

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_the_ipv6_loopback_is_answered(self, client):
        response = preflight(client, "http://[::1]:8080")

        assert response.headers["access-control-allow-origin"] == "http://[::1]:8080"

    def test_an_editor_webview_is_answered(self, client):
        response = preflight(client, "vscode-webview://0e9f1a2b-3c4d")

        assert "access-control-allow-origin" in response.headers

    def test_a_page_from_anywhere_else_is_not(self, client):
        response = preflight(client, "https://evil.example")

        assert "access-control-allow-origin" not in response.headers

    def test_a_null_origin_is_not_answered(self, client):
        """Every sandboxed iframe on the web sends this one.

        Allowing it would be indistinguishable from allowing everything,
        which is why file:// is not in the built-in list.
        """
        response = preflight(client, "null")

        assert "access-control-allow-origin" not in response.headers

    def test_a_lookalike_host_is_not_answered(self, client):
        """The wildcard must not reach past the part it stands for."""
        response = preflight(client, "http://localhost.evil.example")

        assert "access-control-allow-origin" not in response.headers


class TestWhatIsAllowed:
    def test_credentials_are_never_granted(self, client):
        response = preflight(client, "http://localhost")

        assert "access-control-allow-credentials" not in response.headers

    def test_the_openai_sdk_headers_pass_preflight(self, client):
        """Thirteen of them, and the SDK fails outright without them."""
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, x-stainless-lang, x-stainless-retry-count",
            },
        )

        assert response.status_code == 200
        allowed = response.headers["access-control-allow-headers"].lower()
        assert "x-stainless-lang" in allowed
        assert "x-stainless-retry-count" in allowed

    def test_deleting_a_model_is_allowed(self, client):
        response = preflight(client, "http://localhost", method="DELETE", path="/api/delete")

        assert response.headers["access-control-allow-methods"].upper().count("DELETE")

    def test_a_method_no_route_answers_is_not_allowed(self, client):
        response = preflight(client, "http://localhost", method="PUT")

        assert "PUT" not in response.headers["access-control-allow-methods"].upper()

    def test_the_preflight_is_not_cached_for_long(self, client):
        """A cached preflight outlives a change to the origin list."""
        response = preflight(client, "http://localhost")

        assert int(response.headers["access-control-max-age"]) <= 600

    def test_a_real_response_carries_the_origin_back(self, client):
        """Preflight passing is not the same as the answer being readable."""
        response = client.get("/api/tags", headers={"Origin": "http://localhost:5173"})

        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


class TestConfiguredOrigins:
    def test_a_configured_origin_is_added_to_the_built_in_ones(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_ORIGINS", "https://ui.example")

        origins = resolve_cors_origins()

        assert "https://ui.example" in origins
        for builtin in BUILTIN_CORS_ORIGINS:
            assert builtin in origins

    def test_several_are_split_on_commas(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_ORIGINS", "https://a.example, https://b.example")

        origins = resolve_cors_origins()

        assert {"https://a.example", "https://b.example"} <= set(origins)

    def test_empty_entries_are_formatting_not_a_mistake(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_ORIGINS", "https://a.example,,")

        assert "https://a.example" in resolve_cors_origins()

    def test_nothing_configured_leaves_the_built_in_list(self, monkeypatch):
        monkeypatch.delenv("TLLAMA_ORIGINS", raising=False)

        assert resolve_cors_origins() == BUILTIN_CORS_ORIGINS

    def test_a_repeated_origin_is_listed_once(self, monkeypatch):
        monkeypatch.setenv("TLLAMA_ORIGINS", "http://localhost")

        origins = resolve_cors_origins()

        assert origins.count("http://localhost") == 1

    @pytest.mark.parametrize(
        "value",
        [
            "localhost:3000",
            "https://ui.example/app",
            "https://ui.example?x=1",
            "not an origin",
        ],
    )
    def test_something_a_browser_could_never_send_stops_the_start(self, monkeypatch, value):
        """A setting that silently does nothing is worse than a refusal."""
        monkeypatch.setenv("TLLAMA_ORIGINS", value)

        with pytest.raises(ConfigError) as raised:
            resolve_cors_origins()

        # Naming the variable is the point: guessing it from the text of
        # somebody else's exception does not work.
        assert "TLLAMA_ORIGINS" in str(raised.value)
        assert value in str(raised.value)


class TestPatternTranslation:
    def test_a_literal_pattern_stays_literal(self):
        exact, regex = split_origin_patterns(["http://localhost"])

        assert exact == ["http://localhost"]
        assert regex is None

    def test_a_dot_in_a_host_is_not_a_wildcard(self):
        """Unescaped, 'a.example' would match 'aXexample' too."""
        import re

        _, regex = split_origin_patterns(["https://a.example:*"])

        assert re.fullmatch(regex, "https://a.example:1") is not None
        assert re.fullmatch(regex, "https://aXexample:1") is None
