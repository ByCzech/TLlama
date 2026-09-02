from typing import Iterable

# Content types that mean "the client did not say what this is". A body sent
# with any of these is treated as JSON.
#
# curl -d without an explicit header sends application/x-www-form-urlencoded,
# and several HTTP clients default to text/plain. An absent header means the
# same thing. Anything else, notably a deliberate multipart/form-data, is left
# untouched so a real mistake still produces a clear error rather than a
# confusing JSON parse failure.
UNDECLARED_CONTENT_TYPES = frozenset({
    "",
    "application/x-www-form-urlencoded",
    "text/plain",
})

# Only methods that carry a request body are touched, so a plain GET is never
# given a header it did not have.
BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_CONTENT_TYPE = b"content-type"
_ORIGIN = b"origin"
_JSON_HEADER = (_CONTENT_TYPE, b"application/json")


def _base_content_type(value: bytes) -> str:
    """Strip parameters and normalise, so 'text/plain; charset=utf-8' matches."""
    return value.split(b";", 1)[0].strip().lower().decode("latin-1")


def _rewritten_headers(
    headers: Iterable[tuple[bytes, bytes]],
    index: int | None,
) -> list[tuple[bytes, bytes]]:
    headers = list(headers)

    if index is None:
        return [*headers, _JSON_HEADER]

    return [*headers[:index], _JSON_HEADER, *headers[index + 1:]]


class UndeclaredJsonBodyMiddleware:
    """Accept a JSON request body whatever the client says the content type is.

    Ollama is built on Gin and reads request bodies with ShouldBindJSON, which
    ignores Content-Type entirely. Its own documentation shows

        curl http://localhost:11434/api/generate -d '{ "model": ... }'

    with no header at all, which curl sends as application/x-www-form-urlencoded.
    FastAPI parses a body as JSON only when the content type is application/json
    or a +json subtype, so exactly the requests shown in the Ollama
    documentation would fail against TLlama while working against Ollama.

    Rewriting the header before routing keeps that leniency without giving up
    schema validation in the routes themselves. FastAPI's own
    strict_content_type switch is not enough here: it only covers a completely
    absent header, not the form-urlencoded one that curl actually sends.

    This is plain ASGI rather than BaseHTTPMiddleware on purpose, so streaming
    responses pass through untouched.

    Not applied to a request carrying an Origin header. Those three content
    types are exactly the ones that make a POST a "simple request" in the
    CORS sense: no preflight, so the request arrives and runs whatever the
    origin list says, and only the reply is withheld from the page. Any page
    a person visits could therefore drive inference or start a download here.
    Requiring a browser to declare application/json puts those requests back
    behind a preflight, where the origin list applies.

    Origin is the right discriminator because a browser always sends it on a
    cross-origin request and script cannot forge it, while curl -- the client
    this leniency exists for, and the one Ollama's own documentation shows --
    never sends it at all. So nothing that was accepted from a command line
    stops being accepted.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in BODY_METHODS:
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []

        index = None
        content_type = b""
        from_a_browser = False
        for position, (name, value) in enumerate(headers):
            name = name.lower()
            if name == _CONTENT_TYPE and index is None:
                index = position
                content_type = value
            elif name == _ORIGIN:
                from_a_browser = True

        if (
            not from_a_browser
            and _base_content_type(content_type) in UNDECLARED_CONTENT_TYPES
        ):
            scope = {**scope, "headers": _rewritten_headers(headers, index)}

        await self.app(scope, receive, send)
