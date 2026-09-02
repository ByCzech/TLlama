"""Turn origin patterns into the two forms Starlette's CORS middleware takes.

CORSMiddleware compares allow_origins by exact string equality, so a
pattern like 'http://localhost:*' would match nothing at all -- and a port
wildcard is exactly what a local development server needs, since the page
calling TLlama is on some port nobody can name in advance. The middleware's
other door is allow_origin_regex, which it applies with re.fullmatch.

So patterns are sorted into the two: the literal ones stay literal, and the
ones carrying a wildcard become one alternation. Everything but the
wildcard is escaped, so no character in an origin can be read as regex
syntax by accident.
"""

import re
from typing import Iterable, List, Optional, Tuple

_WILDCARD = "*"


def _pattern_to_regex(pattern: str) -> str:
    """Escape a pattern, leaving only its wildcards meaningful.

    '*' becomes '.*' rather than something narrower because it stands in
    for different things in different places -- a port in
    'http://localhost:*', a whole authority in 'vscode-webview://*'. An
    Origin header carries no path, so there is nothing below the authority
    for a greedy match to reach into.
    """
    return ".*".join(re.escape(part) for part in pattern.split(_WILDCARD))


def split_origin_patterns(
    patterns: Iterable[str],
) -> Tuple[List[str], Optional[str]]:
    """Return (exact origins, regex for the wildcard ones or None)."""
    exact: List[str] = []
    wildcards: List[str] = []

    for pattern in patterns:
        if _WILDCARD in pattern:
            wildcards.append(_pattern_to_regex(pattern))
        else:
            exact.append(pattern)

    if not wildcards:
        return exact, None

    return exact, "|".join(wildcards)
