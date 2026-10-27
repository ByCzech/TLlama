"""One admission policy for native runtime identities across worker modes."""

import logging
from collections.abc import Iterable


logger = logging.getLogger(__name__)


class RuntimeFingerprintMismatch(ValueError):
    """The observed native build is outside a strict caller's expected set."""


def evaluate_runtime_fingerprint(observed: str | None, *, expected_fingerprints: Iterable[str],
                                 policy: str, context: str = 'native worker') -> bool:
    """Return match status; warn permits unknown builds, strict refuses them.

    The caller supplies feature-specific audited identities or an explicit
    diagnostic pin. A matching pin does not promote a build into an audit list.
    Protocol validity and cross-endpoint identity consistency are separate.
    """
    if policy not in ('warn', 'strict'):
        raise ValueError('version_policy must be warn or strict')
    if isinstance(expected_fingerprints, str):
        raise ValueError('expected_fingerprints must be a nonempty collection of identities')
    expected = frozenset(expected_fingerprints)
    if not expected or any(not isinstance(value, str) or not value.strip()
                           or value != value.strip() or '\0' in value for value in expected):
        raise ValueError('expected_fingerprints must contain nonblank identities')
    matches = isinstance(observed, str) and observed in expected
    if not matches:
        message = (f'Unsupported runtime fingerprint: {observed!r}; '
                   f'expected {", ".join(sorted(expected))} ({context})')
        if policy == 'strict':
            raise RuntimeFingerprintMismatch(message)
        logger.warning('Unverified native runtime fingerprint %r; expected %s (%s); '
                       'continuing with version policy warn', observed, ', '.join(sorted(expected)), context)
    return matches
