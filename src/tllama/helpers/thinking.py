"""Conservative thinking controls shared by native chat and generate rendering.

Only byte-identical, qualified templates establish a closed contract. A new or
edited template remains unknown, even if its model alias or markers look known.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import logging


logger = logging.getLogger(__name__)
EFFORTS = frozenset({'minimal', 'low', 'medium', 'high', 'xhigh', 'ultra', 'max'})


@dataclass(frozen=True)
class ThinkingProfile:
    name: str
    values: tuple[bool | str, ...]
    default: bool | str
    aliases: tuple[tuple[str, str], ...] = ()


# Sources and qualification are recorded in docs/thinking-controls.md. These
# are exact template fingerprints, not a model-name/architecture allowlist.
_BOOLEAN = ThinkingProfile('qualified on/off template', (False, True), True)
_GPT_OSS = ThinkingProfile('qualified GPT-OSS template', ('low', 'medium', 'high'), 'medium',
                          (('minimal', 'low'), ('xhigh', 'high'), ('ultra', 'high')))
_TEMPLATES = {
    '87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5': _BOOLEAN,
    'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715': _BOOLEAN,
    'a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146': _GPT_OSS,
}


def thinking_profile(metadata_info=None):
    metadata = metadata_info or {}
    template = metadata.get('template')
    if not isinstance(template, str):
        return None
    profile = _TEMPLATES.get(sha256(template.encode('utf-8')).hexdigest())
    # Effort semantics here also rely on the documented GPT-OSS model contract.
    if profile is _GPT_OSS and metadata.get('arch') != 'gpt-oss':
        return None
    return profile


def normalize_think(value):
    if value is None or type(value) is bool:
        return value
    if not isinstance(value, str):
        raise ValueError('think must be a boolean, string or None')
    value = value.strip().lower()
    if not value:
        return None
    if value in {'false', 'none'}:
        return False
    if value == 'true':
        return True
    return value


def _canonical(value, profile):
    """Compare supported aliases, never conflate fallback with equivalence."""
    value = normalize_think(value)
    if profile is None or value is None or type(value) is bool:
        return value
    if value in profile.values:  # Exact supported names win over aliases.
        return value
    if profile.values == (False, True) and value in EFFORTS:
        return True
    return dict(profile.aliases).get(value, value)


def resolve_think(value, metadata_info=None, *, warn=True):
    value = normalize_think(value)
    if value is None:
        return None
    profile = thinking_profile(metadata_info)
    value = _canonical(value, profile)
    if value is None or profile is None:
        return value
    if value in profile.values:
        return value
    if value is True:
        return profile.default
    if value is False:
        selected = profile.values[0]
        if warn:
            logger.warning('Thinking cannot be disabled for %s; using lowest value %r', profile.name, selected)
        return selected
    if warn:
        logger.warning('Thinking value %r is not supported by %s; using model default %r',
                       value, profile.name, profile.default)
    return profile.default


def openai_effort_values(request):
    """Validate both controls even if one would lose precedence, before load."""
    flat = request.reasoning_effort
    nested = request.reasoning.get('effort') if isinstance(request.reasoning, dict) else None
    for value in (flat, nested):
        if value is not None and not isinstance(value, str):
            raise ValueError('reasoning effort must be a string or null')
    return normalize_think(flat), normalize_think(nested)


def select_openai_think(request, metadata_info=None):
    flat, nested = openai_effort_values(request)
    fields_set = request.model_fields_set
    both = 'reasoning' in fields_set and 'reasoning_effort' in fields_set
    profile = thinking_profile(metadata_info) if both else None
    selected, source = (nested, 'reasoning.effort') if nested is not None else (flat, 'reasoning_effort')
    if both and _canonical(flat, profile) != _canonical(nested, profile):
        logger.warning('Different reasoning.effort=%r and reasoning_effort=%r for model %r; using %s=%r',
                       nested, flat, request.model, source, selected)
    return selected
