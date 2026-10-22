"""Locale-aware human output; wire formats and model text remain unchanged."""

from datetime import datetime, timezone
import locale
import re


_TOKENS = re.compile(r'%[-_0^#]*\d*(?:E|O)?[A-Za-z%]|[^%]+|%')


def configure_locale():
    # Only an explicit client entry point changes these process-wide categories.
    for category in (locale.LC_NUMERIC, locale.LC_TIME):
        try:
            locale.setlocale(category, '')
        except locale.Error:
            locale.setlocale(category, 'C')


def number_text(value, digits=None):
    return locale.format_string('%d' if digits is None else f'%.{digits}f', value, grouping=True)


def _locale_format(name, fallback):
    try:
        return locale.nl_langinfo(getattr(locale, name)) or fallback
    except (AttributeError, ValueError):
        return fallback


def _expand(pattern, replacements):
    for _ in range(4):
        expanded = ''.join(replacements.get(token, token) for token in _TOKENS.findall(pattern))
        if expanded == pattern:
            break
        pattern = expanded
    return pattern


def _omit(pattern, codes, *, date=False):
    parts = _TOKENS.findall(pattern)
    while True:
        index = next((i for i, part in enumerate(parts)
                      if part.startswith('%') and part[-1] in codes), None)
        if index is None:
            return ''.join(parts).strip()
        parts.pop(index)
        # Remove a unit suffix such as 年/秒, keeping the preceding month/minute.
        suffix = (index < len(parts) and not parts[index].startswith('%')
                  and any(char.isalpha() for char in parts[index]))
        if suffix:
            parts.pop(index)
        if index and not parts[index - 1].startswith('%'):
            previous = parts[index - 1]
            # A trailing period is conventional in Czech/German day+month dates.
            if date and previous.strip() == '.':
                continue
            if date or not any(char.isalpha() for char in previous):
                parts.pop(index - 1)
        elif not suffix and index < len(parts) and not parts[index].startswith('%'):
            parts.pop(index)


def _time_format():
    pattern = _expand(_locale_format('T_FMT', '%H:%M:%S'), {
        '%T': '%H:%M:%S', '%R': '%H:%M',
        '%r': _locale_format('T_FMT_AMPM', '%I:%M:%S %p'),
    })
    if '%X' in pattern or not any(token.endswith('M') and token.startswith('%')
                                   for token in _TOKENS.findall(pattern)):
        pattern = '%H:%M:%S'
    return _omit(pattern, {'S', 'z', 'Z'})


def timestamp_text(value, *, now=None):
    """Local minute precision, omitting today's date or this year's year."""
    if not isinstance(value, str):
        return 'unknown'
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            return 'unknown'
        stamp = stamp.astimezone()
        today = (now if now is not None else datetime.now(timezone.utc)).astimezone()
        clock = stamp.strftime(_time_format())
        if stamp.date() == today.date():
            return clock
        pattern = _expand(_locale_format('D_FMT', '%m/%d/%y'), {
            '%D': '%m/%d/%y', '%F': '%Y-%m-%d',
        })
        if stamp.year == today.year:
            pattern = _omit(pattern, {'Y', 'y', 'C'}, date=True)
        return f'{stamp.strftime(pattern)} {clock}'
    except (ValueError, OverflowError, OSError):
        return 'unknown'
